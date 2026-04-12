import argparse
import os
import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.optim as optim
from tqdm.auto import tqdm

from app.eval import evaluate_multi_seed, greedy_rollout, save_eval_results
from app.helpers import double_dqn_update, epsilon_by_step
from app.open_spiel_2048_env import OpenSpiel2048Env
from app.q_network import QNetwork
from app.replay_buffer import ReplayBuffer, make_legal_mask

MetaTransition = namedtuple("MetaTransition", ["obs", "goal_idx", "reward", "next_obs", "done"])


def parse_goal_tiles(goal_tiles_arg: str):
    goals = [int(v.strip()) for v in goal_tiles_arg.split(",") if v.strip()]
    if not goals:
        raise ValueError("goal_tiles cannot be empty")
    return sorted(goals)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate H-DQN on OpenSpiel 2048")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for Python, NumPy, and PyTorch.")
    parser.add_argument(
        "--num_episodes",
        dest="num_episodes",
        type=int,
        default=300,
        help="Number of training episodes to run.",
    )
    parser.add_argument(
        "--goal_tiles",
        type=str,
        default="32,64,128,256,512,1024,2048",
        help="Comma-separated intrinsic goal tiles for the meta-controller.",
    )
    parser.add_argument(
        "--option_duration",
        type=int,
        default=8,
        help="Maximum primitive steps before forcing a new meta-goal.",
    )
    parser.add_argument(
        "--intrinsic_success_reward",
        type=float,
        default=1.0,
        help="Intrinsic reward when the selected goal is achieved.",
    )
    parser.add_argument(
        "--intrinsic_step_penalty",
        type=float,
        default=-0.01,
        help="Intrinsic step penalty while pursuing a goal.",
    )

    parser.add_argument("--ctrl_buffer_size", type=int, default=50_000, help="Controller replay capacity.")
    parser.add_argument("--ctrl_batch_size", type=int, default=128, help="Controller mini-batch size.")
    parser.add_argument("--ctrl_gamma", type=float, default=0.99, help="Controller discount factor.")
    parser.add_argument("--ctrl_lr", type=float, default=1e-3, help="Controller learning rate.")
    parser.add_argument(
        "--ctrl_target_sync_every",
        type=int,
        default=250,
        help="Controller target sync interval in env steps.",
    )
    parser.add_argument(
        "--ctrl_learn_start",
        type=int,
        default=1_000,
        help="Controller replay warm-up size.",
    )
    parser.add_argument(
        "--ctrl_learn_every",
        type=int,
        default=4,
        help="Controller optimization interval in env steps.",
    )

    parser.add_argument("--meta_buffer_size", type=int, default=20_000, help="Meta replay capacity.")
    parser.add_argument("--meta_batch_size", type=int, default=64, help="Meta mini-batch size.")
    parser.add_argument("--meta_gamma", type=float, default=0.99, help="Meta-controller discount factor.")
    parser.add_argument("--meta_lr", type=float, default=1e-3, help="Meta-controller learning rate.")
    parser.add_argument(
        "--meta_target_sync_every",
        type=int,
        default=50,
        help="Meta target sync interval in completed options.",
    )
    parser.add_argument(
        "--meta_learn_start",
        type=int,
        default=500,
        help="Meta replay warm-up size.",
    )
    parser.add_argument(
        "--meta_learn_every",
        type=int,
        default=2,
        help="Meta optimization interval in completed options.",
    )

    parser.add_argument("--eps_start", type=float, default=1.0, help="Initial epsilon for controller exploration.")
    parser.add_argument("--eps_end", type=float, default=0.05, help="Final epsilon for controller exploration.")
    parser.add_argument(
        "--eps_decay_steps",
        type=int,
        default=20_000,
        help="Controller epsilon decay horizon in env steps.",
    )
    parser.add_argument("--meta_eps_start", type=float, default=1.0, help="Initial epsilon for meta-goal exploration.")
    parser.add_argument("--meta_eps_end", type=float, default=0.05, help="Final epsilon for meta-goal exploration.")
    parser.add_argument(
        "--meta_eps_decay_steps",
        type=int,
        default=2_000,
        help="Meta epsilon decay horizon in completed options.",
    )

    parser.add_argument(
        "--max_steps_per_episode",
        type=int,
        default=5_000,
        help="Cap on environment steps per episode.",
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=10.0,
        help="Global gradient-norm clipping threshold.",
    )
    parser.add_argument(
        "--num_eval_seeds",
        type=int,
        default=100,
        help="Number of random seeds for final greedy evaluation.",
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=20,
        help="Evaluate greedy policy every N training episodes.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional directory for checkpoint and evaluation artifacts. If omitted, no files are saved.",
    )
    return parser


def normalize_goal(goal_tile: int) -> float:
    # log2(goal)/11 keeps values in a compact range roughly [0, 1].
    return float(np.log2(max(2, goal_tile)) / 11.0)


def augment_obs_with_goal(obs, goal_tile: int):
    goal_feat = normalize_goal(goal_tile)
    return np.concatenate([np.asarray(obs, dtype=np.float32), np.array([goal_feat], dtype=np.float32)], axis=0)


@torch.no_grad()
def select_meta_goal(
    meta_net: QNetwork,
    obs,
    goals,
    epsilon: float,
    device: torch.device,
):
    if random.random() < epsilon:
        return random.randrange(len(goals))
    obs_t = torch.tensor(np.asarray([obs]), dtype=torch.float32, device=device)
    q_goals = meta_net(obs_t).squeeze(0)
    return int(torch.argmax(q_goals).item())


@torch.no_grad()
def select_controller_action(
    controller_net: QNetwork,
    obs,
    goal_tile: int,
    legal_actions_list,
    num_actions: int,
    epsilon: float,
    device: torch.device,
):
    if random.random() < epsilon:
        return random.choice(legal_actions_list)

    obs_aug = augment_obs_with_goal(obs, goal_tile)
    obs_t = torch.tensor(obs_aug, dtype=torch.float32, device=device).unsqueeze(0)
    q = controller_net(obs_t).squeeze(0)

    legal_mask = torch.zeros(num_actions, dtype=torch.bool, device=device)
    legal_mask[legal_actions_list] = True
    q_masked = q.masked_fill(~legal_mask, -1e9)
    return int(torch.argmax(q_masked).item())


def achieved_goal(obs, next_obs, goal_tile: int) -> bool:
    current_max = int(np.max(obs)) if obs is not None and len(obs) > 0 else 0
    next_max = int(np.max(next_obs)) if next_obs is not None and len(next_obs) > 0 else 0
    return current_max < goal_tile <= next_max


class MetaReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def add(self, *args):
        self.buffer.append(MetaTransition(*args))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        return MetaTransition(*zip(*batch))


def meta_dqn_update(
    batch,
    meta_net: QNetwork,
    target_meta_net: QNetwork,
    optimizer: optim.Optimizer,
    gamma: float,
    grad_clip: float,
    device: torch.device,
) -> float:
    obs = torch.tensor(np.asarray(batch.obs), dtype=torch.float32, device=device)
    goal_indices = torch.tensor(batch.goal_idx, dtype=torch.int64, device=device).unsqueeze(1)
    rewards = torch.tensor(batch.reward, dtype=torch.float32, device=device)
    next_obs = torch.tensor(np.asarray(batch.next_obs), dtype=torch.float32, device=device)
    dones = torch.tensor(batch.done, dtype=torch.float32, device=device)

    q_values = meta_net(obs)
    q_sa = q_values.gather(1, goal_indices).squeeze(1)

    with torch.no_grad():
        next_online_q = meta_net(next_obs)
        next_actions = torch.argmax(next_online_q, dim=1, keepdim=True)
        next_target_q = target_meta_net(next_obs).gather(1, next_actions).squeeze(1)
        next_target_q = torch.where(dones > 0.5, torch.zeros_like(next_target_q), next_target_q)
        target = rewards + gamma * next_target_q

    loss = torch.nn.functional.mse_loss(q_sa, target)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(meta_net.parameters(), grad_clip)
    optimizer.step()

    return float(loss.item())


class HDQNGreedyPolicy(torch.nn.Module):
    """Adapter that exposes hierarchical policy as a flat Q-value model for evaluation."""

    def __init__(self, meta_net: QNetwork, controller_net: QNetwork, goals):
        super().__init__()
        self.meta_net = meta_net
        self.controller_net = controller_net
        self.goals = list(goals)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        goal_idx = torch.argmax(self.meta_net(x), dim=1)
        goal_values = torch.tensor(
            [normalize_goal(self.goals[int(i.item())]) for i in goal_idx],
            dtype=x.dtype,
            device=x.device,
        ).unsqueeze(1)
        x_aug = torch.cat([x, goal_values], dim=1)
        return self.controller_net(x_aug)


def train(args: argparse.Namespace, device: torch.device):
    log_episodes = bool(getattr(args, "log_episodes", True))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    goals = parse_goal_tiles(args.goal_tiles)

    train_env = OpenSpiel2048Env(seed=args.seed)
    obs_dim = train_env.obs_dim
    num_actions = train_env.num_actions

    controller_obs_dim = obs_dim + 1
    controller_net = QNetwork(controller_obs_dim, num_actions).to(device)
    target_controller_net = QNetwork(controller_obs_dim, num_actions).to(device)
    target_controller_net.load_state_dict(controller_net.state_dict())
    target_controller_net.eval()

    meta_net = QNetwork(obs_dim, len(goals)).to(device)
    target_meta_net = QNetwork(obs_dim, len(goals)).to(device)
    target_meta_net.load_state_dict(meta_net.state_dict())
    target_meta_net.eval()

    controller_optimizer = optim.Adam(controller_net.parameters(), lr=args.ctrl_lr)
    meta_optimizer = optim.Adam(meta_net.parameters(), lr=args.meta_lr)

    controller_replay = ReplayBuffer(args.ctrl_buffer_size)
    meta_replay = MetaReplayBuffer(args.meta_buffer_size)

    q_policy = HDQNGreedyPolicy(meta_net, controller_net, goals)

    global_step = 0
    option_count = 0

    for episode in tqdm(range(1, args.num_episodes + 1), desc="Training"):
        obs = train_env.reset(seed=args.seed + episode)
        done = False
        ep_return = 0.0
        ep_len = 0
        max_tile = 0
        illegal_attempts = 0
        ctrl_losses = []
        meta_losses = []

        current_goal_idx = None
        current_goal_tile = None
        option_start_obs = None
        option_ext_return = 0.0
        option_steps = 0

        while not done and ep_len < args.max_steps_per_episode:
            if current_goal_idx is None:
                meta_eps = epsilon_by_step(
                    option_count,
                    eps_start=args.meta_eps_start,
                    eps_end=args.meta_eps_end,
                    eps_decay_steps=args.meta_eps_decay_steps,
                )
                current_goal_idx = select_meta_goal(meta_net, obs, goals, meta_eps, device)
                current_goal_tile = goals[current_goal_idx]
                option_start_obs = np.asarray(obs, dtype=np.float32)
                option_ext_return = 0.0
                option_steps = 0

            if current_goal_tile is None:
                raise RuntimeError("Internal error: current_goal_tile must be set before controller step")

            ctrl_eps = epsilon_by_step(
                global_step,
                eps_start=args.eps_start,
                eps_end=args.eps_end,
                eps_decay_steps=args.eps_decay_steps,
            )

            legal = train_env.legal_actions()
            legal_mask = make_legal_mask(num_actions, legal)

            if obs is not None and len(obs) > 0:
                max_tile = max(max_tile, int(np.max(obs)))

            with torch.no_grad():
                obs_aug_t = torch.tensor(
                    np.asarray([augment_obs_with_goal(obs, current_goal_tile)]),
                    dtype=torch.float32,
                    device=device,
                )
                q_vals_np = controller_net(obs_aug_t).cpu().numpy()[0]

            action = select_controller_action(
                controller_net=controller_net,
                obs=obs,
                goal_tile=current_goal_tile,
                legal_actions_list=legal,
                num_actions=num_actions,
                epsilon=ctrl_eps,
                device=device,
            )

            if ctrl_eps > 0 and len(legal) < num_actions:
                best_raw_action = int(np.argmax(q_vals_np))
                if best_raw_action not in legal:
                    illegal_attempts += 1

            next_obs, ext_reward, done, info = train_env.step(action)
            option_ext_return += ext_reward
            option_steps += 1

            goal_hit = achieved_goal(obs, next_obs, current_goal_tile)
            intrinsic_reward = args.intrinsic_success_reward if goal_hit else args.intrinsic_step_penalty

            ctrl_done = bool(done or goal_hit)
            next_legal = info["legal_actions"] if not done else []
            next_legal_mask = make_legal_mask(num_actions, next_legal)

            controller_replay.add(
                augment_obs_with_goal(obs, current_goal_tile),
                action,
                intrinsic_reward,
                augment_obs_with_goal(next_obs, current_goal_tile),
                ctrl_done,
                legal_mask,
                next_legal_mask,
            )

            obs = next_obs
            ep_return += ext_reward
            ep_len += 1
            global_step += 1

            option_terminated = bool(done or goal_hit or option_steps >= args.option_duration)
            if option_terminated:
                meta_replay.add(
                    option_start_obs,
                    current_goal_idx,
                    option_ext_return,
                    np.asarray(next_obs, dtype=np.float32),
                    done,
                )
                current_goal_idx = None
                current_goal_tile = None
                option_count += 1

                if len(meta_replay) >= args.meta_learn_start and option_count % args.meta_learn_every == 0:
                    meta_batch = meta_replay.sample(args.meta_batch_size)
                    loss = meta_dqn_update(
                        batch=meta_batch,
                        meta_net=meta_net,
                        target_meta_net=target_meta_net,
                        optimizer=meta_optimizer,
                        gamma=args.meta_gamma,
                        grad_clip=args.grad_clip,
                        device=device,
                    )
                    meta_losses.append(loss)

                if option_count % args.meta_target_sync_every == 0:
                    target_meta_net.load_state_dict(meta_net.state_dict())

            if len(controller_replay) >= args.ctrl_learn_start and global_step % args.ctrl_learn_every == 0:
                ctrl_batch = controller_replay.sample(args.ctrl_batch_size)
                loss = double_dqn_update(
                    batch=ctrl_batch,
                    q_net=controller_net,
                    target_net=target_controller_net,
                    optimizer=controller_optimizer,
                    gamma=args.ctrl_gamma,
                    grad_clip=args.grad_clip,
                    device=device,
                )
                ctrl_losses.append(loss)

            if global_step % args.ctrl_target_sync_every == 0:
                target_controller_net.load_state_dict(controller_net.state_dict())

        eval_return = float("nan")
        if episode % args.eval_every == 0:
            eval_return, _, _, _, _ = greedy_rollout(
                q_net=q_policy,
                env=OpenSpiel2048Env(seed=1000 + episode),
                num_actions=num_actions,
                max_steps=args.max_steps_per_episode,
                device=device,
            )

        ctrl_mean_loss = float(np.mean(ctrl_losses)) if ctrl_losses else float("nan")
        meta_mean_loss = float(np.mean(meta_losses)) if meta_losses else float("nan")
        if log_episodes:
            print(
                f"Episode {episode}/{args.num_episodes} | "
                f"return={ep_return:.1f} | "
                f"length={ep_len} | "
                f"ctrl_loss={ctrl_mean_loss:.4f} | "
                f"meta_loss={meta_mean_loss:.4f} | "
                f"max_tile={max_tile} | "
                f"illegal_attempts={illegal_attempts} | "
                f"eval_return={eval_return:.1f}"
            )

    return controller_net, target_controller_net, meta_net, target_meta_net, q_policy, goals, obs_dim, num_actions


def evaluate_policy(
    args: argparse.Namespace,
    q_policy: torch.nn.Module,
    num_actions: int,
    device: torch.device,
):
    """Run multi-seed evaluation and print summary metrics."""
    print_summary = bool(getattr(args, "print_eval_summary", True))
    eval_data = evaluate_multi_seed(
        q_net=q_policy,
        env_class=OpenSpiel2048Env,
        num_eval_seeds=args.num_eval_seeds,
        num_actions=num_actions,
        max_steps_per_episode=args.max_steps_per_episode,
        device=device,
        seed_offset=5000,
    )

    summary = eval_data["summary"]
    if print_summary:
        print(f"Results over {args.num_eval_seeds} seeds:")
        print(f"  Average return:           {summary['avg_return']:.1f} +/- {summary['std_return']:.1f}")
        print(f"  Average episode length:   {summary['avg_length']:.1f}")
        print(f"  Average max tile:         {summary['avg_max_tile']:.1f}")
        print(f"  Average illegal attempts: {summary['avg_illegal_action_attempts']:.1f}")
        print(f"  Total illegal attempts:   {summary['total_illegal_action_attempts']}")

    return eval_data


def save_artifacts(
    args: argparse.Namespace,
    controller_net: QNetwork,
    target_controller_net: QNetwork,
    meta_net: QNetwork,
    target_meta_net: QNetwork,
    goals,
    obs_dim: int,
    num_actions: int,
    eval_data,
):
    """Save checkpoint and evaluation files only when output_dir is explicitly provided."""
    log_save = bool(getattr(args, "log_save", True))
    if not args.output_dir:
        if log_save:
            print("Skipping save: pass --output_dir to persist checkpoint and evaluation files.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    npz_path, json_path = save_eval_results(
        eval_data=eval_data,
        output_dir=args.output_dir,
        num_actions=num_actions,
        obs_dim=obs_dim,
        source_notebook="H-DQN",
        checkpoint_file="dqn.pt",
    )
    if log_save:
        print(f"Saved evaluation rollout archive to: {npz_path}")
        print(f"Saved evaluation metadata to: {json_path}")

    checkpoint_path = os.path.join(args.output_dir, "dqn.pt")
    torch.save(
        {
            "controller_state_dict": controller_net.state_dict(),
            "target_controller_state_dict": target_controller_net.state_dict(),
            "meta_state_dict": meta_net.state_dict(),
            "target_meta_state_dict": target_meta_net.state_dict(),
            "goals": list(goals),
            "obs_dim": obs_dim,
            "num_actions": num_actions,
            "controller_obs_dim": obs_dim + 1,
        },
        checkpoint_path,
    )
    if log_save:
        print(f"Saved checkpoint to: {checkpoint_path}")


def main():
    args = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    (
        controller_net,
        target_controller_net,
        meta_net,
        target_meta_net,
        q_policy,
        goals,
        obs_dim,
        num_actions,
    ) = train(args, device)
    eval_data = evaluate_policy(args, q_policy, num_actions, device)
    save_artifacts(
        args,
        controller_net,
        target_controller_net,
        meta_net,
        target_meta_net,
        goals,
        obs_dim,
        num_actions,
        eval_data,
    )


if __name__ == "__main__":
    main()
