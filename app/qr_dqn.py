import argparse
import os
import random

import numpy as np
import torch
import torch.optim as optim
from tqdm.auto import tqdm

from app.eval import evaluate_multi_seed, greedy_rollout, save_eval_results
from app.helpers import epsilon_by_step
from app.open_spiel_2048_env import OpenSpiel2048Env
from app.qr_q_network import QuantileQNetwork
from app.replay_buffer import ReplayBuffer, make_legal_mask


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate QR-DQN on OpenSpiel 2048")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for Python, NumPy, and PyTorch.")
    parser.add_argument(
        "--num_episodes",
        dest="num_episodes",
        type=int,
        default=300,
        help="Number of training episodes to run.",
    )
    parser.add_argument("--buffer_size", type=int, default=50_000, help="Replay buffer capacity.")
    parser.add_argument("--batch_size", type=int, default=128, help="Mini-batch size for optimization.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for Bellman targets.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Adam optimizer.")
    parser.add_argument(
        "--target_sync_every",
        type=int,
        default=250,
        help="How many env steps between target-network syncs.",
    )
    parser.add_argument(
        "--learn_start",
        type=int,
        default=1_000,
        help="Minimum replay size before starting gradient updates.",
    )
    parser.add_argument(
        "--learn_every",
        type=int,
        default=4,
        help="Run one optimization step every N env steps.",
    )
    parser.add_argument("--eps_start", type=float, default=1.0, help="Initial epsilon for exploration.")
    parser.add_argument("--eps_end", type=float, default=0.05, help="Final epsilon after decay.")
    parser.add_argument(
        "--eps_decay_steps",
        type=int,
        default=20_000,
        help="Number of steps for linear epsilon decay.",
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
        "--num_quantiles",
        type=int,
        default=51,
        help="Number of quantile atoms in the distributional head.",
    )
    parser.add_argument(
        "--kappa",
        type=float,
        default=1.0,
        help="Huber threshold for quantile regression loss.",
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


class ExpectedQPolicy(torch.nn.Module):
    """Adapter so eval helpers can consume a quantile network as a Q-value model."""

    def __init__(self, quantile_net: QuantileQNetwork):
        super().__init__()
        self.quantile_net = quantile_net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.quantile_net.q_values(x)


@torch.no_grad()
def masked_greedy_quantile_action(
    q_net: QuantileQNetwork,
    obs,
    legal_actions_list,
    num_actions: int,
    epsilon: float,
    device: torch.device,
) -> int:
    if random.random() < epsilon:
        return random.choice(legal_actions_list)

    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    q = q_net.q_values(obs_t).squeeze(0)

    legal_mask = torch.zeros(num_actions, dtype=torch.bool, device=device)
    legal_mask[legal_actions_list] = True
    q_masked = q.masked_fill(~legal_mask, -1e9)
    return int(torch.argmax(q_masked).item())


def quantile_huber_loss(
    pred_quantiles: torch.Tensor,
    target_quantiles: torch.Tensor,
    taus: torch.Tensor,
    kappa: float,
) -> torch.Tensor:
    """Compute QR-DQN quantile Huber loss for [batch, num_quantiles] tensors."""
    td_error = target_quantiles.unsqueeze(1) - pred_quantiles.unsqueeze(2)
    abs_error = torch.abs(td_error)
    huber = torch.where(
        abs_error <= kappa,
        0.5 * td_error.pow(2),
        kappa * (abs_error - 0.5 * kappa),
    )
    quantile_weight = torch.abs(taus.view(1, -1, 1) - (td_error.detach() < 0).float())
    return (quantile_weight * huber).mean()


def qr_dqn_update(
    batch,
    q_net: QuantileQNetwork,
    target_net: QuantileQNetwork,
    optimizer: optim.Optimizer,
    gamma: float,
    grad_clip: float,
    kappa: float,
    device: torch.device,
) -> float:
    obs = torch.tensor(np.asarray(batch.obs), dtype=torch.float32, device=device)
    actions = torch.tensor(batch.action, dtype=torch.int64, device=device)
    rewards = torch.tensor(batch.reward, dtype=torch.float32, device=device)
    next_obs = torch.tensor(np.asarray(batch.next_obs), dtype=torch.float32, device=device)
    dones = torch.tensor(batch.done, dtype=torch.float32, device=device)
    next_legal_mask = torch.tensor(np.asarray(batch.next_legal_mask), dtype=torch.bool, device=device)

    batch_size = obs.shape[0]
    num_quantiles = q_net.num_quantiles

    all_quantiles = q_net(obs)
    action_idx = actions.view(batch_size, 1, 1).expand(batch_size, 1, num_quantiles)
    pred_quantiles = all_quantiles.gather(1, action_idx).squeeze(1)

    with torch.no_grad():
        next_online_q = q_net.q_values(next_obs).masked_fill(~next_legal_mask, -1e9)
        next_actions = torch.argmax(next_online_q, dim=1)

        next_action_idx = next_actions.view(batch_size, 1, 1).expand(batch_size, 1, num_quantiles)
        next_quantiles = target_net(next_obs).gather(1, next_action_idx).squeeze(1)
        next_quantiles = torch.where(dones.unsqueeze(1) > 0.5, torch.zeros_like(next_quantiles), next_quantiles)

        target_quantiles = rewards.unsqueeze(1) + gamma * next_quantiles

    taus = (
        (torch.arange(num_quantiles, dtype=torch.float32, device=device) + 0.5)
        / float(num_quantiles)
    )
    loss = quantile_huber_loss(pred_quantiles, target_quantiles, taus, kappa=kappa)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q_net.parameters(), grad_clip)
    optimizer.step()

    return float(loss.item())


def train(args: argparse.Namespace, device: torch.device):
    log_episodes = bool(getattr(args, "log_episodes", True))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_env = OpenSpiel2048Env(seed=args.seed)
    obs_dim = train_env.obs_dim
    num_actions = train_env.num_actions

    q_net = QuantileQNetwork(obs_dim, num_actions, num_quantiles=args.num_quantiles).to(device)
    target_net = QuantileQNetwork(obs_dim, num_actions, num_quantiles=args.num_quantiles).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=args.lr)
    replay = ReplayBuffer(args.buffer_size)

    q_policy = ExpectedQPolicy(q_net)

    global_step = 0
    for episode in tqdm(range(1, args.num_episodes + 1), desc="Training"):
        obs = train_env.reset(seed=args.seed + episode)
        done = False
        ep_return = 0.0
        ep_len = 0
        max_tile = 0
        illegal_attempts = 0
        ep_losses = []

        while not done and ep_len < args.max_steps_per_episode:
            eps = epsilon_by_step(
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
                q_vals = q_net.q_values(torch.tensor(np.asarray([obs]), dtype=torch.float32, device=device))
                q_vals_np = q_vals.cpu().numpy()[0]

            action = masked_greedy_quantile_action(
                q_net=q_net,
                obs=obs,
                legal_actions_list=legal,
                num_actions=num_actions,
                epsilon=eps,
                device=device,
            )

            if eps > 0 and len(legal) < num_actions:
                best_raw_action = int(np.argmax(q_vals_np))
                if best_raw_action not in legal:
                    illegal_attempts += 1

            next_obs, reward, done, info = train_env.step(action)
            next_legal = info["legal_actions"] if not done else []
            next_legal_mask = make_legal_mask(num_actions, next_legal)
            replay.add(obs, action, reward, next_obs, done, legal_mask, next_legal_mask)

            obs = next_obs
            ep_return += reward
            ep_len += 1
            global_step += 1

            if len(replay) >= args.learn_start and global_step % args.learn_every == 0:
                batch = replay.sample(args.batch_size)
                loss = qr_dqn_update(
                    batch=batch,
                    q_net=q_net,
                    target_net=target_net,
                    optimizer=optimizer,
                    gamma=args.gamma,
                    grad_clip=args.grad_clip,
                    kappa=args.kappa,
                    device=device,
                )
                ep_losses.append(loss)

            if global_step % args.target_sync_every == 0:
                target_net.load_state_dict(q_net.state_dict())

        eval_return = float("nan")
        if episode % args.eval_every == 0:
            eval_return, _, _, _, _ = greedy_rollout(
                q_net=q_policy,
                env=OpenSpiel2048Env(seed=1000 + episode),
                num_actions=num_actions,
                max_steps=args.max_steps_per_episode,
                device=device,
            )

        mean_loss = float(np.mean(ep_losses)) if ep_losses else float("nan")
        if log_episodes:
            print(
                f"Episode {episode}/{args.num_episodes} | "
                f"return={ep_return:.1f} | "
                f"length={ep_len} | "
                f"loss={mean_loss:.4f} | "
                f"max_tile={max_tile} | "
                f"illegal_attempts={illegal_attempts} | "
                f"eval_return={eval_return:.1f}"
            )

    return q_net, target_net, q_policy, obs_dim, num_actions


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
    q_net: QuantileQNetwork,
    target_net: QuantileQNetwork,
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
        source_notebook="QR-DQN",
        checkpoint_file="dqn.pt",
    )
    if log_save:
        print(f"Saved evaluation rollout archive to: {npz_path}")
        print(f"Saved evaluation metadata to: {json_path}")

    checkpoint_path = os.path.join(args.output_dir, "dqn.pt")
    torch.save(
        {
            "model_state_dict": q_net.state_dict(),
            "target_state_dict": target_net.state_dict(),
            "obs_dim": obs_dim,
            "num_actions": num_actions,
            "num_quantiles": q_net.num_quantiles,
        },
        checkpoint_path,
    )
    if log_save:
        print(f"Saved checkpoint to: {checkpoint_path}")


def main():
    args = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    q_net, target_net, q_policy, obs_dim, num_actions = train(args, device)
    eval_data = evaluate_policy(args, q_policy, num_actions, device)
    save_artifacts(args, q_net, target_net, obs_dim, num_actions, eval_data)


if __name__ == "__main__":
    main()
