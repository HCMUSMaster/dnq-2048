import random
import re
from collections import namedtuple

import numpy as np
import torch

# Lightweight RL transition container used by the replay buffer
Transition = namedtuple(
    "Transition",
    ["obs", "action", "reward", "next_obs", "done", "legal_mask", "next_legal_mask"],
)


def extract_obs(state, player_id=0):
    """Return a flat float32 observation vector for the player."""
    for fn_name, args in [
        ("observation_tensor", (player_id,)),
        ("observation_tensor", tuple()),
        ("information_state_tensor", (player_id,)),
        ("information_state_tensor", tuple()),
    ]:
        fn = getattr(state, fn_name, None)
        if fn is None:
            continue
        try:
            obs = fn(*args)
            obs = np.asarray(obs, dtype=np.float32).reshape(-1)
            return obs
        except TypeError:
            pass
    raise RuntimeError("Could not extract an observation tensor from state.")


def legal_actions(state, player_id=0):
    """Return legal actions for the current player state."""
    try:
        return list(state.legal_actions(player_id))
    except TypeError:
        return list(state.legal_actions())


def sample_chance_action(state, rng):
    outcomes = state.chance_outcomes()  # list of (action, prob)
    actions, probs = zip(*outcomes)
    idx = rng.choice(len(actions), p=np.asarray(probs, dtype=np.float64))
    return actions[idx]


def auto_resolve_chance_nodes(state, rng):
    """Mutate state until it is no longer a chance node."""
    while state.is_chance_node() and not state.is_terminal():
        a = sample_chance_action(state, rng)
        state.apply_action(a)
    return state


def state_return(state, player_id=0):
    vals = state.returns()
    return float(vals[player_id]) if len(vals) > player_id else 0.0


def state_reward(state, player_id=0):
    vals = state.rewards()
    return float(vals[player_id]) if len(vals) > player_id else 0.0


def parse_board_numbers(state):
    """Best-effort text parser for showing the board as a 4x4 integer array."""
    txt = str(state)
    nums = [int(x) for x in re.findall(r"\d+", txt)]
    if len(nums) >= 16:
        nums = nums[-16:]
        return np.array(nums, dtype=np.int64).reshape(4, 4)
    return None


def epsilon_by_step(
    step: int,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay_steps: int = 20_000,
):
    """Linear epsilon schedule from eps_start to eps_end over eps_decay_steps."""
    decay_steps = max(1, int(eps_decay_steps))
    frac = min(1.0, step / decay_steps)
    return eps_start + frac * (eps_end - eps_start)


@torch.no_grad()
def masked_greedy_action(
    q_net, obs, legal_actions_list, num_actions, epsilon=0.0, device=None
):
    if random.random() < epsilon:
        return random.choice(legal_actions_list)

    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    q = q_net(obs_t).squeeze(0)

    legal_mask = torch.zeros(num_actions, dtype=torch.bool, device=device)
    legal_mask[legal_actions_list] = True

    q_masked = q.masked_fill(~legal_mask, -1e9)
    action = int(torch.argmax(q_masked).item())
    return action


def dqn_update(
    batch,
    q_net,
    target_net,
    optimizer,
    gamma: float,
    grad_clip: float,
    device,
):
    """Run one DQN update step and return scalar loss."""
    obs = torch.tensor(np.asarray(batch.obs), dtype=torch.float32, device=device)
    actions = torch.tensor(batch.action, dtype=torch.int64, device=device).unsqueeze(1)
    rewards = torch.tensor(batch.reward, dtype=torch.float32, device=device)
    next_obs = torch.tensor(np.asarray(batch.next_obs), dtype=torch.float32, device=device)
    dones = torch.tensor(batch.done, dtype=torch.float32, device=device)

    next_legal_mask = torch.tensor(
        np.asarray(batch.next_legal_mask), dtype=torch.bool, device=device
    )

    q_values = q_net(obs)
    q_sa = q_values.gather(1, actions).squeeze(1)

    with torch.no_grad():
        next_q = target_net(next_obs)
        next_q = next_q.masked_fill(~next_legal_mask, -1e9)

        next_max_q = torch.max(next_q, dim=1).values
        next_max_q = torch.where(dones > 0.5, torch.zeros_like(next_max_q), next_max_q)

        target = rewards + gamma * next_max_q

    loss = torch.nn.functional.mse_loss(q_sa, target)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q_net.parameters(), grad_clip)
    optimizer.step()

    return float(loss.item())


def double_dqn_update(
    batch,
    q_net,
    target_net,
    optimizer,
    gamma: float,
    grad_clip: float,
    device,
):
    """Run one Double DQN update step and return scalar loss."""
    obs = torch.tensor(np.asarray(batch.obs), dtype=torch.float32, device=device)
    actions = torch.tensor(batch.action, dtype=torch.int64, device=device).unsqueeze(1)
    rewards = torch.tensor(batch.reward, dtype=torch.float32, device=device)
    next_obs = torch.tensor(np.asarray(batch.next_obs), dtype=torch.float32, device=device)
    dones = torch.tensor(batch.done, dtype=torch.float32, device=device)
    next_legal_mask = torch.tensor(
        np.asarray(batch.next_legal_mask), dtype=torch.bool, device=device
    )

    q_values = q_net(obs)
    q_sa = q_values.gather(1, actions).squeeze(1)

    with torch.no_grad():
        next_online_q = q_net(next_obs)
        next_online_q = next_online_q.masked_fill(~next_legal_mask, -1e9)
        next_actions = torch.argmax(next_online_q, dim=1, keepdim=True)

        next_target_q = target_net(next_obs).gather(1, next_actions).squeeze(1)
        next_target_q = torch.where(dones > 0.5, torch.zeros_like(next_target_q), next_target_q)
        target = rewards + gamma * next_target_q

    loss = torch.nn.functional.mse_loss(q_sa, target)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q_net.parameters(), grad_clip)
    optimizer.step()

    return float(loss.item())


__all__ = [
    "extract_obs",
    "legal_actions",
    "sample_chance_action",
    "auto_resolve_chance_nodes",
    "state_return",
    "state_reward",
    "parse_board_numbers",
    "epsilon_by_step",
    "masked_greedy_action",
    "dqn_update",
    "double_dqn_update",
]
