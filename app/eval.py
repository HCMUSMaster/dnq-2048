"""
Evaluation utilities for DQN agents on OpenSpiel 2048.
Supports single-episode greedy rollouts and multi-seed evaluation with TensorBoard logging.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from app.helpers import masked_greedy_action
from app.replay_buffer import make_legal_mask


def greedy_rollout(
    q_net: torch.nn.Module,
    env,
    num_actions: int,
    max_steps: int = 5_000,
    device: Optional[torch.device] = None,
) -> Tuple[float, int, int, int, List[Dict]]:
    """
    Execute a single greedy rollout (no exploration) and collect transition data.

    Args:
        q_net: Trained Q-network.
        env: OpenSpiel2048Env instance.
        num_actions: Number of discrete actions.
        max_steps: Maximum steps per episode.
        device: PyTorch device.

    Returns:
        Tuple of (episode_return, episode_length, max_tile, illegal_action_attempts, rollout_list)
        where rollout_list is a list of dicts with keys:
        action, reward, legal_actions, board, state_text, raw_greedy_action, illegal_action_attempt
    """
    if device is None:
        device = torch.device("cpu")

    obs = env.reset()
    done = False
    episode_return = 0.0
    episode_length = 0
    max_tile = 0
    illegal_action_attempts = 0
    rollout = []

    while not done and episode_length < max_steps:
        # Track max tile
        if obs is not None and len(obs) > 0:
            max_tile = max(max_tile, int(np.max(obs)))

        legal = env.legal_actions()
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            q_values = q_net(obs_t).squeeze(0)
        raw_greedy_action = int(torch.argmax(q_values).item())
        illegal_action_attempt = raw_greedy_action not in legal
        if illegal_action_attempt:
            illegal_action_attempts += 1

        action = masked_greedy_action(
            q_net=q_net,
            obs=obs,
            legal_actions_list=legal,
            num_actions=num_actions,
            epsilon=0.0,
            device=device,
        )

        next_obs, reward, done, info = env.step(action)
        rollout.append(
            {
                "action": action,
                "reward": reward,
                "legal_actions": legal,
                "board": info.get("board"),
                "state_text": info.get("state_text"),
                "raw_greedy_action": raw_greedy_action,
                "illegal_action_attempt": bool(illegal_action_attempt),
            }
        )
        obs = next_obs
        episode_return += reward
        episode_length += 1

    return episode_return, episode_length, max_tile, illegal_action_attempts, rollout


def evaluate_multi_seed(
    q_net: torch.nn.Module,
    env_class,
    num_eval_seeds: int = 10,
    num_actions: int = 4,
    max_steps_per_episode: int = 5_000,
    device: Optional[torch.device] = None,
    seed_offset: int = 5000,
) -> Dict:
    """
    Evaluate the greedy policy over multiple random seeds.

    Collects transitions and summary statistics, suitable for saving as NPZ + JSON.

    Args:
        q_net: Trained Q-network.
        env_class: OpenSpiel2048Env class (or similar).
        num_eval_seeds: Number of evaluation seeds to run.
        num_actions: Number of discrete actions.
        max_steps_per_episode: Maximum steps per episode.
        device: PyTorch device.
        seed_offset: Base seed offset for eval runs.

    Returns:
        Dict with keys:
        - 'seed_ids', 'episode_ids', 'step_ids': Indexing arrays
        - 'states', 'next_states', 'actions', 'rewards', 'dones', 'legal_masks': Transition data
        - 'episode_returns', 'episode_lengths', 'max_tiles': Per-episode summaries
        - 'summary': Dict with mean/std statistics
    """
    if device is None:
        device = torch.device("cpu")

    eval_rollout_seed_ids = []
    eval_rollout_episode_ids = []
    eval_rollout_step_ids = []
    eval_rollout_states = []
    eval_rollout_next_states = []
    eval_rollout_actions = []
    eval_rollout_rewards = []
    eval_rollout_dones = []
    eval_rollout_legal_masks = []
    eval_rollout_illegal_action_attempt_flags = []
    eval_rollout_episode_returns = []
    eval_rollout_episode_lengths = []
    eval_rollout_episode_illegal_action_attempts = []

    multi_seed_returns = []
    multi_seed_lengths = []
    multi_seed_max_tiles = []

    for seed_idx in range(num_eval_seeds):
        eval_seed = seed_offset + seed_idx
        eval_env = env_class(seed=eval_seed)
        obs = eval_env.reset(seed=eval_seed)
        done = False
        ret = 0.0
        steps = 0
        max_tile = 0
        illegal_attempts = 0

        eval_rollout_seed_ids.append(eval_seed)

        while not done and steps < max_steps_per_episode:
            # Track max tile
            if obs is not None and len(obs) > 0:
                max_tile = max(max_tile, int(np.max(obs)))

            legal = eval_env.legal_actions()
            legal_mask = make_legal_mask(num_actions, legal)
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                q_values = q_net(obs_t).squeeze(0)
            raw_greedy_action = int(torch.argmax(q_values).item())
            illegal_action_attempt = raw_greedy_action not in legal
            if illegal_action_attempt:
                illegal_attempts += 1

            action = masked_greedy_action(
                q_net=q_net,
                obs=obs,
                legal_actions_list=legal,
                num_actions=num_actions,
                epsilon=0.0,
                device=device,
            )
            next_obs, reward, done, info = eval_env.step(action)

            eval_rollout_episode_ids.append(seed_idx)
            eval_rollout_step_ids.append(steps)
            eval_rollout_states.append(np.asarray(obs, dtype=np.float32))
            eval_rollout_next_states.append(np.asarray(next_obs, dtype=np.float32))
            eval_rollout_actions.append(int(action))
            eval_rollout_rewards.append(float(reward))
            eval_rollout_dones.append(bool(done))
            eval_rollout_legal_masks.append(legal_mask.astype(np.bool_))
            eval_rollout_illegal_action_attempt_flags.append(bool(illegal_action_attempt))

            obs = next_obs
            ret += reward
            steps += 1

        multi_seed_returns.append(ret)
        multi_seed_lengths.append(steps)
        multi_seed_max_tiles.append(max_tile)
        eval_rollout_episode_returns.append(ret)
        eval_rollout_episode_lengths.append(steps)
        eval_rollout_episode_illegal_action_attempts.append(illegal_attempts)

    # Compute summary statistics
    avg_return = float(np.mean(multi_seed_returns))
    std_return = float(np.std(multi_seed_returns))
    avg_length = float(np.mean(multi_seed_lengths))
    avg_max_tile = float(np.mean(multi_seed_max_tiles))
    avg_illegal_action_attempts = float(np.mean(eval_rollout_episode_illegal_action_attempts))
    total_illegal_action_attempts = int(np.sum(eval_rollout_episode_illegal_action_attempts))

    return {
        "seed_ids": np.asarray(eval_rollout_seed_ids, dtype=np.int64),
        "episode_ids": np.asarray(eval_rollout_episode_ids, dtype=np.int64),
        "step_ids": np.asarray(eval_rollout_step_ids, dtype=np.int64),
        "states": np.stack(eval_rollout_states).astype(np.float32),
        "next_states": np.stack(eval_rollout_next_states).astype(np.float32),
        "actions": np.asarray(eval_rollout_actions, dtype=np.int64),
        "rewards": np.asarray(eval_rollout_rewards, dtype=np.float32),
        "dones": np.asarray(eval_rollout_dones, dtype=np.bool_),
        "legal_masks": np.stack(eval_rollout_legal_masks).astype(np.bool_),
        "illegal_action_attempt_flags": np.asarray(
            eval_rollout_illegal_action_attempt_flags, dtype=np.bool_
        ),
        "episode_returns": np.asarray(eval_rollout_episode_returns, dtype=np.float32),
        "episode_lengths": np.asarray(eval_rollout_episode_lengths, dtype=np.int64),
        "episode_illegal_action_attempts": np.asarray(
            eval_rollout_episode_illegal_action_attempts, dtype=np.int64
        ),
        "max_tiles": np.asarray(multi_seed_max_tiles, dtype=np.int64),
        "summary": {
            "avg_return": avg_return,
            "std_return": std_return,
            "min_return": float(np.min(multi_seed_returns)),
            "max_return": float(np.max(multi_seed_returns)),
            "avg_length": avg_length,
            "avg_max_tile": avg_max_tile,
            "avg_illegal_action_attempts": avg_illegal_action_attempts,
            "total_illegal_action_attempts": total_illegal_action_attempts,
        },
    }


def save_eval_results(
    eval_data: Dict,
    output_dir: str,
    num_actions: int,
    obs_dim: int,
    source_notebook: str = "DQN Baseline",
    checkpoint_file: str = "dqn.pt",
) -> Tuple[str, str]:
    """
    Save evaluation results to NPZ and metadata JSON.

    Args:
        eval_data: Dict from evaluate_multi_seed().
        output_dir: Directory to save files.
        num_actions: Number of actions.
        obs_dim: Observation dimension.
        source_notebook: Name of source notebook.
        checkpoint_file: Name of checkpoint file.

    Returns:
        Tuple of (npz_path, json_path)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save NPZ
    npz_path = os.path.join(output_dir, "eval_rollout.npz")
    np.savez_compressed(
        npz_path,
        seed_ids=eval_data["seed_ids"],
        episode_ids=eval_data["episode_ids"],
        step_ids=eval_data["step_ids"],
        states=eval_data["states"],
        next_states=eval_data["next_states"],
        actions=eval_data["actions"],
        rewards=eval_data["rewards"],
        dones=eval_data["dones"],
        legal_masks=eval_data["legal_masks"],
        illegal_action_attempt_flags=eval_data["illegal_action_attempt_flags"],
        episode_returns=eval_data["episode_returns"],
        episode_lengths=eval_data["episode_lengths"],
        episode_illegal_action_attempts=eval_data["episode_illegal_action_attempts"],
        max_tiles=eval_data["max_tiles"],
        num_actions=np.int64(num_actions),
        obs_dim=np.int64(obs_dim),
    )

    # Save JSON metadata
    eval_meta = {
        "schema_version": 1,
        "game": "2048",
        "library": "OpenSpiel",
        "source_notebook": source_notebook,
        "checkpoint_file": checkpoint_file,
        "rollout_file": "eval_rollout.npz",
        "num_eval_seeds": int(len(eval_data["seed_ids"])),
        "seed_ids": [int(s) for s in eval_data["seed_ids"]],
        "obs_dim": int(obs_dim),
        "num_actions": int(num_actions),
        "action_order": "OpenSpiel action indices 0..num_actions-1",
        "state_encoding": "flat observation tensor from app.helpers.extract_obs",
        "state_storage": "pre-action and post-action observations in eval_rollout.npz",
        "episode_boundaries": "episode_ids and step_ids define episode grouping",
        "summary": eval_data["summary"],
    }

    json_path = os.path.join(output_dir, "eval_meta.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(eval_meta, f, indent=2)

    return npz_path, json_path


__all__ = [
    "greedy_rollout",
    "evaluate_multi_seed",
    "save_eval_results",
]
