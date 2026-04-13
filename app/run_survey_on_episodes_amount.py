import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from app import dqn, double_dqn, dueling_double_dqn, h_dqn, qr_dqn


ALLOWED_MODELS = ("dqn", "ddqn", "dddqn", "qrdqn", "hdqn")
ALL_MODELS_TOKEN = "all"
DEFAULT_EPISODES_CSV = "300,500,1000"


MODEL_REGISTRY = {
    "dqn": {
        "slug": "dqn",
        "module": dqn,
        "params": {
            "buffer_size": 200000,
            "batch_size": 128,
            "gamma": 0.9556791110090774,
            "lr": 0.00014394430404112647,
            "target_sync_every": 250,
            "learn_start": 1000,
            "learn_every": 8,
            "eps_start": 1.0,
            "eps_end": 0.1488425606123552,
            "eps_decay_steps": 20000,
            "grad_clip": 4.3076689475049506,
        },
    },
    "ddqn": {
        "slug": "double-dqn",
        "module": double_dqn,
        "params": {
            "buffer_size": 20000,
            "batch_size": 256,
            "gamma": 0.9798426351600259,
            "lr": 0.0009768958892956907,
            "target_sync_every": 100,
            "learn_start": 1000,
            "learn_every": 2,
            "eps_start": 1.0,
            "eps_end": 0.17004595587593185,
            "eps_decay_steps": 50000,
            "grad_clip": 6.668923925628828,
        },
    },
    "dddqn": {
        "slug": "dueling-double-dqn",
        "module": dueling_double_dqn,
        "params": {
            "buffer_size": 200000,
            "batch_size": 64,
            "gamma": 0.9654707747743687,
            "lr": 0.0029222928006765354,
            "target_sync_every": 250,
            "learn_start": 2000,
            "learn_every": 8,
            "eps_start": 1.0,
            "eps_end": 0.19214154949975376,
            "eps_decay_steps": 10000,
            "grad_clip": 1.1189121985227757,
        },
    },
    "qrdqn": {
        "slug": "qr-dqn",
        "module": qr_dqn,
        "params": {
            "buffer_size": 200000,
            "batch_size": 256,
            "gamma": 0.9922439589929455,
            "lr": 0.0001017167470236331,
            "target_sync_every": 1000,
            "learn_start": 5000,
            "learn_every": 2,
            "eps_start": 1.0,
            "eps_end": 0.16221083409453396,
            "eps_decay_steps": 20000,
            "grad_clip": 15.05157300548717,
            "num_quantiles": 51,
            "kappa": 1.0,
        },
    },
    "hdqn": {
        "slug": "h-dqn",
        "module": h_dqn,
        "params": {
            "goal_tiles": "32,64,128,256,512,1024,2048",
            "option_duration": 8,
            "intrinsic_success_reward": 1.0,
            "intrinsic_step_penalty": -0.01,
            "ctrl_buffer_size": 20000,
            "ctrl_batch_size": 256,
            "ctrl_gamma": 0.9974851723474096,
            "ctrl_lr": 0.00017931204091428885,
            "ctrl_target_sync_every": 250,
            "ctrl_learn_start": 5000,
            "ctrl_learn_every": 8,
            "meta_buffer_size": 20000,
            "meta_batch_size": 64,
            "meta_gamma": 0.9974851723474096,
            "meta_lr": 0.00017931204091428885,
            "meta_target_sync_every": 50,
            "meta_learn_start": 500,
            "meta_learn_every": 2,
            "eps_start": 1.0,
            "eps_end": 0.04539376238997539,
            "eps_decay_steps": 20000,
            "meta_eps_start": 1.0,
            "meta_eps_end": 0.04539376238997539,
            "meta_eps_decay_steps": 2000,
            "grad_clip": 7.010189344276133,
        },
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Survey the effect of training episode count on selected models."
    )
    parser.add_argument(
        "--models",
        type=str,
        default="dqn,ddqn,dddqn,qrdqn,hdqn",
        help="Comma-separated models to run: dqn,ddqn,dddqn,qrdqn,hdqn,all",
    )
    parser.add_argument(
        "--num_episodes",
        type=str,
        default=DEFAULT_EPISODES_CSV,
        help="Comma-separated training episode counts, for example: 300,500,1000",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument(
        "--max_steps_per_episode",
        type=int,
        default=5000,
        help="Maximum steps per environment episode.",
    )
    parser.add_argument(
        "--num_eval_seeds",
        type=int,
        default=100,
        help="Evaluation seeds per run.",
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=20,
        help="In-training greedy evaluation interval.",
    )
    return parser


def parse_models(models_csv: str) -> list[str]:
    models: list[str] = []
    seen: set[str] = set()
    for raw_model in models_csv.split(","):
        model = raw_model.strip().lower()
        if not model:
            allowed = ",".join((*ALLOWED_MODELS, ALL_MODELS_TOKEN))
            raise ValueError(f"Unknown model in --models: {model}. Allowed models: {allowed}")
        if model == ALL_MODELS_TOKEN:
            return list(ALLOWED_MODELS)
        if model not in ALLOWED_MODELS:
            allowed = ",".join((*ALLOWED_MODELS, ALL_MODELS_TOKEN))
            raise ValueError(f"Unknown model in --models: {model}. Allowed models: {allowed}")
        if model not in seen:
            models.append(model)
            seen.add(model)
    return models


def parse_episode_list(episodes_csv: str) -> list[int]:
    episodes: list[int] = []
    seen: set[int] = set()
    for raw_value in episodes_csv.split(","):
        value = raw_value.strip()
        if not value:
            raise ValueError("Empty value in --num_episodes list.")
        episode_count = int(value)
        if episode_count <= 0:
            raise ValueError("Episode values in --num_episodes must be positive integers.")
        if episode_count not in seen:
            episodes.append(episode_count)
            seen.add(episode_count)
    return episodes


def build_run_args(
    *,
    model: str,
    num_episodes: int,
    output_dir: Path,
    seed: int,
    max_steps_per_episode: int,
    num_eval_seeds: int,
    eval_every: int,
) -> SimpleNamespace:
    common = {
        "seed": seed,
        "num_episodes": num_episodes,
        "max_steps_per_episode": max_steps_per_episode,
        "num_eval_seeds": num_eval_seeds,
        "eval_every": eval_every,
        "output_dir": str(output_dir),
        "log_episodes": False,
        "print_eval_summary": True,
        "log_save": False,
    }
    params = MODEL_REGISTRY[model]["params"]
    return SimpleNamespace(**common, **params)


def run_single_model(model: str, run_args: SimpleNamespace, device: torch.device) -> dict:
    module = MODEL_REGISTRY[model]["module"]

    if model == "qrdqn":
        q_net, target_net, q_policy, obs_dim, num_actions = module.train(run_args, device)
        eval_data = module.evaluate_policy(run_args, q_policy, num_actions, device)
        module.save_artifacts(run_args, q_net, target_net, obs_dim, num_actions, eval_data)
        return eval_data

    if model == "hdqn":
        (
            controller_net,
            target_controller_net,
            meta_net,
            target_meta_net,
            q_policy,
            goals,
            obs_dim,
            num_actions,
        ) = module.train(run_args, device)
        eval_data = module.evaluate_policy(run_args, q_policy, num_actions, device)
        module.save_artifacts(
            run_args,
            controller_net,
            target_controller_net,
            meta_net,
            target_meta_net,
            goals,
            obs_dim,
            num_actions,
            eval_data,
        )
        return eval_data

    q_net, target_net, obs_dim, num_actions = module.train(run_args, device)
    eval_data = module.evaluate_policy(run_args, q_net, num_actions, device)
    module.save_artifacts(run_args, q_net, target_net, obs_dim, num_actions, eval_data)
    return eval_data


def to_jsonable_summary(summary: dict) -> dict:
    return {
        "avg_return": float(summary["avg_return"]),
        "std_return": float(summary["std_return"]),
        "avg_length": float(summary["avg_length"]),
        "avg_max_tile": float(summary["avg_max_tile"]),
        "avg_illegal_action_attempts": float(summary["avg_illegal_action_attempts"]),
        "total_illegal_action_attempts": int(summary["total_illegal_action_attempts"]),
    }


def main() -> None:
    args = build_parser().parse_args()
    selected_models = parse_models(args.models)
    episode_values = parse_episode_list(args.num_episodes)

    root_dir = Path(__file__).resolve().parents[1]
    survey_root = root_dir / "output" / "survey"
    survey_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    full_summary: dict[str, list[dict]] = {}

    for model in selected_models:
        model_slug = MODEL_REGISTRY[model]["slug"]
        model_root = survey_root / model_slug
        model_root.mkdir(parents=True, exist_ok=True)
        model_summary: list[dict] = []

        for episode_count in episode_values:
            run_output_dir = model_root / f"episodes_{episode_count}"
            print(f"Running model={model} episodes={episode_count}...")
            run_args = build_run_args(
                model=model,
                num_episodes=episode_count,
                output_dir=run_output_dir,
                seed=args.seed,
                max_steps_per_episode=args.max_steps_per_episode,
                num_eval_seeds=args.num_eval_seeds,
                eval_every=args.eval_every,
            )
            eval_data = run_single_model(model, run_args, device)
            summary = to_jsonable_summary(eval_data["summary"])
            model_summary.append(
                {
                    "num_episodes": episode_count,
                    "output_dir": str(run_output_dir.relative_to(root_dir)),
                    "summary": summary,
                }
            )

        model_summary_path = model_root / "summary_by_episodes.json"
        model_summary_path.write_text(json.dumps(model_summary, indent=2), encoding="utf-8")
        full_summary[model] = model_summary
        print(f"Saved survey summary: {model_summary_path}")

    full_summary_path = survey_root / "summary_all_models.json"
    full_summary_path.write_text(json.dumps(full_summary, indent=2), encoding="utf-8")
    print(f"Saved combined survey summary: {full_summary_path}")


if __name__ == "__main__":
    main()
