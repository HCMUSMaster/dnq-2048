import argparse
from types import SimpleNamespace

import optuna
import torch

from app import dqn, double_dqn, dueling_double_dqn

ALGO_REGISTRY = {
    "dqn": {
        "train": dqn.train,
        "evaluate": dqn.evaluate_policy,
        "save": dqn.save_artifacts,
    },
    "double_dqn": {
        "train": double_dqn.train,
        "evaluate": double_dqn.evaluate_policy,
        "save": double_dqn.save_artifacts,
    },
    "dueling_double_dqn": {
        "train": dueling_double_dqn.train,
        "evaluate": dueling_double_dqn.evaluate_policy,
        "save": dueling_double_dqn.save_artifacts,
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tune DQN-family hyperparameters with Optuna for DQN, Double DQN, and Dueling Double DQN."
    )
    parser.add_argument(
        "--algorithm",
        choices=["dqn", "double_dqn", "dueling_double_dqn", "all"],
        default="all",
        help="Which algorithm to tune.",
    )
    parser.add_argument("--n_trials", type=int, default=30, help="Number of Optuna trials per algorithm.")
    parser.add_argument(
        "--study_name",
        type=str,
        default="dqn2048_tuning",
        help="Base name for Optuna study.",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optional Optuna storage URL, for example sqlite:///optuna.db.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Optional timeout in seconds per algorithm study.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed used by training scripts.")
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=120,
        help="Training episodes per trial (keep small for faster tuning).",
    )
    parser.add_argument(
        "--num_eval_seeds",
        type=int,
        default=20,
        help="Evaluation seeds per trial (final runs can use 100).",
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=20,
        help="In-training evaluation interval in episodes.",
    )
    parser.add_argument(
        "--max_steps_per_episode",
        type=int,
        default=5_000,
        help="Maximum steps per environment episode.",
    )
    parser.add_argument(
        "--save_best_dir",
        type=str,
        default=None,
        help="Optional base directory to save best model/eval artifacts per algorithm.",
    )
    parser.add_argument(
        "--metric_penalty",
        type=float,
        default=0.1,
        help="Objective penalty factor: score = avg_return - metric_penalty * std_return.",
    )
    return parser


def suggest_hparams(trial: optuna.Trial) -> dict:
    return {
        "buffer_size": trial.suggest_categorical("buffer_size", [20_000, 50_000, 100_000, 200_000]),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "gamma": trial.suggest_float("gamma", 0.95, 0.999),
        "lr": trial.suggest_float("lr", 1e-5, 3e-3, log=True),
        "target_sync_every": trial.suggest_categorical("target_sync_every", [100, 250, 500, 1000, 2000]),
        "learn_start": trial.suggest_categorical("learn_start", [500, 1000, 2000, 5000]),
        "learn_every": trial.suggest_categorical("learn_every", [1, 2, 4, 8]),
        "eps_start": 1.0,
        "eps_end": trial.suggest_float("eps_end", 0.01, 0.2),
        "eps_decay_steps": trial.suggest_categorical("eps_decay_steps", [5000, 10000, 20000, 50000, 100000]),
        "grad_clip": trial.suggest_float("grad_clip", 1.0, 20.0),
    }


def make_trial_args(cli_args: argparse.Namespace, hparams: dict, output_dir=None) -> SimpleNamespace:
    return SimpleNamespace(
        seed=cli_args.seed,
        num_episodes=cli_args.num_episodes,
        buffer_size=hparams["buffer_size"],
        batch_size=hparams["batch_size"],
        gamma=hparams["gamma"],
        lr=hparams["lr"],
        target_sync_every=hparams["target_sync_every"],
        learn_start=hparams["learn_start"],
        learn_every=hparams["learn_every"],
        eps_start=hparams["eps_start"],
        eps_end=hparams["eps_end"],
        eps_decay_steps=hparams["eps_decay_steps"],
        max_steps_per_episode=cli_args.max_steps_per_episode,
        grad_clip=hparams["grad_clip"],
        num_eval_seeds=cli_args.num_eval_seeds,
        eval_every=cli_args.eval_every,
        output_dir=output_dir,
        log_episodes=False,
        print_eval_summary=False,
        log_save=False,
    )


def tune_one_algorithm(algo_name: str, cli_args: argparse.Namespace):
    algo = ALGO_REGISTRY[algo_name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== Tuning {algo_name} on device={device} ===")

    def objective(trial: optuna.Trial) -> float:
        hparams = suggest_hparams(trial)
        trial_args = make_trial_args(cli_args, hparams)

        q_net, target_net, obs_dim, num_actions = algo["train"](trial_args, device)
        eval_data = algo["evaluate"](trial_args, q_net, num_actions, device)
        summary = eval_data["summary"]

        avg_return = float(summary["avg_return"])
        std_return = float(summary["std_return"])
        score = avg_return - cli_args.metric_penalty * std_return

        trial.set_user_attr("avg_return", avg_return)
        trial.set_user_attr("std_return", std_return)
        trial.set_user_attr("avg_length", float(summary["avg_length"]))
        trial.set_user_attr("avg_max_tile", float(summary["avg_max_tile"]))
        trial.set_user_attr(
            "avg_illegal_attempts", float(summary["avg_illegal_action_attempts"])
        )

        return score

    study_name = f"{cli_args.study_name}_{algo_name}"
    if cli_args.storage:
        study = optuna.create_study(
            study_name=study_name,
            storage=cli_args.storage,
            load_if_exists=True,
            direction="maximize",
        )
    else:
        study = optuna.create_study(direction="maximize")

    study.optimize(objective, n_trials=cli_args.n_trials, timeout=cli_args.timeout)

    best = study.best_trial
    print(f"\nBest trial for {algo_name}: #{best.number}")
    print(f"  Score: {best.value:.3f}")
    print(f"  avg_return: {best.user_attrs.get('avg_return', float('nan')):.1f}")
    print(f"  std_return: {best.user_attrs.get('std_return', float('nan')):.1f}")
    print(f"  avg_length: {best.user_attrs.get('avg_length', float('nan')):.1f}")
    print(f"  avg_max_tile: {best.user_attrs.get('avg_max_tile', float('nan')):.1f}")
    print(f"  avg_illegal_attempts: {best.user_attrs.get('avg_illegal_attempts', float('nan')):.1f}")
    print("  best_params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    if cli_args.save_best_dir:
        save_dir = f"{cli_args.save_best_dir.rstrip('/')}/{algo_name}"
        best_args = make_trial_args(cli_args, {**best.params, "eps_start": 1.0}, output_dir=save_dir)
        best_args.log_episodes = True
        best_args.print_eval_summary = True
        best_args.log_save = True

        print(f"\nRetraining best {algo_name} config and saving to: {save_dir}")
        q_net, target_net, obs_dim, num_actions = algo["train"](best_args, device)
        eval_data = algo["evaluate"](best_args, q_net, num_actions, device)
        algo["save"](best_args, q_net, target_net, obs_dim, num_actions, eval_data)


def main():
    args = build_parser().parse_args()
    algorithms = ["dqn", "double_dqn", "dueling_double_dqn"] if args.algorithm == "all" else [args.algorithm]

    for algo_name in algorithms:
        tune_one_algorithm(algo_name, args)


if __name__ == "__main__":
    main()
