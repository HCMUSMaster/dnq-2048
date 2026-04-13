import argparse
import subprocess
import sys
from pathlib import Path


ALLOWED_MODELS = ("dqn", "ddqn", "dddqn", "qrdqn", "hdqn")
ALL_MODELS_TOKEN = "all"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run best-known hyperparameters for selected 2048 DQN-family models."
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=300,
        help="Number of training episodes for each selected model.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="dqn,ddqn,dddqn,qrdqn,hdqn",
        help="Comma-separated models to run: dqn,ddqn,dddqn,qrdqn,hdqn,all",
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


def run_command(command: list[str], cwd: Path) -> None:
    subprocess.run(command, check=True, cwd=cwd)


def main() -> None:
    args = build_parser().parse_args()

    root_dir = Path(__file__).resolve().parents[1]
    base_out = root_dir / "output"
    base_out.mkdir(parents=True, exist_ok=True)

    selected_models = parse_models(args.models)

    if "dqn" in selected_models:
        print("Running best DQN hyperparameters...")
        run_command(
            [
                sys.executable,
                "-m",
                "app.dqn",
                "--num_episodes",
                str(args.num_episodes),
                "--buffer_size",
                "200000",
                "--batch_size",
                "128",
                "--gamma",
                "0.9556791110090774",
                "--lr",
                "0.00014394430404112647",
                "--target_sync_every",
                "250",
                "--learn_start",
                "1000",
                "--learn_every",
                "8",
                "--eps_start",
                "1.0",
                "--eps_end",
                "0.1488425606123552",
                "--eps_decay_steps",
                "20000",
                "--max_steps_per_episode",
                "5000",
                "--grad_clip",
                "4.3076689475049506",
                "--num_eval_seeds",
                "100",
                "--output_dir",
                str(base_out / "dqn"),
            ],
            cwd=root_dir,
        )

    if "ddqn" in selected_models:
        print("Running best Double DQN hyperparameters...")
        run_command(
            [
                sys.executable,
                "-m",
                "app.double_dqn",
                "--num_episodes",
                str(args.num_episodes),
                "--buffer_size",
                "20000",
                "--batch_size",
                "256",
                "--gamma",
                "0.9798426351600259",
                "--lr",
                "0.0009768958892956907",
                "--target_sync_every",
                "100",
                "--learn_start",
                "1000",
                "--learn_every",
                "2",
                "--eps_start",
                "1.0",
                "--eps_end",
                "0.17004595587593185",
                "--eps_decay_steps",
                "50000",
                "--max_steps_per_episode",
                "5000",
                "--grad_clip",
                "6.668923925628828",
                "--num_eval_seeds",
                "100",
                "--output_dir",
                str(base_out / "double-dqn"),
            ],
            cwd=root_dir,
        )

    if "dddqn" in selected_models:
        print("Running best Dueling Double DQN hyperparameters...")
        run_command(
            [
                sys.executable,
                "-m",
                "app.dueling_double_dqn",
                "--num_episodes",
                str(args.num_episodes),
                "--buffer_size",
                "200000",
                "--batch_size",
                "64",
                "--gamma",
                "0.9654707747743687",
                "--lr",
                "0.0029222928006765354",
                "--target_sync_every",
                "250",
                "--learn_start",
                "2000",
                "--learn_every",
                "8",
                "--eps_start",
                "1.0",
                "--eps_end",
                "0.19214154949975376",
                "--eps_decay_steps",
                "10000",
                "--max_steps_per_episode",
                "5000",
                "--grad_clip",
                "1.1189121985227757",
                "--num_eval_seeds",
                "100",
                "--output_dir",
                str(base_out / "dueling-double-dqn"),
            ],
            cwd=root_dir,
        )

    if "qrdqn" in selected_models:
        print("Running QR-DQN hyperparameters...")
        run_command(
            [
                sys.executable,
                "-m",
                "app.qr_dqn",
                "--num_episodes",
                str(args.num_episodes),
                "--buffer_size",
                "200000",
                "--batch_size",
                "256",
                "--gamma",
                "0.9922439589929455",
                "--lr",
                "0.0001017167470236331",
                "--target_sync_every",
                "1000",
                "--learn_start",
                "5000",
                "--learn_every",
                "2",
                "--eps_start",
                "1.0",
                "--eps_end",
                "0.16221083409453396",
                "--eps_decay_steps",
                "20000",
                "--max_steps_per_episode",
                "5000",
                "--grad_clip",
                "15.05157300548717",
                "--num_quantiles",
                "51",
                "--kappa",
                "1.0",
                "--num_eval_seeds",
                "100",
                "--output_dir",
                str(base_out / "qr-dqn"),
            ],
            cwd=root_dir,
        )

    if "hdqn" in selected_models:
        print("Running H-DQN hyperparameters...")
        run_command(
            [
                sys.executable,
                "-m",
                "app.h_dqn",
                "--num_episodes",
                str(args.num_episodes),
                "--goal_tiles",
                "32,64,128,256,512,1024,2048",
                "--option_duration",
                "8",
                "--intrinsic_success_reward",
                "1.0",
                "--intrinsic_step_penalty",
                "-0.01",
                "--ctrl_buffer_size",
                "20000",
                "--ctrl_batch_size",
                "256",
                "--ctrl_gamma",
                "0.9974851723474096",
                "--ctrl_lr",
                "0.00017931204091428885",
                "--ctrl_target_sync_every",
                "250",
                "--ctrl_learn_start",
                "5000",
                "--ctrl_learn_every",
                "8",
                "--meta_buffer_size",
                "20000",
                "--meta_batch_size",
                "64",
                "--meta_gamma",
                "0.9974851723474096",
                "--meta_lr",
                "0.00017931204091428885",
                "--meta_target_sync_every",
                "50",
                "--meta_learn_start",
                "500",
                "--meta_learn_every",
                "2",
                "--eps_start",
                "1.0",
                "--eps_end",
                "0.04539376238997539",
                "--eps_decay_steps",
                "20000",
                "--meta_eps_start",
                "1.0",
                "--meta_eps_end",
                "0.04539376238997539",
                "--meta_eps_decay_steps",
                "2000",
                "--max_steps_per_episode",
                "5000",
                "--grad_clip",
                "7.010189344276133",
                "--num_eval_seeds",
                "100",
                "--output_dir",
                str(base_out / "h-dqn"),
            ],
            cwd=root_dir,
        )

    print(f"Selected models: {','.join(selected_models)}")
    print(f"All selected runs completed. Outputs are in: {base_out}")


if __name__ == "__main__":
    main()
