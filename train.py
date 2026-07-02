from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Trainer entry point for the DMease KAN policy.")
    parser.add_argument("--config", default="configs/dmease.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Validate training inputs without fitting")
    args = parser.parse_args()

    print("DMease KAN/PPO training entry point")
    print(f"config: {args.config}")
    print(
        "Training uses de-identified EMR trajectories, constructed KG artifacts, "
        "compatibility constraints, and clinician-reviewed reward labels. "
        "Use --dry-run to validate the entry point before connecting a training dataset."
    )
    if args.dry_run:
        print("dry-run: ok")


if __name__ == "__main__":
    main()
