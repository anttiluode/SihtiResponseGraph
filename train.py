"""Train both memory models on the same tiny 2D image-query experiment."""

import argparse
import json
from pathlib import Path

from response_graph.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--train-count", type=int, default=2048)
    parser.add_argument("--val-count", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/first"))
    args = parser.parse_args()
    result = run_experiment(args.steps, args.batch_size, args.seed, args.device,
                            args.output_dir, args.train_count, args.val_count)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
