"""Fine-tune image readers on quantized frozen graph and vector memories."""

import argparse
from pathlib import Path

from response_graph.reader_experiment import run_reader_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-checkpoint", type=Path, default=Path("runs/first/graph.pt"))
    parser.add_argument("--vector-checkpoint", type=Path, default=Path("runs/first/vector.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/mark_reader"))
    parser.add_argument("--writer-seed", type=int, default=7)
    parser.add_argument("--train-seed", type=int, default=5_000_010)
    parser.add_argument("--dev-seed", type=int, default=5_000_011)
    parser.add_argument("--test-seed", type=int, default=6_000_010)
    parser.add_argument("--train-count", type=int, default=8192)
    parser.add_argument("--dev-count", type=int, default=128)
    parser.add_argument("--test-count", type=int, default=256)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--mark-weight", type=float, default=0.1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    run_reader_experiment(args.graph_checkpoint, args.vector_checkpoint, args.output_dir,
                          writer_seed=args.writer_seed, train_seed=args.train_seed,
                          dev_seed=args.dev_seed, test_seed=args.test_seed,
                          train_count=args.train_count, dev_count=args.dev_count,
                          test_count=args.test_count, steps=args.steps,
                          batch_size=args.batch_size, mark_weight=args.mark_weight,
                          device=args.device)
    print(args.output_dir / "metrics.json")


if __name__ == "__main__":
    main()
