"""Test whether frozen byte-valued memories retain the tiny colored mark."""

import argparse
from pathlib import Path

from response_graph.probe import run_mark_probe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-checkpoint", type=Path, default=Path("runs/first/graph.pt"))
    parser.add_argument("--vector-checkpoint", type=Path, default=Path("runs/first/vector.pt"))
    parser.add_argument("--output", type=Path, default=Path("runs/mark_probe/metrics.json"))
    parser.add_argument("--seed", type=int, default=7, help="writer training seed")
    parser.add_argument("--probe-train-seed", type=int,
                        help="optional fresh seed for probe examples; default is writer seed")
    parser.add_argument("--train-count", type=int, default=2048)
    parser.add_argument("--val-count", type=int, default=128)
    parser.add_argument("--steps", type=int, default=600, help="optimizer steps per probe")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--probe-seeds", type=int, default=3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    run_mark_probe(args.graph_checkpoint, args.vector_checkpoint, args.output,
                   seed=args.seed, train_count=args.train_count, val_count=args.val_count,
                   steps=args.steps, batch_size=args.batch_size,
                   probe_seeds=args.probe_seeds, device=args.device,
                   probe_train_seed=args.probe_train_seed)
    print(args.output)


if __name__ == "__main__":
    main()
