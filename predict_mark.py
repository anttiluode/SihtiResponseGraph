"""Render one query with a fine-tuned image reader and explicit mark head."""

import argparse
from pathlib import Path

from response_graph.reader_experiment import render_mark_aware_query


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reader_checkpoint", type=Path)
    parser.add_argument("mark_checkpoint", type=Path)
    parser.add_argument("input_image", type=Path)
    parser.add_argument("output_image", type=Path)
    parser.add_argument("--query", type=int, choices=range(4), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    render_mark_aware_query(args.reader_checkpoint, args.mark_checkpoint,
                            args.input_image, args.query, args.output_image,
                            device=args.device)
    print(args.output_image)


if __name__ == "__main__":
    main()
