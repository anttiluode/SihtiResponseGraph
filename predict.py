"""Render one of four planar view queries from a saved graph model."""

import argparse
from pathlib import Path

from response_graph.render import render_query


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("input_image", type=Path)
    parser.add_argument("output_image", type=Path)
    parser.add_argument("--query", type=int, choices=range(4), required=True,
                        help="quarter-turns from the observed view (0, 1, 2, 3)")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    render_query(args.checkpoint, args.input_image, args.query, args.output_image,
                 args.device)
    print(args.output_image)


if __name__ == "__main__":
    main()
