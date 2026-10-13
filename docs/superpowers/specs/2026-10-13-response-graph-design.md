# Sihti Response Graph v0

## Purpose

Build a small, trainable image model that stores one observation in the conductances of a spatial graph. A later image query starts with fresh, image-independent activity. The experiment tests whether the stored graph changes what the query can recover after active state has been cleared.

This is a controlled 2D gate, not a claim about novel 3D geometry, natural photographs, biological synapses, or compression superiority. It is a new model and experiment in this repository, informed by [Sihti's image-derived pixel graph](https://github.com/anttiluode/Sihti), not a copy of Sihti's code.

## Model

- Input: RGB image, shape `[B, 3, 32, 32]`, normalized to `[0,1]`.
- Memory: a fixed 8×8 lattice of undirected horizontal and vertical edges. Eight learned conductances per edge, each in `[0,1]`. There are `2×8×7×8 = 896` stored scalars, or 896 bytes if quantized to uint8. Shared learned parameters are excluded from this per-instance count and reported separately.
- Write: a small convolutional network pools the observation to 8×8 features. An edge network uses endpoint features to produce its conductances. `write(image, previous=None, visibility=None)` blends visible new evidence into a previous graph; zero visibility must leave the previous graph untouched. No RGB tensor, encoded image features, or hidden activity may be retained by `GraphMemory`.
- Read: each `read(memory, query)` constructs a fresh coordinate-and-query ping, runs 8 lazy, degree-normalized graph propagation steps, concatenates responses from depths 0, 2, 4, and 8, and decodes a 32×32 RGB image. It cannot access the observed image or a previous read's activity. Query is one of four quarter-turns relative to the observed canonical image.
- Baseline: a vector encoder/decoder with exactly 896 uint8 per-instance values and the same images, queries, optimizer, step count, and split. Parameter counts and forward times are reported, not silently assumed equal.

## Data and training

Generate deterministic 32×32 asymmetric colored glyphs with a small identity mark. The source is the canonical view; the target is its queried quarter-turn. Train/test objects come from disjoint random seeds. This tests question-conditioned visual recall and planar change of coordinates. It cannot distinguish a genuinely unseen surface from a plausible completion; that requires a later multiview dataset and uncertainty model.

Train both models with pixel MSE using identical minibatches. Use an explicit seed, support CPU and CUDA, write checkpoints, machine-readable metrics and a source/target/prediction contact sheet. No pretrained teacher is required for this gate; a teacher can later add identity-sensitive targets without seeing held-out pixels at inference.

## Measurements and controls

1. Held-out MSE and PSNR for all four query turns, graph versus vector at matched serialized state bytes.
2. Fixed query after a fresh reset, repeated twice: deterministic output; read order cannot change memory.
3. Swap two examples' graphs before reading with the same query: output must change, showing the graph actually participates.
4. Replace learned per-example conductances by a graph mean computed only from training examples: report held-out MSE; a fixed graph is the substrate control.
5. Quantize/dequantize the graph and vector memory to uint8 for evaluation and report the measured byte count. Training remains differentiable in float.
6. Zero-visibility update leaves memory identical; full-visibility update changes it.

No performance win is assumed. A graph loss close to fixed-graph loss or worse than the vector baseline is a valid negative result. The README must separate measured outcomes from the mechanism and show the exact command to repeat them.

## Boundaries and operation

Python 3.11+, PyTorch, NumPy, Pillow. No remote dataset, model weights, or API required. Training defaults must fit an RTX 3060 at batch size 32; CPU smoke runs need only a few batches. Fail clearly on invalid shapes, query indices, and absent checkpoints. Save generated runs below an ignored `runs/` directory. The CI workflow runs unit tests and a two-step CPU smoke train.
