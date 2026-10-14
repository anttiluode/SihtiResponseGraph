# Can the stored graph answer a question about the tiny mark?

The [first 1,000-step CUDA model](../runs/first/metrics.json) renders blurry images that largely lose a five-pixel colored identity mark. This probe asks whether the *frozen, quantized* 896-byte memory still carries that mark's RGB color and source-view center `(x, y)`. It tests accessibility with a new supervised reader; it does not retrain the original writer or image decoder.

## Protocol

- Load `runs/first/graph.pt` and `runs/first/vector.pt` (writer seed 7). For each synthetic glyph, run its writer, round the memory to `uint8`, and give **only those 896 bytes** to a linear probe (4,485 parameters) or a 64-hidden-unit MLP (57,733 parameters). The graph's horizontal and vertical edges are flattened in fixed order. No source pixels, query, or image decoder activations reach either probe.
- Supervise five outputs: the mark's three RGB values in `[0, 1]` and its source-view pixel center `(x/31, y/31)`. Normalize inputs and labels using **probe training data only**. Use AdamW, batch 128, and three initialization/shuffle seeds (7, 107, 207). Hold out 128 **new objects**, seed 2,000,010, from both writer and probe training.
- Compare against the training-label mean (constant prior) and the trained probe evaluated with validation memories shuffled among objects. A separate probe is trained from images whose mark alone has been replaced by body color; its labels remain the original mark. This erasure control tells us how much position can be inferred from the body's geometry. Count exact `uint8` entries changed by erasure.
- The first probe used 2,048 objects from the writer's training seed and 600 steps. It overfit mark color, so a **post hoc follow-up** used 8,192 objects from fresh seed 3,000,010 and 1,200 steps. Both use the same held-out probe validation set. These are exploratory choices, not an untouched test or multiple independently trained writers.

## Results on the fresh 8,192-object probe run

Means over three probe initializations; lower is better. Position is mean Euclidean error in pixels. Full per-run metrics and shuffled controls are in [JSON](mark_probe_first_fresh8192.json).

| Byte memory / reader | Mark RGB MSE | Mark position error |
| --- | ---: | ---: |
| Constant training-label prior | 0.031341 | 2.459 px |
| Graph, linear probe | 0.026882 | 0.806 px |
| Graph, MLP probe | **0.024140** | **0.723 px** |
| Graph with mark erased, MLP probe | 0.032990 | 0.982 px |
| Graph, shuffled validation memories, MLP probe | 0.042086 | 3.312 px |
| Vector, MLP probe | 0.024716 | 0.850 px |
| Vector with mark erased, MLP probe | 0.032360 | 1.009 px |

The graph MLP reduces mark-color MSE about 23% relative to the constant prior; erasing the mark eliminates that gain. Its color RMSE is still about 0.155 on `[0, 1]`, so this is **partial recovery**, not exact identity reproduction. Erasing the mark increases its position error from 0.723 to 0.982 px, but the erased probe still beats the 2.459 px prior: body geometry already predicts much of the mark's location. Shuffling validation memories destroys both gains.

Mark erasure changes an average of **75.09 / 896 graph bytes** per held-out scene (all 128 scenes had a change). The vector changes **26.26 / 896**. A changed byte alone does not establish recoverable mark color; the supervised held-out probe and erasure control supply that evidence. The graph and vector have different trained architectures, so this run cannot establish that graph storage is generally superior.

The earlier [2,048-object probe run](mark_probe_first_2048.json) explains why the larger follow-up mattered: graph MLP training color MSE was 0.01165, but held-out color MSE was **0.03551**, worse than the 0.03133 prior. With fresh 8,192-object probe training, held-out MSE fell to **0.02414**. The original image renderer still mostly ignores the mark in its output. That points to readout training and the pixel-MSE objective as the next bottleneck, while leaving open how much detail the same byte budget can ultimately preserve.

## Reproduce

From the repository root after `python -m pip install -e .`:

```bash
python probe_memory.py --graph-checkpoint runs/first/graph.pt --vector-checkpoint runs/first/vector.pt --output runs/probe_2048.json --seed 7 --train-count 2048 --val-count 128 --steps 600 --batch-size 128 --probe-seeds 3 --device cpu
python probe_memory.py --graph-checkpoint runs/first/graph.pt --vector-checkpoint runs/first/vector.pt --output runs/probe_fresh8192.json --seed 7 --probe-train-seed 3000010 --train-count 8192 --val-count 128 --steps 1200 --batch-size 128 --probe-seeds 3 --device cpu
```

The next image-model gate is to train a mark-aware *reader* on the same frozen graph, then evaluate the rendered mark on unseen objects against the equal-byte vector and erasure controls. A positive probe result is an existence demonstration for supervised access, not a claim that the current RGB decoder can already show the detail.
