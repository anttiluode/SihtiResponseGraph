# Sihti Response Graph

A small image model in which **the observation writes the conductances of a graph** and a later question excites that graph. The read starts from fresh, image-independent activity. It is a measured first step toward the question raised by [Sihti](https://github.com/anttiluode/Sihti): can experience change not only what remains visible, but how a future input propagates?

The first task is deliberately narrow: learn to output one of four **2D quarter-turns** of an asymmetric colored 32×32 glyph. This is a trained image generator and a mechanism test. It does **not** infer a never-observed 3D backside, create photorealistic images, or claim a compression win. Those are different experiments.

## What is stored

| Model | Per-image state after seeing a picture | Read from a cleared state |
| --- | --- | --- |
| Response graph | 8 learned conductance channels on every undirected neighbor edge of an 8×8 lattice | A query-dependent spatial ping propagates for 8 lazy random-walk steps; responses at depths 0, 2, 4 and 8 feed a small RGB decoder |
| Vector comparison | 896 learned scalar activations | A query embedding and the vector feed a comparable RGB decoder |

There are 8×7 horizontal edges and 7×8 vertical edges; `112 × 8 = 896` byte-valued conductances. Both memories are rounded to actual `uint8` tensors at evaluation. The shared network weights are *not* charged per image; their parameter counts are reported separately. The graph reader receives only those edge values and a query, never the source pixels or a prior read's activation. A zero-visibility observation leaves a previous graph unchanged, including its byte representation. Masked writes zero hidden pixels before encoding; the pilot trains only on full images, so partial-view quality is unmeasured. The encoder currently assumes views have already been put into the same coordinates; it does not track a moving camera.

The graph is inspired by Sihti's fixed neighbor pattern and image-dependent weights. This implementation trains the writer and RGB readout, whereas Sihti's original own-EQ graph is derived directly from pixel similarity. Graph memory and input-conditioned filters also have extensive prior art; the claim under test here is this particular *write → clear → ping → image* behavior at a measured memory cost.

## Run

Python 3.11 or newer. For an NVIDIA GPU, install a [PyTorch build compatible with your CUDA setup](https://pytorch.org/get-started/locally/) first. Then:

```bash
python -m venv .venv
source .venv/bin/activate
# On Windows PowerShell, use .venv\Scripts\Activate.ps1 instead.
python -m pip install -e .
python -m unittest discover -s tests -v
python train.py --steps 1000 --batch-size 32 --device auto --output-dir runs/first
```

CPU smoke run:

```bash
python train.py --steps 2 --batch-size 4 --train-count 16 --val-count 8 --device cpu --output-dir runs/smoke
```

After training, inspect `runs/first/metrics.json` and `runs/first/predictions.png`. Four columns show input, exact target, graph answer, and vector answer on held-out shapes. The `.pt` checkpoints hold the learned model weights. To ask the graph about a PNG:

```bash
python predict.py runs/first/graph.pt picture.png runs/first/query.png --query 1
```

`--query 0,1,2,3` denotes counterclockwise quarter-turns from the input. A real photo is outside the synthetic training distribution, so the resulting pixels are an exploratory response rather than reliable geometry. `predict.py` resizes the input to 32×32.

## Experimental contract

- `GlyphViews` builds asymmetric objects with a colored mark from a fixed seed. Train and validation use different seeds. Training both models on each identical batch keeps examples, targets, optimization steps, and loss aligned.
- Validation computes pixel MSE and PSNR on 128 held-out objects by default, broken out by requested quarter-turn. Its graph/vector states are quantized before read. A separate fixed-graph control gives every validation object the same quantized graph: the mean over the first 64 **training** objects.
- `metrics.json` records the actual per-instance byte count, shared parameter count, and evaluation time. Timing here is descriptive, not a matched-compute comparison. The visual panel and checkpoint accompany the metrics in the ignored `runs/` directory.
- Swapping two memories under the same ping, repeated read after clearing, masked writes, writer gradients, query validation, and checkpoint loading have behavior tests. CI also runs a two-step CPU smoke train.

No gate asserts that the response graph beats the vector. Compare `graph.mse`, `vector.mse` and `fixed_graph.mse`. If graph and fixed graph agree, the image-specific conductances have contributed little under this task. If vector wins, storing the information in the operator has not earned its cost here.

### First pilot (one seed)

With seed 7, 1,200 steps, 1,024 training objects and 128 held-out objects, the quantized graph scored MSE **0.023825** (16.23 dB), the equal-byte vector **0.025985** (15.85 dB), and the fixed-graph control **0.033369** (14.77 dB). The graph did better on three turned views and worse on the unrotated view. The [exact metrics](results/pilot_1200.json) and [four-example contact sheet](results/pilot_1200.png) are preserved. Both methods make blurred predictions and lose the tiny identity mark. This one-seed pilot tests the mechanism; it does not establish a robust advantage or solve identity preservation. The positive conductance floor prevents quantization from erasing weak edges: on this pilot float and byte graph MSE differ by less than 0.000001.

## Next gate if the mechanism matters

Use sequences with an actual hidden side: one group sees its distinctive mark before occlusion, another never sees it. A trained model should recover the observed mark and represent uncertainty about a never-observed one. This v0 only rotates a planar image; the next gate needs multiview observations, a camera-coordinate update rule, and a probabilistic target. A frozen vision teacher could then add identity-sensitive supervision, but it cannot provide evidence about a particular unseen mark.

The [design](docs/superpowers/specs/2026-10-13-response-graph-design.md) and [implementation plan](docs/superpowers/plans/2026-10-13-response-graph.md) record the v0 boundary and the tests.
