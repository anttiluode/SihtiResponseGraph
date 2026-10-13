# Pilot result: byte-valued graph memory on a planar image query

Protocol: `python train.py --steps 1200 --batch-size 32 --train-count 1024 --val-count 128 --seed 7 --device cpu --output-dir runs/local1200`. Python 3.12, PyTorch 2.14.1+cpu. Training and validation seeds are 7 and 1,000,010. This was a development pilot after a 400-step run exposed a quantization failure; it is **not** a preregistered or multi-seed result.

| Held-out metric, 128 objects | Response graph | Equal-byte vector | Fixed training-mean graph |
| --- | ---: | ---: | ---: |
| MSE, all queries | 0.023825 | 0.025985 | 0.033369 |
| PSNR, dB | 16.230 | 15.853 | 14.767 |
| Query 0 MSE | 0.017692 | 0.014229 | 0.034777 |
| Query 1 MSE | 0.025832 | 0.030899 | 0.032835 |
| Query 2 MSE | 0.026969 | 0.027707 | 0.032801 |
| Query 3 MSE | 0.024808 | 0.031105 | 0.033064 |
| State payload per image | 896 bytes | 896 bytes | 896 bytes |
| Shared learned parameters | 39,355 | 33,217 | 39,355 |

The learned per-image graph affects the answer: replacing it with a mean graph degrades held-out MSE. It outscored the vector overall in this single run but did worse when asked for the original, unrotated view. The [contact sheet](results/pilot_1200.png) shows why MSE alone is insufficient: both render blurry silhouettes and mostly lose the colored mark. These outputs are not an identity-preserving scene model or evidence for unobserved 3D geometry.

The initial 400-step pilot had graph MSE 0.0321 versus 0.0294 for the vector and 0.0362 for a fixed graph. Diagnostics found 31% of conductances rounded to zero at byte evaluation, moving graph MSE from 0.0260 in float to 0.0321 in bytes. We added a weak-conductance round-trip regression test and a positive floor to the writer. At 1,200 steps after this change, float MSE was 0.0238250 and byte MSE 0.0238254, with no zero-valued conductances. More seeds and different objects are needed before making a comparative performance claim.

Source data and parameters: [machine-readable pilot metrics](results/pilot_1200.json). The graph stores edge values only; the experiment does not account for checkpoint weight bytes per scene, since those weights are shared across all scenes. The vector and graph have different compute and parameter counts, which the JSON reports.
