# Can a reader put the retained mark back into the image?

The first CUDA writer stores one 32×32 colored glyph in **896 quantized bytes**. Its original queried RGB images are blurry and usually omit the five-pixel identity mark. The [frozen-memory probe](mark_probe_first.md) found that a supervised reader could recover some of the mark's color and position. This experiment asks whether that information can be rendered in the output image.

## Fixed writers and trained readers

The graph and equal-byte vector start from the user's `runs/first/{graph,vector}.pt` checkpoints (writer seed 7). Their encoders and the graph's edge writer remain unchanged, verified tensor for tensor after training. Each example writes its observed image to a `uint8[896]` memory. Subsequent readers receive **only those bytes and the quarter-turn query**; target images, mark masks, and mark labels are used in training and scoring, never at inference.

1. Fine-tune each checkpoint's image readout for 1,200 steps with whole-image pixel MSE plus `0.1 ×` MSE on the exact mark pixels. The graph trains its ping and RGB decoder (27,875 parameters); the vector trains its RGB decoder (21,779 parameters).
2. Train the same 896→64→5 mark head for each memory (57,733 parameters). It predicts source-view RGB and mark center from all 896 bytes. The query rotates that center; a fixed soft five-pixel cross composites the predicted color over the tuned RGB image. This explicit shape prior is part of the reader, not a discovery about natural image structure.
3. For each architecture, train a **separate image readout and mark head on mark-erased input memories**, with the original targets and labels. This gives an in-distribution control for what geometry and learned priors can recover without the actual mark. Also evaluate the intact-trained pair after erasing the test mark as a causal input intervention, and compare the head with a constant training-label prior.

The main reader train set has 8,192 generated objects (seed 5,000,010), the development set 128 (seed 5,000,011), and the reserved test set 256 (seed 6,000,010). A preceding exploratory development run used separate seeds 4,100,010–4,100,012 to choose the explicit mark head after weighted RGB fine-tuning alone failed to make the mark visible. An initial process wrote results for the reserved seed before the erased-trained control was added; those metrics were **not inspected**. The controlled run was then completed and its results inspected. Results below are from **one writer training seed and one reader/head initialization**, not a repeated-seed estimate.

## Fresh 256-object test result

Lower MSE is better. “Mark RGB MSE” compares the rendered pixels *at the exact target mark mask* to their target colors. The erasure control is separately trained on erased memories rather than an intact-trained reader exposed only at test time. [Full metrics](mark_reader_first/metrics.json) include the latter intervention, local-body error, PSNR, and the development split.

| Reader output | Graph image MSE | Graph mark RGB MSE | Vector image MSE | Vector mark RGB MSE |
| --- | ---: | ---: | ---: | ---: |
| Original image reader | 0.023340 | 0.046524 | 0.025249 | 0.059417 |
| Mark-weighted RGB reader | 0.020007 | 0.034939 | 0.023067 | 0.038404 |
| Weighted RGB + byte-to-mark head | **0.019980** | **0.026436** | **0.023030** | **0.028318** |
| Separately trained on erased mark | 0.019932 | 0.033082 | 0.023047 | 0.034843 |

The explicit graph head's own color MSE is **0.023995** and source-position error is **0.772 px**, against **0.030331** and **2.548 px** for the constant training-label prior. The separately erased-trained head scores **0.032394** and **1.002 px**. The vector head gives color MSE **0.023736**, position **0.871 px**, versus erased-trained **0.031834**, **1.029 px**. Thus the intact memories contribute mark identity beyond what these readers infer from body geometry and label priors. Both memory formats support the effect; this does not show that graph storage is intrinsically better.

The [full prediction panel](mark_reader_first/predictions.png) and [nine-pixel mark crops](mark_reader_first/mark_crops.png) show the practical limit. The added crosses are often visible, but some are displaced or the wrong color, and the surrounding object remains blurred. The graph's aggregate mark-contrast projection is 0.255 against an erased-trained 0.165; this statistic is **not** a fraction of correctly identified marks and can be positive for a color prior. The per-object mark-color MSE and erased-trained comparison carry the narrower recoverability claim.

## Reproduce and render

Install the package as in the repository README, then run:

```bash
python train_reader.py --graph-checkpoint runs/first/graph.pt --vector-checkpoint runs/first/vector.pt --output-dir runs/mark_reader_reproduction --train-seed 5000010 --dev-seed 5000011 --test-seed 6000010 --train-count 8192 --dev-count 128 --test-count 256 --steps 1200 --batch-size 128 --mark-weight 0.1 --device cpu
python predict_mark.py results/mark_reader_first/graph_reader.pt results/mark_reader_first/graph_mark_head.pt results/pilot_input.png runs/pilot_mark_query.png --query 2
```

The saved reader and head form a pair; inference rejects mismatched pair IDs, architectures, writer seeds, or writer-weight fingerprints. The published paired checkpoints are in [`mark_reader_first/`](mark_reader_first/). The model is trained only on synthetic colored 2D glyphs and queried quarter-turns. It does not infer an unseen side, retain an arbitrary natural-image identity, or create a general-purpose image generator.
