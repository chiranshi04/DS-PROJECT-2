# Experiments and research questions

This table maps the questions in the project brief to the implemented protocol, configuration, and command. A command can only produce a scientific result after the required licensed data, manifests, checkpoint, and prediction artifacts exist.

| Research question | Protocol | Configurations | Command | Primary artifacts |
|---|---|---|---|---|
| How well does B1 separate genuine and synthetic speech under the training distribution? | P1 in-distribution, speaker-disjoint | `configs/data/protocols/P1.yaml`, `configs/experiments/p1_b1_small.yaml` | `make train EXP=p1_b1_small`; `python -m indivoice.eval --run runs/p1_b1_small --protocol P1 --split test` | `runs/p1_b1_small/`, protocol result JSON/CSV |
| How does the classical LFCC+GMM baseline compare with B1? | P1 | `configs/experiments/p1_b0_gmm.yaml` | `make train EXP=p1_b0_gmm` then use the B0 fitting/evaluation artifact path documented by the resulting run | B0 run and evaluation JSON/CSV |
| Does performance transfer to a generator absent from training? | P2 leave-one-generator-out | `configs/data/protocols/P2.yaml`, `configs/experiments/p2_b1_logo.yaml` | `make train EXP=p2_b1_logo`; `python -m indivoice.eval --run runs/p2_b1_logo --protocol P2 --split test` | Per-generator table and P2 result JSON/CSV |
| Does performance transfer to a language absent from training? | P3 leave-one-language-out | `configs/data/protocols/P3.yaml`, `configs/experiments/p3_b1_lolo.yaml` | `make train EXP=p3_b1_lolo`; `python -m indivoice.eval --run runs/p3_b1_lolo --protocol P3 --split test` | Language-transfer matrix and P3 result JSON/CSV |
| What happens when training and testing languages are paired differently? | P4 pairwise language transfer | `configs/data/protocols/P4.yaml`, `configs/experiments/p4_pairwise.yaml` | `make train EXP=p4_pairwise`; `python -m indivoice.eval --run runs/p4_pairwise --protocol P4 --split test` | Pairwise matrix and P4 result JSON/CSV |
| What false-positive rate is observed on independent genuine corpora? | P5 external-test-only | `configs/data/protocols/P5.yaml`, `configs/eval.yaml` | `python -m indivoice.eval --run runs/p1_b1_small --protocol P5 --split test` | Genuine-only FPR by corpus/language with clustered CI |
| Does optional silence trimming change results? | P1 ablation | `configs/experiments/silence_trim_on_off.yaml`, `configs/features.yaml` | `make train EXP=silence_trim_on_off`; evaluate both settings with P1 | Ablation result JSON/CSV and comparison plot |
| Does codec augmentation change robustness? | Robustness paired comparison | `configs/experiments/codec_aug_on_off.yaml`, `configs/robustness.yaml` | `make train EXP=codec_aug_on_off`; `make robustness SCORES=... THRESHOLD=... OUT=results/robustness` | Degradation-level JSON/CSV and EER plot |
| Does model size trade accuracy for efficiency? | P1 plus export benchmark | `configs/experiments/model_size.yaml`, `configs/export.yaml` | Train tiny/small, export each, then `python scripts/benchmark.py --model-dir releases --runs 100 --warmup 10` | Parameter/file-size/latency/memory table |

## Fixed evaluation conventions

The evaluation defaults are in [`configs/eval.yaml`](../configs/eval.yaml): ROC-AUC, EER, FPR/FNR at a development-fixed threshold, minDCF, ECE, 1,000 speaker-clustered bootstrap resamples, mean window-logit aggregation, and development-only temperature scaling. A test set must not be used to tune either the operating threshold or calibration temperature.
