# HAC-KT

History-based Additive Correction for Knowledge Tracing (HAC-KT) adjusts the estimated probability that a student will answer the next item correctly. It adds an intercept and learned one- and two-feature functions to the logit from a fixed upstream prediction model. These functions use item difficulty estimated from training data and the student's history before the target response.

## Run the example

Use Python 3.11 in a virtual environment. From the repository root, run the following commands.

```bash
python -m pip install -r requirements.txt
python example.py
```

The example uses generated inputs. It changes the answer and observed behavior on the target row and checks that the features available before that response stay identical. It then fits a short example model, compares table lookup with native EBM predictions and scores 560 held-out generated rows. The example reduces the number of rounds and interactions to run quickly.

## Fit and predict with an upstream model

Prepare a NumPy `.npz` file for each split. Use the same upstream model and the same feature names in the same order for both splits. Each row refers to one target response. Derive difficulty and normalization from the training split. During fitting, use the difficulty estimate from the other training-student folds for each training row. Within each student's ordered records, compute features before adding the current result or behavior to the history.

| Array | Contents |
| --- | --- |
| `x` | A finite, two-dimensional array of pre-response features, with one row per target |
| `feature_names` | A one-dimensional NumPy Unicode array of column names |
| `known` | One boolean or 0/1 flag per row indicating whether the current skill label is available |
| `logit` | The fixed upstream model's log odds of a correct response, aligned with each row |
| `target` | A 0/1 response label for each training row |
| `students` | An integer student identifier for each training row, used to construct the EBM training bags |

```bash
python correction.py fit train.npz fitted_model --seed 42
python correction.py predict test.npz fitted_model predicted_probability.npy
```

The fit command uses the paper's correction settings, including up to 64 learned feature pairs and 2,000 rounds. It fits separate branches for available and missing skill labels and saves the retained training columns, branch mapping, intercepts, bin boundaries and learned functions. A branch with fewer than 500 observations or a single outcome uses the other fitted branch. The model manifest records which branches were fitted. Scoring combines the saved functions with the upstream logit supplied for the same row. The output is an array of probabilities in the input row order.

## Study source

The historical source files in `study/src` retain their original research-project paths. The manifest records the SHA-256 digest of each file and the origin of each function extracted into the portable interface.

| Component | Source |
| --- | --- |
| ASSISTments 2012 preparation and sequence windows | [Data preparation](study/src/prepare_sequences.py) |
| ASSISTments 2009 preparation, difficulty estimation and prior counts | [Original adapter](study/src/mica_generalization_20260930/prepare.py) |
| Bridge to Algebra 2006–2007 preparation and target construction | [BD2006 adapter](study/src/hackt_bd2006_20261123/bd_data.py) |
| Student, skill and item histories | [Feature assembly](study/src/mica_generalization_20260930/features.py), [difficulty-adjusted histories](study/src/mica_transparent_20260923/state_features.cpp), [uncertainty updates](study/src/mica_transparent_20260923/uncertainty_features.cpp) |
| Upstream sequence model and training | [Model](study/src/mica_generalization_20260930/frozen_train_eckt.py), [training and inference](study/src/mica_generalization_20260930/neural.py) |
| Main corrections and scoring | [ASSISTments 2012](study/src/mica_crossfit_20260924/full_crossfit_readout.py), [ASSISTments 2009](study/src/mica_generalization_20260930/readout.py), [BD2006](study/src/hackt_bd2006_20261123/worker.py), [table export and scoring](study/src/mica_transparent_20260923/gam_search.py) |
| Ablations and paired comparisons | [Historical feature ablation](study/src/mica_hist_completion_20261014/historical_ablation.py), [interaction-only refits](study/src/hackt_pairs_only_20261124/experiment.py), [paired analysis](study/src/hackt_pairs_only_20261124/finish.py) |

The commands above run with the packages in `requirements.txt`. The historical experiments used separate Linux/GPU training scripts, data splits, feature caches, upstream checkpoints and saved logits. Reproducing the manuscript's dataset-level scores requires those inputs. The study archive recorded Python 3.10.8, PyTorch 2.1.2+cu121 and interpret-core 0.7.7 for the BD2006 training run. Complete upstream training configurations for two ASSISTments 2012 seeds were unavailable. The public repository contains source and a generated example. Individual student records, local checkpoints and saved per-response predictions are outside this release.

Original dataset information is available from the providers. [ASSISTments 2012–2013](https://sites.google.com/site/assistmentsdata/datasets/2012-13-school-data-with-affect) · [ASSISTments 2009–2010](https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data) · [KDD Cup 2010](https://kdd.org/kdd-cup/view/kdd-cup-2010-student-performance-evaluation/Data)
