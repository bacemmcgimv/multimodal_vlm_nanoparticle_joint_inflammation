# LAMSS-VLM

LAMSS-VLM scores Hoffa synovitis, effusion synovitis, and bone marrow lesions from multi-sequence three-dimensional knee MRI. A hierarchical volumetric transformer maps MRI volumes to a stochastic latent representation. A frozen BiomedCLIP PubMedBERT text tower supplies normalized embeddings for a 100-phrase OMERACT MOAKS vocabulary. Cross-attention, cumulative ordinal regression, language alignment, and acquisition-source information suppression connect the two representations.

## Scientific scope

The training objective is

```text
L = L_CORAL - beta I(Z; text(Y)) + gamma I(Z; source)
```

The vision path emits a 768-dimensional feature and parameterizes a diagonal Gaussian latent. The bridge maps the visual and frozen text representations to 512 dimensions with eight attention heads and four layers. Three cumulative-link heads estimate grades zero through three. MINE estimates acquisition-source information during normal operation, while CLUB is selected if the MINE estimate becomes non-finite or exceeds the configured stability range.

The active phrase set contains 10 phrases before epoch 30, 25 phrases from epoch 30, and 50 phrases from epoch 60. The language weight follows `0.5 * (1 - exp(-epoch / 10))`. The site weight follows `min(1, epoch / 20)`.

## Repository map

```text
code/lamss_vlm/config.py          configuration parsing and validation
code/lamss_vlm/data.py            manifest loading and volume batches
code/lamss_vlm/information.py     MINE, CLUB, gradient reversal, information terms
code/lamss_vlm/model.py           volumetric transformer, bridge, ordinal heads
code/lamss_vlm/objectives.py      CORAL and composite optimization objective
code/lamss_vlm/phrase_bank.py     structured MOAKS phrase vocabulary
code/lamss_vlm/preprocessing.py   resampling, normalization, physical augmentation
code/lamss_vlm/protocols.py       cohort partitions and source-balanced sampling
code/lamss_vlm/runtime.py         distributed state and atomic snapshots
code/lamss_vlm/schedules.py       optimizer, warm-up, cosine and weight schedules
code/lamss_vlm/statistics.py      AUC, DeLong, bootstrap, agreement and calibration
code/lamss_vlm/train.py           training command
code/lamss_vlm/trainer.py         distributed mixed-precision training loop
configs/                          main study and experiment-specific settings
scripts/prepare_manifest.py       controlled-data manifest preparation
dataset_links.txt                 verified official dataset entry points
```

No cohort images, annotations, subject identifiers, credentials, or institution-local paths are distributed in this repository.

## Installation

Python 3.11 is required. The package pins the numerical and training dependencies used by the command-line entry point.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

The principal packages are PyTorch 2.3.1, NumPy 1.26.4, pandas 2.2.2, scikit-learn 1.5.1, and PyYAML 6.0.2.

## Controlled data access

OAI and MOST require acceptance of their respective data-use terms. The official access pages were checked on 9 August 2026 and are recorded in `dataset_links.txt`. Dataset licenses are data-use agreements rather than open-source software licenses. This project does not automate account creation or acceptance of those agreements.

After access is approved, convert each DICOM series to NIfTI and create a local CSV with the following fields:

```text
participant_id,volume_path,site_id,hoffa_grade,effusion_grade,bml_grade,phrase
```

The six source identifiers are `oai_md`, `oai_oh`, `oai_pa`, `oai_ri`, `most_al`, and `most_ia`. Each participant must belong to exactly one partition. Grades must be integers from zero through three.

Prepare a validated training manifest with:

```bash
python scripts/prepare_manifest.py \
  --input data/cohort.csv \
  --output data/manifest.csv
```

Create the frozen text embedding bank with:

```bash
python scripts/prepare_phrase_bank.py --output data/moaks_phrases.pt
```

The preprocessing path resamples each channel to 0.6 mm isotropic voxels, extracts a knee-centered `160 x 160 x 160` field of view, clips intensities at the first and ninety-ninth percentiles, and applies channel-wise z-scoring. Training transforms include spatial flips, elastic deformation, bias-field jitter, contrast polarity inversion with probability 0.25, and relaxivity rescaling over 3.8, 5.6, 9.2, 14.1, and 19.0.

## Main training run

The main configuration uses 60 epochs, AdamW, learning rate `3e-4`, weight decay `1e-2`, a two-epoch warm-up, cosine decay, batch size 16 per process, gradient accumulation of two, four A100 80 GB devices, and 20 seeds. The effective batch size is 128 examinations.

```bash
torchrun --standalone --nproc-per-node=4 \
  -m lamss_vlm.train \
  --config configs/main.yaml
```

The 20-seed study requires approximately 71.4 A100-hours when runs are executed sequentially. Measured peak training allocation is 38.2 GB per device. Local storage must accommodate the controlled source images, converted volumes, manifests, metric histories, and atomic training snapshots; the source custodians determine the raw archive size.

Training uses equal source contributions within every mini-batch. Exhausted source pools are sampled with replacement. Participant-level partitions use 75% for training, 10% for validation, and 15% for held-out evaluation, stratified by source, sex, age decile, Kellgren-Lawrence grade, and all three MOAKS grades.

## Experiment configurations

The `configs` directory contains separate settings for component removal, fixed vocabulary sizes, OAI-to-MOST and MOST-to-OAI transfer, polarity and relaxivity studies, pretraining comparisons, explicit site embeddings, cohort scaling, uncertainty estimation, ensembles, and beta, gamma, and learning-rate sweeps.

Run a configuration by replacing the main path:

```bash
torchrun --standalone --nproc-per-node=4 \
  -m lamss_vlm.train \
  --config configs/ablation_no_language.yaml
```

Cross-cohort runs use four OAI acquisition sources for training and two MOST sources for held-out evaluation, then reverse the direction. Contrast-transfer settings cover T1-positive gadolinium, intermediate relaxivity, SPION, and ferumoxytol conditions. Ensemble settings aggregate five, ten, or twenty independently seeded models.

## Statistical protocol

Every acquisition source is reported separately. Confidence intervals use 1,000 case-wise bootstrap iterations stratified by source. AUC comparisons use paired DeLong statistics with Holm-Bonferroni correction over the five primary hypothesis families. Agreement measures include ICC(3,1), quadratic-weighted Cohen kappa, Spearman correlation, and smallest-detectable-difference crossing rate.

The primary reference values are cross-site Hoffa AUC `0.892`, effusion AUC `0.905`, BML AUC `0.878`, cross-site AUC standard deviation `0.025`, and weighted agreement of at least `0.81`. The statistical module also computes expected calibration error, maximum calibration error, Brier score, negative log likelihood, reliability gaps, Bland-Altman limits, subgroup gaps, fixed-effect seed aggregation, power-law cohort scaling, and contrast AUC retention.

## Output integrity

Metric histories are written as JSON lines. Snapshots include the model, optimizer, scheduler, scaler, epoch, optimizer step, and random seed state. Snapshot replacement is atomic so an interrupted write does not replace the last complete state. Resumed training restores the stored seed state before continuing.

Restricted data must remain outside version control. Manifest identifiers should be study identifiers rather than names or direct identifiers. Logs must not contain accession numbers, account details, tokens, or local absolute paths.

## License

The software is distributed under the MIT License. Dataset access and use remain governed by the OAI and MOST custodians.
