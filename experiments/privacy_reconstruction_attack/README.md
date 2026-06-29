# Privacy Reconstruction Attack Offline Experiment

This directory contains an isolated offline experiment for measuring whether
Plank-road privacy leakage scores correspond to actual reconstruction leakage.
It does not modify or hook into online inference, continual-learning training,
or baseline scheduling.

The experiment evaluates four split points with target privacy leakage scores
`0.8`, `0.6`, `0.4`, and `0.2`. When a split point is set to `auto`, the
collector enumerates TorchLens split candidates and selects the nearest unique
candidate using Plank-road's existing score definition:

```text
privacy_leakage_score = 1 - edge_parameter_ratio
```

The resolved split IDs are written to `resolved_split_points.json` and reused by
later attack scripts.

## Attack Methods

Pixel optimization DRA directly optimizes an image tensor so that the split
prefix produces boundary features matching the target payload. It trains no
decoder, generator, inverse model, or reconstructor.

DRAG guided diffusion DRA optimizes Stable Diffusion latents with the same
boundary-feature matching guidance. It uses Stable Diffusion as an image prior,
but it still does not train an inverse decoder or attack network.

InfoDecom-style inverse model attacks are intentionally excluded because the
goal here is to measure leakage from cloud-visible boundary payloads without
introducing a separately trained inverse model. DRAG++ is also excluded because
it adds inverse-network initialization beyond the no-training attack setting.

## Full Run

```bash
# 1. Collect attack targets
python experiments/privacy_reconstruction_attack/collect_attack_targets.py \
  --yaml_path ./config/config.yaml \
  --config experiments/privacy_reconstruction_attack/configs/privacy_reconstruction.yaml \
  --video_path ./video_data/road.mp4 \
  --output_dir ./outputs/privacy_reconstruction/targets \
  --num_frames 100 \
  --frame_stride 5 \
  --device cuda:0

# 2. Run Pixel optimization DRA
python experiments/privacy_reconstruction_attack/pixel_dra_attack.py \
  --config experiments/privacy_reconstruction_attack/configs/privacy_reconstruction.yaml \
  --targets_dir ./outputs/privacy_reconstruction/targets \
  --output_dir ./outputs/privacy_reconstruction/pixel_dra \
  --device cuda:0

# 3. Run DRAG
python experiments/privacy_reconstruction_attack/drag_attack.py \
  --config experiments/privacy_reconstruction_attack/configs/privacy_reconstruction.yaml \
  --targets_dir ./outputs/privacy_reconstruction/targets \
  --output_dir ./outputs/privacy_reconstruction/drag \
  --device cuda:0

# 4. Summarize evaluation
python experiments/privacy_reconstruction_attack/evaluate_privacy_score.py \
  --config experiments/privacy_reconstruction_attack/configs/privacy_reconstruction.yaml \
  --pixel_dir ./outputs/privacy_reconstruction/pixel_dra \
  --drag_dir ./outputs/privacy_reconstruction/drag \
  --output_dir ./outputs/privacy_reconstruction/results

# 5. Plot figures
python experiments/privacy_reconstruction_attack/plot_privacy_reconstruction.py \
  --results_dir ./outputs/privacy_reconstruction/results \
  --pixel_dir ./outputs/privacy_reconstruction/pixel_dra \
  --drag_dir ./outputs/privacy_reconstruction/drag \
  --output_dir ./outputs/privacy_reconstruction/figures
```

## Outputs

Target collection writes one subdirectory per split and one subdirectory per
sample. Each sample contains `raw_frame.png`, `model_input_tensor.pt`,
`boundary_payload.pt.gz`, `boundary_feature.pt`, prediction JSON files, and
`metadata.json`.

Each attack writes `recon.png`, `raw.png`, `metrics.json`, and
`feature_loss_curve.csv` per sample. Evaluation produces
`pixel_dra_per_sample.csv`, `drag_per_sample.csv`, `summary_by_score.csv`, and
`score_correlation.json`.

The plotting script generates reconstruction grids and score-vs-leakage curves:
`reconstruction_grid.png/pdf`, `score_vs_object_f1.png/pdf`, and
`score_vs_actual_leakage.png/pdf`.

