# Video-aware offline teacher replay accuracy

Teacher replay evaluates archived student detections after an experiment. It is
not part of the online edge/cloud path and its runtime is never added to
inference, upload, adaptation, annotation, or training latency.

The metric is **Teacher-supervised F1** (`teacher_supervised_f1`). Teacher
detections are pseudo labels, not human ground truth, so reports and figures
must not call this actual, real, or ground-truth accuracy.

## Workflow

Run all three methods with video-aware comparison and run IDs. For example:

```bash
python cloud_server.py \
  --yaml_path ./config/config.yaml \
  --mode main \
  --run_id plank_road_road_001 \
  --comparison_id exp_road_plankroad_vs_baselines_001

python edge_client.py \
  --yaml_path ./config/config.yaml \
  --mode main \
  --run_id plank_road_road_001 \
  --comparison_id exp_road_plankroad_vs_baselines_001 \
  --video_path ./video_data/road.mp4 \
  --headless
```

Use corresponding `pure_edge_road_001` and `accuracy_trigger_road_001` run IDs
for the baselines. After all artifacts are present:

```bash
python tools/experiments/evaluate_plank_road_baseline_teacher_accuracy.py \
  --comparison_dir results/experiments/exp_road_plankroad_vs_baselines_001 \
  --teacher_model rtdetr_x \
  --yaml_path ./config/config.yaml \
  --device cuda:0 \
  --update_manifest

python tools/experiments/normalize_plank_road_baseline_logs.py \
  --comparison_dir results/experiments/exp_road_plankroad_vs_baselines_001

python tools/experiments/plot_plank_road_baseline_figures.py \
  --normalized_dir results/experiments/exp_road_plankroad_vs_baselines_001/normalized \
  --figure_dir results/experiments/exp_road_plankroad_vs_baselines_001/figures
```

The evaluator writes `teacher_accuracy_<video_slug>.jsonl`, its report, and a
cache under `teacher_replay_cache/<video_slug>/<teacher_model>/`. Multiple
scenarios use `teacher_accuracy_all.jsonl`.

## Video identity and replay

`client.source.video_slug` and `client.source.scenario_name` take precedence.
For a fixed video they otherwise derive from the filename:

- `road.mp4` becomes `road`.
- `road_night-rain.mp4` becomes `road_night_rain`.

Slugs contain only lowercase letters, digits, and underscores. RTSP, camera,
and other URI sources require an explicit slug or scenario name.

Fixed files are replayed from the original video using the logged 1-based
`frame_index`. They do not require saved images. For a remote source, enable:

```yaml
client:
  source:
    scenario_name: north_gate
    video_slug: north_gate
    teacher_replay:
      save_sampled_frames: true
      jpeg_quality: 90
      queue_size: 64
      archive_chunk_max_bytes: 67108864
```

Sampled JPEGs are encoded by a bounded background writer and archived only
after the run. If snapshots are disabled, dropped, or missing, the evaluator
records the frames under `unreplayable_frames` or `failed_video_reads`; it does
not silently invent accuracy.

## Label alignment and outputs

New edge logs include the student label schema and class names. COCO teacher
labels are mapped by normalized class name into the student label space before
class-aware F1 is calculated. Unmapped teacher classes are omitted and counted
in the evaluator report. Legacy logs fall back to the student class names in
`--yaml_path`.

The accuracy JSONL keeps the existing `ACCURACY_FIELDS` schema. The first
implementation fills only `f1`; `map` and `window_accuracy` remain empty. mAP
is not synthesized from F1.
