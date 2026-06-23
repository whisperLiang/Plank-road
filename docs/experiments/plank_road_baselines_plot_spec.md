# Plank-road Baseline Plot Specification

All labels and legends are English. PDF and PNG are emitted together. A
missing required metric skips the figure and records the reason in
`plot_report.json`; no placeholder series are drawn.

When `normalization_report.json` declares
`accuracy_definition: teacher_supervised_f1`, F1 axes and panels use the
explicit labels “Teacher-supervised F1” and “Average teacher-supervised F1”.

| Figure | Compared methods | Purpose | Input CSV | X-axis | Y-axis | Grouping | Missing-data behavior | Expected insight |
|---|---|---|---|---|---|---|---|---|
| Fig. 1 Accuracy Over Time | Three current methods | Show recovery after drift | `frame_metrics.csv`, `adaptation_events.csv` | Frame ID | F1 or mAP | Scenario facets; method lines | Skip if neither real F1 nor mAP exists | Plank-road restores accuracy promptly |
| Fig. 2 Adaptation Timeline | Three current methods | Compare first-update response | `adaptation_events.csv`, `latency_breakdown.csv` | Seconds since first trigger/event | Method rows | Scenario facets and event markers | Skip without timestamped adaptation events | Trigger-to-update stages differ by method |
| Fig. 3 Tradeoff | Three methods; optional external Ekya | Compare accuracy, latency, and communication | `summary.csv` | Mean adaptation latency | Mean F1/mAP | Method points; upload controls bubble size | Downgrade to latency-upload when accuracy is missing | Plank-road balances the three costs |
| Fig. 4 Upload Breakdown | Three current methods | Explain upload composition | `upload_breakdown.csv` | Method | Bytes | Stacked measured components | Omit missing components and report partial data | Plank-road uses features plus selective raw samples |
| Fig. 5 Latency Breakdown | Three current methods | Explain adaptation response time | `latency_breakdown.csv` | Method | Milliseconds | Stacked measured stages | Omit missing stages and report partial data | Split-tail training changes the critical path |
| Fig. 6 Multi-edge Scalability | Three methods; optional external Ekya | Show scaling with edge count | `summary.csv` | Edge count | Best available common metric | Scenario facets and method lines | Require at least two edge-count points per plotted method | Plank-road remains scalable with more edges |
| Fig. 7 Resource Timeline | Three current methods | Show queueing and execution | `resource_timeline.csv`, `adaptation_events.csv` | Time | Run/edge rows | Measured stage intervals | Skip if consecutive timestamps cannot define intervals | GPU requests wait and execute visibly |
| Fig. 8 Component-style Summary | Three current methods; optional external Ekya | Paper overview | `summary.csv`, `upload_breakdown.csv`, `latency_breakdown.csv` | Method | Accuracy, latency, bytes | Three side-by-side panels | Require all three metrics for every displayed current method | Overall method differences are immediately visible |

Method order is Pure Edge, Accuracy-Trigger, Plank-road. The same color is used
for a method in every figure. Ekya is ignored unless both
`--include_external_ekya` and `--external_ekya_summary` are supplied.
