# Stage runbooks

`pipeline.md` is the authoritative numbered stage map, while `experiment_protocol.md` defines the frozen methodological and execution rules. These files remain at the repository root because the shared utility and run manifests fingerprint those exact paths.

The runbooks below add operational detail where a stage has special configuration, cross-machine transfer, decision, or review requirements:

- [Stage 2 — Contextual preprocessing](stage_02_contextual_preprocessing.md)
- [Stage 8 — Contextual length, tuning, timing, and latency](stage_08_timing_pilot.md)
- [Stages 9–10 — Experiment II-A ablation](stages_09_10_context_ablation.md)
- [Stage 11 — Context and backbone decision gate](stage_11_backbone_selection.md)
- [Stage 14 — Direct fine-tuning](stage_14_direct_finetuning.md)
- [Stage 15 — Quantitative evaluation and training-strategy selection](stage_15_quantitative_evaluation.md)
- [Stage 16 — Qualitative error analysis](stage_16_error_analysis.md)
- [Stage 17 — Final model training](stage_17_final_models.md)

Not every stage needs its own README. For routine stages, use the commands and requirements in `pipeline.md`, this root `README.md`, and the protocol. If a focused runbook becomes necessary for another stage, add it here rather than creating another root-level `README_STAGE*.md` file.

For VS Code/Colab artifact transfers, see [the Colab artifact handoff checklist](../COLAB_ARTIFACT_HANDOFF.md).
