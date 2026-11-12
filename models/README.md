# Public evidence policy

`public_evidence_kan_ppo.pt` is the current KAN actor/critic checkpoint. It uses actual public database associations, with query tasks split before training. See `public_model_manifest.json` for the hash and `public_evidence_evaluation.json` for the complete split, validation history, test and exclusion-stress results.

The selected checkpoint is from PPO iteration 40 after 60 warm-start epochs (seed42). Three training seeds were evaluated. The validation objective does not exceed the evidence-greedy baseline, so production defaults to that baseline. KAN/PPO remains selectable for transparent comparison and gradient inspection. Models are never selected by their test-set performance.

`nonclinical_kan_ppo.pt` is the historical graph-constructed regression checkpoint. The current application does not load it by default. It is only used by explicit legacy tests/workflows. Its original parameters remain in `model_manifest.json`.

Neither checkpoint establishes clinical validity. Current weights optimize evidence coverage, not symptom relief or binding affinity. No learned constitution effect is claimed.

## Fourth-round planning study

`planning_v4/` contains selected and warm-only snapshots for three architectures
(`residual_kan`, `residual_mlp`, `kan_without_prior`) and three seeds (42, 43, 44).
The 18 files represent nine training runs, not 18 independent runs. Checksums,
actor/critic parameter counts, every validation observation and the selected
iteration are recorded in `experiments/v4/planning_training/`.

A selected iteration of -1 means the initial gain prior remained best on validation;
0 means supervised warm-up; positive values identify PPO iterations. A completed
training run does not imply the learned update improved the policy. The optional
conservative override uses the frozen choice in `gating_selection.json`, which may
choose pure greedy, a warm snapshot or a selected snapshot. Margins are score
differences, not calibrated confidence probabilities.

These checkpoints require the matching graph fingerprint and fourth-round state
definition (complete requested-pair matrix, remaining budget and explicit STOP).
They are used through the dedicated research workspace and do not replace the
default patient-analysis model. Training and test requests are engineering
evidence-retrieval combinations, not clinical cases or contraindication annotations.
The source graph attribution remains in `data/README.md`.
