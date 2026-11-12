# Public evidence policy

`public_evidence_kan_ppo.pt` is the current KAN actor/critic checkpoint. It uses actual public database associations, with query tasks split before training. See `public_model_manifest.json` for the hash and `public_evidence_evaluation.json` for the complete split, validation history, test and exclusion-stress results.

The selected checkpoint is from PPO iteration 40 after 60 warm-start epochs (seed42). Three training seeds were evaluated. The validation objective does not exceed the evidence-greedy baseline, so production defaults to that baseline. KAN/PPO remains selectable for transparent comparison and gradient inspection. Models are never selected by their test-set performance.

`nonclinical_kan_ppo.pt` is the historical graph-constructed regression checkpoint. The current application does not load it by default. It is only used by explicit legacy tests/workflows. Its original parameters remain in `model_manifest.json`.

Neither checkpoint establishes clinical validity. Current weights optimize evidence coverage, not symptom relief or binding affinity. No learned constitution effect is claimed.
