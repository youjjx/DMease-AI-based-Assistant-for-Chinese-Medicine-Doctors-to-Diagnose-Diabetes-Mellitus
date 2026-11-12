# Non-clinical KAN/PPO checkpoint

`nonclinical_kan_ppo.pt` contains trained KAN actor/critic weights produced by the repository's PPO trainer. Training uses the included, manually constructed knowledge graph, with 50 iterations, 8 episodes per iteration and seed 42. See `model_manifest.json` for the command and SHA-256 digest.

The application loads this checkpoint by default. A local `checkpoints/kan_ppo.pt` takes precedence when present; `DMEASE_CHECKPOINT` explicitly overrides either path. Entity ordering and knowledge version must match. Failed or incompatible loading switches to graph scoring with an explicit status message.

The weights support reproducible engineering verification. They are not clinically validated and are not a reproduction of performance on the paper's private clinical datasets.
