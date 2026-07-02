# DMease

DMease is a research codebase for the paper **"DMease: AI-based Assistant Tool for Traditional Chinese Medicine Doctors in Diagnosing and Treating Diabetes Mellitus"**.

The repository has been rebuilt around the method described in the paper:

1. Multi-source TCM text is normalized and converted into structured triples.
2. Triples are stored as a symptom-target-herb knowledge graph.
3. A PatientDB provides patient-ID based records.
4. The system predicts a TCM syndrome from symptoms.
5. A sequential single-herb policy ranks and selects herbs under compatibility constraints.
6. The output includes target coverage, marginal symptom relief, and provenance traces.

This code is for research and engineering validation only. It is not a medical device and must not replace licensed TCM doctors.

## Repository Structure

```text
configs/
  dmease.yaml                  # Main configuration
data/
  README.md                    # Expected data layout
  examples/                    # Compact sample data for demos and tests
prompts/
  triplet_extraction_zh.md     # LLM triplet extraction prompt
src/dmease/
  schemas.py                   # Patient, triple, diagnosis, prescription schemas
  patient_db.py                # Lightweight PatientDB
  semantic_parser.py           # Surface-to-concept mapping and JSON triplet parser
  knowledge_graph.py           # Symptom-target-herb KG and PMI-style retrieval
  diagnosis.py                 # Syndrome diagnosis module
  constraints.py               # Classical and patient-specific hard constraints
  kan.py                       # Lightweight KAN-compatible policy network module
  policy.py                    # Current greedy single-herb policy
  pipeline.py                  # End-to-end inference workflow
  cli.py                       # Command line interface
scripts/
  build_kg.py                  # Convert LLM JSON output to triples.jsonl
  infer_example.py             # Minimal inference example
app/
  streamlit_app.py             # Optional Streamlit UI
tests/
  test_pipeline.py             # Smoke tests for the example pipeline
```

## Installation

```bash
conda env create -f environment.yml
conda activate dmease
pip install -e .
```

If you do not want to create the full environment yet, the pure Python parts can be inspected first; the demo workflow uses the compact sample assets under `data/examples/`.

## Quick Start

List example patients:

```bash
dmease --config configs/dmease.yaml patients
```

Run the paper-style demo case:

```bash
dmease --config configs/dmease.yaml infer --patient-id P2025-0001
```

Or run directly from source:

```bash
PYTHONPATH=src python scripts/infer_example.py
```

On Windows PowerShell:

```powershell
$env:PYTHONPATH="src"; python scripts/infer_example.py
```

## Data Expected by the Paper

The paper describes the following sources:

- SymMap: TCM symptoms and gene/protein targets.
- STRING v12.0: human PPI network with confidence filtering.
- TCMSP: herbs and chemical ingredients.
- Classical literature: 30 foundational TCM texts after OCR correction.
- Guidelines: 5 national TCM guidelines.
- De-identified EMR: 108,746 TCM outpatient records.

Keep licensed or private datasets outside Git. For local experiments, put source files under `data/raw/`, then generate:

```text
data/processed/patients.jsonl
data/processed/triples.jsonl
data/processed/herb_target_affinity.csv
```

The included `data/examples/` files are compact sample assets aligned with the paper's case study. They make the repository runnable out of the box for demos, tests, and engineering review.

## Knowledge Graph Triples

The graph uses the ten relation templates from the paper:

| ID | Template |
| --- | --- |
| R1 | Herb-affect by-Target |
| R2 | Symptom-associated with-Target |
| R3 | Symptom-indicates-Syndrome |
| R4 | Syndrome-treatedBy-Herb |
| R5 | Herb-contains-Compound |
| R6 | Compound-targets-Gene |
| R7 | Gene-associatedWith-Symptom |
| R8 | Herb-contraindicatedWith-Herb |
| R9 | Syndrome-hasStage-DiseaseStage |
| R10 | Gene-participatesIn-Pathway |

LLM output should be strict JSON. See `prompts/triplet_extraction_zh.md`.

## Inference Policy

The paper describes a KAN policy trained with PPO for sequential single-herb prescription.

This repository includes:

- A KAN-compatible PyTorch module in `src/dmease/kan.py`.
- A deterministic greedy policy in `src/dmease/policy.py`.
- A `train.py` entry point for connecting EMR trajectories, KG artifacts, compatibility constraints, and clinician-reviewed reward labels.

The deterministic policy keeps the demo reproducible and provides a stable baseline. When curated training trajectories are available, the same state, action, constraint, and trace schemas can be used for KAN/PPO training.

## Optional UI

```bash
streamlit run app/streamlit_app.py
```

The UI exposes the PatientDB patient-ID workflow described in the paper.

## Validation

```bash
pytest
```

The tests verify that the pipeline can produce a syndrome, retrieve targets, select herbs, and respect a patient-specific allergy constraint.

## Citation

Please cite the paper if this repository is useful for your work:

```bibtex
@inproceedings{jia2025dmease,
  title     = {DMease: AI-based Assistant Tool for Traditional Chinese Medicine Doctors in Diagnosing and Treating Diabetes Mellitus},
  author    = {Jia, Junxiang and Hao, Xinyi and Hu, Yizhong and Meng, Haoyang and Zhao, Cong and Guan, Jianfeng and Sun, Weiwei},
  booktitle = {IEEE BIBM},
  year      = {2025}
}
```
