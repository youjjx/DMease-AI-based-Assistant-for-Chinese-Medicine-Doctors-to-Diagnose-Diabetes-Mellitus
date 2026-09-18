# DMease

DMease is a complete research implementation for the paper **"DMease: AI-based Assistant Tool for Traditional Chinese Medicine Doctors in Diagnosing and Treating Diabetes Mellitus"**.

The repository has been rebuilt around the method described in the paper:

1. Multi-source TCM text is normalized and converted into structured triples.
2. Triples are stored as a symptom-target-herb knowledge graph.
3. A PatientDB provides patient-ID based records.
4. The system predicts a TCM syndrome from symptoms.
5. A sequential single-herb policy ranks and selects herbs under compatibility constraints.
6. The output includes target coverage, marginal symptom relief, and provenance traces.

The repository includes directly runnable non-clinical sample data. The software is for research and engineering validation; clinical deployment requires authorized data, governance review, and clinician oversight.

## Repository Structure

```text
configs/
  dmease.yaml                  # Main configuration
data/
  README.md                    # Expected data layout
  examples/                    # Non-clinical sample data for validation and tests
  knowledge_graph.json         # Structured symptom-syndrome-herb-mechanism graph
prompts/
  triplet_extraction_zh.md     # LLM triplet extraction prompt
src/dmease/
  schemas.py                   # Patient, triple, diagnosis, prescription schemas
  patient_db.py                # Lightweight PatientDB
  semantic_parser.py           # Surface-to-concept mapping and JSON triplet parser
  knowledge_graph.py           # Symptom-target-herb KG and PMI-style retrieval
  diagnosis.py                 # Syndrome diagnosis module
  constraints.py               # Classical and patient-specific hard constraints
  kan.py                       # KAN actor, critic, and edge-function layers
  ppo.py                       # Clipped PPO environment and trainer
  policy.py                    # Current greedy single-herb policy
  recommender.py               # Hybrid graph/KAN ranking and constraints
  symptom_parser.py            # Chinese symptom normalization and negation
  service.py                   # Interactive end-to-end service facade
  types.py                     # Interactive workflow data structures
  pipeline.py                  # End-to-end inference workflow
  cli.py                       # Command line interface
scripts/
  build_kg.py                  # Convert LLM JSON output to triples.jsonl
  infer_example.py             # Minimal inference example
app/
  streamlit_app.py             # Patient-ID workflow
app.py                         # Full Chinese Streamlit workflow
tests/
  test_pipeline.py             # Smoke tests for the example pipeline
```

## Installation

```bash
conda env create -f environment.yml
conda activate dmease
pip install -e .
```

The included non-clinical sample assets under `data/examples/` and `data/knowledge_graph.json` make both inference workflows directly reproducible.

## Quick Start

List example patients:

```bash
dmease --config configs/dmease.yaml patients
```

Run the paper-style sample case:

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

The included `data/examples/` files are compact non-clinical sample assets aligned with the paper's case structure. They make the repository runnable out of the box for tests and engineering review.

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

This repository implements:

- KAN actor and critic networks with trainable edge-wise basis functions in `src/dmease/kan.py`.
- A clipped PPO environment, rollout collector, optimizer, and checkpoint writer in `src/dmease/ppo.py`.
- A deterministic sequential policy for reproducible PatientDB inference in `src/dmease/policy.py`.
- A graph/KAN hybrid ranking service with hard safety constraints and evidence paths in `src/dmease/recommender.py`.
- A `train.py` workflow that trains the KAN/PPO policy on graph-derived sequential herb-selection tasks.

The deterministic and learned policies share the same graph, state, constraint, and trace concepts. Authorized de-identified trajectories can replace the included non-clinical training tasks without changing the application boundary.

Train and save a KAN/PPO checkpoint:

```bash
python train.py --iterations 50 --episodes 8
```

## Streamlit UI

```bash
streamlit run app.py
```

The Chinese UI supports free-text symptom parsing, syndrome inference, constrained herb ranking, evidence paths, knowledge graph browsing, and persistent SQLite patient records. `app/streamlit_app.py` additionally retains the paper-style PatientDB patient-ID workflow.

## Validation

```bash
pytest
```

The tests verify both workflows: symptom negation, syndrome reasoning, evidence paths, safety constraints, SQLite persistence, target retrieval, sequential selection, and patient-specific allergy handling.

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
