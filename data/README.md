# Data Layout

The original paper describes three data sources:

- Classical TCM literature: 30 foundational books after OCR correction.
- Guidelines: 5 national TCM guidelines for diabetes, hypertension, and chronic kidney disease.
- De-identified EMR: 108,746 outpatient TCM records.

Keep licensed or private data out of Git and use the following layout for local research datasets:

```text
data/
  raw/
    classical_literature/
    guidelines/
    emr/
    symmap/
    string/
    tcmsp/
  processed/
    patients.jsonl
    triples.jsonl
    herb_target_affinity.csv
  examples/
    patients.jsonl
    triples.jsonl
    herb_target_affinity.csv
```

The `examples/` directory contains a compact paper-aligned sample dataset for reproducible demos and tests.
It is not clinical evidence and must not be used for real medical decisions.
