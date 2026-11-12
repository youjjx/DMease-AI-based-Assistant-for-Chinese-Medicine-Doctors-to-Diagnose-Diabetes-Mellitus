"""Real public TCM-SD train/dev/test benchmark; no narratives or models in Git."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
import time
import zipfile
import requests

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from sklearn.naive_bayes import ComplementNB

from dmease.evidence_service import PublicEvidenceService

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/v2"
RAW = ROOT / "data/raw/public/tcm_sd"


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line.strip()]


def digest(text):
    return hashlib.sha256(re.sub(r"\s+", "", text).encode()).hexdigest()


def scores(gold, top, labels):
    if not len(gold):
        return {"records": 0}
    prediction = [x[0] if len(x) else "__abstain__" for x in top]
    return {"records": len(gold), "accuracy": float(accuracy_score(gold, prediction)),
            "macro_f1_148_train_labels": float(f1_score(gold, prediction, labels=labels, average="macro", zero_division=0)),
            "kappa": float(cohen_kappa_score(gold, prediction)),
            "top3_hit": float(np.mean([g in p for g,p in zip(gold,top)])),
            "abstentions": sum(not len(p) for p in top)}


def rank(classifier, matrix):
    values = classifier.predict_log_proba(matrix) if isinstance(classifier, ComplementNB) else classifier.decision_function(matrix)
    order = np.argsort(-values, axis=1, kind="stable")[:,:3]
    return classifier.classes_[order].tolist()


def main():
    protocol = json.loads((OUT/"syndrome_protocol.json").read_text(encoding="utf8"))
    repository = ROOT/"outputs/zybert-source"
    if not repository.exists():
        subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", protocol["repository"], str(repository)], check=True)
    RAW.mkdir(parents=True, exist_ok=True)
    manifest = {"repository": protocol["repository"], "commit": protocol["commit"], "files": {}}
    def fetch(name):
        path = RAW/name
        object_id = subprocess.check_output(["git", "-C", str(repository), "rev-parse", f"{protocol['commit']}:{name}"], text=True).strip()
        try:
            content = subprocess.check_output(["git", "-C", str(repository), "show", f"{protocol['commit']}:{name}"], timeout=60)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            response = requests.get(f"https://raw.githubusercontent.com/Borororo/ZY-BERT/{protocol['commit']}/{name}", timeout=60)
            response.raise_for_status(); content = response.content
        assert hashlib.sha1(f"blob {len(content)}\0".encode()+content).hexdigest() == object_id
        path.write_bytes(content)
        manifest["files"][name] = {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
        return path
    fetch("README.md")
    with zipfile.ZipFile(fetch("TCM_SD_train_dev.zip")) as archive:
        for name in ("train.json", "dev.json"):
            (RAW/name).write_bytes(archive.read(name))
    train, dev = read_lines(RAW/"train.json"), read_lines(RAW/"dev.json")
    def text(row):
        return "\n".join(row.get(k, "") for k in protocol["input_fields"])[:protocol["feature"]["max_text_chars"]]
    xt, xv = [text(r) for r in train], [text(r) for r in dev]
    yt, yv = [r[protocol["label_field"]] for r in train], [r[protocol["label_field"]] for r in dev]
    labels = sorted(set(yt)); assert len(labels)==148
    train_hash = {digest(x) for x in xt}
    validation_idx = [i for i,x in enumerate(xv) if digest(x) not in train_hash]
    assert validation_idx
    config = protocol["feature"]
    start = time.perf_counter()
    checkpoint = ROOT/"outputs/tcm_sd_classifier.joblib"
    selection_file = OUT/"syndrome_selection.json"
    if selection_file.exists() and checkpoint.exists():
        selection = json.loads(selection_file.read_text(encoding="utf8"))
        assert selection["protocol"] == protocol
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == selection["checkpoint_sha256"]
        stored = joblib.load(checkpoint)
        vectorizer, best = stored["vectorizer"], stored["classifier"]
        print("Resuming the already frozen, checksum-verified classifier; no retraining or reselection.", flush=True)
    else:
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=tuple(config["ngram_range"]),
                                   max_features=config["max_features"], min_df=config["min_df"], sublinear_tf=True, dtype=np.float32)
        tx = vectorizer.fit_transform(xt); vx = vectorizer.transform(xv)
        print(f"Train {tx.shape}, dev {vx.shape}; validation unseen texts {len(validation_idx)}", flush=True)
        candidate_reports, best, best_score = [], None, -1.
        for candidate in protocol["candidates"]:
            if candidate["method"] == "complement_nb":
                classifier = ComplementNB(alpha=candidate["alpha"])
            else:
                classifier = SGDClassifier(loss="log_loss", alpha=candidate["alpha"], random_state=42, max_iter=30, tol=.001, n_jobs=1)
            classifier.fit(tx, yt)
            prediction = rank(classifier, vx)
            evaluation = scores([yv[i] for i in validation_idx], [prediction[i] for i in validation_idx], labels)
            candidate_reports.append({"configuration": candidate, "unseen_text_validation": evaluation})
            print(f"Validation {candidate}: macroF1={evaluation['macro_f1_148_train_labels']:.4f}", flush=True)
            if evaluation["macro_f1_148_train_labels"] > best_score:
                best, best_score, selected = classifier, evaluation["macro_f1_148_train_labels"], candidate
        joblib.dump({"vectorizer": vectorizer, "classifier": best, "protocol": protocol}, checkpoint, compress=3)
        checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        selection = {"protocol": protocol, "candidates": candidate_reports, "selected": selected,
                     "checkpoint_sha256": checkpoint_hash, "frozen_before_test_gold_download": True}
        selection_file.write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf8")
    # The official test labels enter only after the selection above is saved.
    test = read_lines(fetch("test.json"))
    xx = [text(r) for r in test]; yy = [r[protocol["label_field"]] for r in test]
    predictions = rank(best, vectorizer.transform(xx))
    seen = train_hash | {digest(x) for x in xv}
    unseen_idx = [i for i,x in enumerate(xx) if digest(x) not in seen]
    no_label_idx = [i for i,x in enumerate(xx) if yy[i] not in x]
    graph_predictions = []
    with tempfile.TemporaryDirectory(prefix="dmease-syndrome-") as tmp:
        service = PublicEvidenceService(ROOT/"data/processed/evidence_bundle.json", Path(tmp)/"records.db", None)
        # Lexical terminal-证 normalization only; no learned mapping from test labels.
        canon = lambda name: name[:-1] if name.endswith("证") else name
        label_map = {canon(name): name for name in labels}
        for x in xx:
            graph_predictions.append([label_map.get(canon(s.name), s.name) for s in service.diagnose(service.parser.parse(x))])
        evidence = service.evidence.fingerprint
    results = {}
    for name, pred in (("supervised_text_baseline", predictions), ("frozen_public_graph", graph_predictions)):
        results[name] = {"official_test": scores(yy,pred,labels),
                         "unseen_text_test": scores([yy[i] for i in unseen_idx], [pred[i] for i in unseen_idx],labels),
                         "without_literal_gold_mention": scores([yy[i] for i in no_label_idx],[pred[i] for i in no_label_idx],labels)}
    # Cross-dataset transfer is separate from the official single-label benchmark.
    external_path = ROOT/"data/raw/public/tcmeval_sdt/Train_TCM_Data_v1.json"
    transfer = None
    if external_path.exists():
        external = json.loads(external_path.read_text(encoding="utf8"))
        pred = rank(best, vectorizer.transform([r["Clinical Data"][:4096] for r in external]))
        gold = [{canon(s.strip()) for s in re.split("[;；]",r["TCM Syndrome"]) if s.strip()} for r in external]
        pred = [[canon(s) for s in row] for row in pred]
        transfer = {"records": len(gold), "top1_any_gold_hit": float(np.mean([p[0] in g for p,g in zip(pred,gold)])),
                    "top3_any_gold_hit": float(np.mean([bool(set(p)&g) for p,g in zip(pred,gold)])),
                    "source_sha256": hashlib.sha256(external_path.read_bytes()).hexdigest(),
                    "interpretation": "Exploratory transfer; label ontology differs and baseline already inspected"}
    train_users = {r["user_id"] for r in train if r["user_id"]}
    source_files = ["scripts/tcm_sd_benchmark.py", "src/dmease/evidence_service.py", "src/dmease/symptom_parser.py"]
    result = {"selection": selection, "source_manifest": manifest, "evidence_fingerprint": evidence,
              "source_lf_sha256": {f:hashlib.sha256((ROOT/f).read_text(encoding="utf8").encode()).hexdigest() for f in source_files},
              "train":len(train), "dev":len(dev), "test":len(test), "vocabulary_size":len(vectorizer.vocabulary_),
              "validation_excluded_seen_texts":len(dev)-len(validation_idx), "test_seen_texts_train_or_dev":len(test)-len(unseen_idx),
              "test_literal_gold_mentions":len(test)-len(no_label_idx),
              "test_rows_with_user_id_in_train":sum(r["user_id"] in train_users for r in test),
              "duplicate_test_text_excess":len(xx)-len({digest(x) for x in xx}),
              "elapsed_seconds":time.perf_counter()-start,"runtime_scope":"current invocation, excluding resumed training", "evaluation":results,"tcmeval_transfer":transfer}
    (OUT/"supervised_syndrome.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf8")
    print(json.dumps({"evaluation":results,"transfer":transfer},ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()
