"""Describe cross-dataset failures of a frozen classifier; never fit on test."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import joblib
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"experiments/v3"


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line.strip()]


def norm(text):
    return re.sub(r"\s+", "", text).removesuffix("证")


def main():
    original = read(ROOT/"experiments/v2/supervised_syndrome.json")
    selection = original["selection"]
    checkpoint = ROOT/"outputs/tcm_sd_classifier.joblib"
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == selection["checkpoint_sha256"]
    raw = ROOT/"data/raw/public/tcm_sd"
    assert hashlib.sha256((raw/"test.json").read_bytes()).hexdigest() == original["source_manifest"]["files"]["test.json"]["sha256"]
    # Train/dev checksums are fixed from the immutable source ZIP used by v2.
    import zipfile
    archive = raw/"TCM_SD_train_dev.zip"
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == original["source_manifest"]["files"][archive.name]["sha256"]
    with zipfile.ZipFile(archive) as z:
        for name in ("train.json", "dev.json"):
            assert (raw/name).read_bytes() == z.read(name)
    model = joblib.load(checkpoint)
    vectorizer, classifier = model["vectorizer"], model["classifier"]
    train, dev, test = (lines(raw/f"{s}.json") for s in ("train", "dev", "test"))
    inputs = lambda rows: ["\n".join(r.get(k, "") for k in selection["protocol"]["input_fields"])[:4096] for r in rows]
    xt, xv, xx = (inputs(rows) for rows in (train, dev, test))
    labels = classifier.classes_.tolist()
    frequency = Counter(r["norm_syndrome"] for r in train)
    gold = [r["norm_syndrome"] for r in test]
    scores = classifier.decision_function(vectorizer.transform(xx))
    top = np.asarray(labels)[np.argsort(-scores, axis=1, kind="stable")[:,:3]]
    precision, recall, f1, support = precision_recall_fscore_support(gold, top[:,0], labels=labels, zero_division=0)
    by_label = [{"label": label, "train": frequency[label], "test": int(support[i]), "precision": float(precision[i]),
                 "recall": float(recall[i]), "f1": float(f1[i])} for i,label in enumerate(labels)]
    bands = []
    for name, lower, upper in (("<20", 0, 20), ("20-99", 20, 100), ("100-999", 100, 1000), (">=1000", 1000, float("inf"))):
        members = [r for r in by_label if lower <= r["train"] < upper]
        index = [i for i,g in enumerate(gold) if lower <= frequency[g] < upper]
        bands.append({"training_frequency": name, "labels": len(members), "test_records": len(index),
                      "top1_accuracy": float(np.mean([top[i,0] == gold[i] for i in index])) if index else None,
                      "top3_hit": float(np.mean([gold[i] in top[i] for i in index])) if index else None,
                      "macro_f1": float(np.mean([r["f1"] for r in members])) if members else None})
    external_path = ROOT/"data/raw/public/tcmeval_sdt/Train_TCM_Data_v1.json"
    assert hashlib.sha256(external_path.read_bytes()).hexdigest() == original["tcmeval_transfer"]["source_sha256"]
    external = read(external_path)
    xe = [r["Clinical Data"][:4096] for r in external]
    ye = [{norm(x) for x in re.split("[;；]",r["TCM Syndrome"]) if x.strip()} for r in external]
    es = classifier.decision_function(vectorizer.transform(xe))
    ep = [[norm(labels[i]) for i in row] for row in np.argsort(-es, axis=1, kind="stable")[:,:3]]
    known = {norm(label) for label in labels}
    eligible = [i for i,g in enumerate(ye) if g & known]
    transfer = {"records": len(ye), "gold_labels": sum(map(len,ye)), "gold_labels_in_training_vocabulary": sum(len(g & known) for g in ye),
                "records_with_any_known_gold": len(eligible), "top1_known_label_ceiling": len(eligible)/len(ye),
                "overall_top1": float(np.mean([p[0] in g for p,g in zip(ep,ye)])),
                "overall_top3": float(np.mean([bool(set(p)&g) for p,g in zip(ep,ye)])),
                "conditional_top1_known_gold": float(np.mean([ep[i][0] in ye[i] for i in eligible])) if eligible else None,
                "conditional_top3_known_gold": float(np.mean([bool(set(ep[i])&ye[i]) for i in eligible])) if eligible else None,
                "unseen_gold_label_counts": dict(Counter(x for g in ye for x in g-known).most_common())}
    analyzer = vectorizer.build_analyzer()
    def coverage(texts):
        lengths, rates, counts = [], [], []
        vocab = vectorizer.vocabulary_
        for text in texts:
            grams = analyzer(text)
            lengths.append(len(text)); counts.append(len(grams))
            rates.append(sum(g in vocab for g in grams)/len(grams) if grams else 0.)
        return {"records": len(texts), "median_chars": float(np.median(lengths)), "mean_ngram_in_vocabulary": float(np.mean(rates)),
                "median_ngram_in_vocabulary": float(np.median(rates)), "p10_ngram_in_vocabulary": float(np.quantile(rates,.1))}
    digest = lambda text: hashlib.sha256(re.sub(r"\s+", "", text).encode()).hexdigest()
    seen = {digest(x) for x in xt+xv}
    groups = {}
    for group, membership in (("seen_in_train_or_dev", True), ("unseen_in_train_and_dev", False)):
        indices = [i for i,x in enumerate(xx) if (digest(x) in seen) == membership]
        groups[group] = {"records": len(indices), "top1_accuracy": float(np.mean([top[i,0] == gold[i] for i in indices]))}
    result = {"protocol": read(OUT/"diagnostic_protocol.json"), "checkpoint_sha256": selection["checkpoint_sha256"],
              "source_lf_sha256": {"scripts/syndrome_diagnostics_v3.py": hashlib.sha256(Path(__file__).read_text(encoding="utf8").encode()).hexdigest()},
              "by_label": by_label, "frequency_bands": bands, "transfer": transfer, "text_duplicates": groups,
              "feature_coverage": {"TCM_SD_train": coverage(xt), "TCM_SD_test": coverage(xx), "TCMEval_external": coverage(xe)}}
    (OUT/"syndrome_diagnostics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
    print(json.dumps({k: result[k] for k in ("frequency_bands", "feature_coverage", "text_duplicates")}, ensure_ascii=False), flush=True)
    print(json.dumps({k:v for k,v in transfer.items() if k != "unseen_gold_label_counts"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
