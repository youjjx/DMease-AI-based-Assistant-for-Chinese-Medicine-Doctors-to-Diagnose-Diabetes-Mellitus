"""Validation-selected class-frequency weighting; retain all candidates."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_recall_fscore_support

from tcm_sd_benchmark import read_lines, digest, rank, scores

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"experiments/v3"
PRIVATE = ROOT/"outputs/research_v3/imbalance"


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")


def main():
    protocol = read(OUT/"imbalance_protocol.json")
    old = read(ROOT/"experiments/v2/supervised_syndrome.json")
    original = ROOT/"outputs/tcm_sd_classifier.joblib"
    assert sha(original) == old["selection"]["checkpoint_sha256"]
    base = joblib.load(original)
    vectorizer, baseline = base["vectorizer"], base["classifier"]
    raw = ROOT/"data/raw/public/tcm_sd"
    import zipfile
    archive = raw/"TCM_SD_train_dev.zip"
    assert sha(archive) == old["source_manifest"]["files"][archive.name]["sha256"]
    with zipfile.ZipFile(archive) as z:
        for filename in ("train.json", "dev.json"):
            assert (raw/filename).read_bytes() == z.read(filename)
    train, dev = (read_lines(raw/f"{s}.json") for s in ("train", "dev"))
    get_text = lambda rows: ["\n".join(r.get(k, "") for k in base["protocol"]["input_fields"])[:4096] for r in rows]
    xt, xv = get_text(train), get_text(dev)
    yt, yv = [r["norm_syndrome"] for r in train], [r["norm_syndrome"] for r in dev]
    counts = Counter(yt); labels = sorted(counts)
    seen_train = {digest(x) for x in xt}
    vi = [i for i,x in enumerate(xv) if digest(x) not in seen_train]
    stamp = {"protocol": protocol, "baseline_sha256": sha(original), "source_lf_sha256": {
        name: hashlib.sha256((ROOT/name).read_text(encoding="utf8").encode()).hexdigest()
        for name in ("scripts/imbalance_benchmark_v3.py", "scripts/tcm_sd_benchmark.py")}}
    PRIVATE.mkdir(parents=True, exist_ok=True)
    selection_path = OUT/"imbalance_selection.json"
    chosen_path = PRIVATE/"selected.joblib"
    if selection_path.exists() and chosen_path.exists():
        selection = read(selection_path)
        assert selection["provenance"] == stamp and sha(chosen_path) == selection["selected_checkpoint_sha256"]
        selected = joblib.load(chosen_path)
    else:
        tx, vx = vectorizer.transform(xt), vectorizer.transform(xv)
        base_score = scores([yv[i] for i in vi], rank(baseline, vx[vi]), labels)
        candidates = [{"configuration": {"method": "frozen_unweighted_v2"}, "validation": base_score}]
        selected, selected_index, best = baseline, 0, base_score["macro_f1_148_train_labels"]
        for i, configuration in enumerate(protocol["candidates"], 1):
            w = np.array([(len(yt)/(len(labels)*counts[label]))**configuration["weight_power"] for label in yt], dtype=np.float32)
            w /= w.mean()
            model = SGDClassifier(loss="log_loss", alpha=configuration["alpha"], random_state=42, max_iter=30, tol=.001, n_jobs=1)
            model.fit(tx, yt, sample_weight=w)
            evaluation = scores([yv[j] for j in vi], rank(model,vx[vi]), labels)
            checkpoint = PRIVATE/f"candidate_{i}.joblib"; joblib.dump(model,checkpoint,compress=3)
            candidates.append({"configuration": configuration, "validation": evaluation, "checkpoint_sha256": sha(checkpoint),
                               "weight_min": float(w.min()), "weight_max": float(w.max()), "weight_mean": float(w.mean())})
            print(f"Candidate {i}: {configuration}, validation macro-F1 {evaluation['macro_f1_148_train_labels']:.4f}", flush=True)
            if evaluation["macro_f1_148_train_labels"] > best:
                selected, selected_index, best = model, i, evaluation["macro_f1_148_train_labels"]
        joblib.dump(selected, chosen_path, compress=3)
        selection = {"provenance": stamp, "selected_index": selected_index, "selected_checkpoint_sha256": sha(chosen_path),
                     "validation_records": len(vi), "candidates": candidates, "frozen_before_new_test_at": datetime.now(timezone.utc).isoformat()}
        dump(selection_path, selection)
    assert sha(raw/"test.json") == old["source_manifest"]["files"]["test.json"]["sha256"]
    test = read_lines(raw/"test.json"); xx = get_text(test); gold = [r["norm_syndrome"] for r in test]
    seen = seen_train | {digest(x) for x in xv}
    unseen = [i for i,x in enumerate(xx) if digest(x) not in seen]
    matrix = vectorizer.transform(xx)
    ep = ROOT/"data/raw/public/tcmeval_sdt/Train_TCM_Data_v1.json"
    assert sha(ep) == old["tcmeval_transfer"]["source_sha256"]
    external = read(ep)
    ex = vectorizer.transform([r["Clinical Data"][:4096] for r in external])
    norm = lambda s: re.sub(r"\s+", "", s).removesuffix("证")
    ey = [{norm(s) for s in re.split("[;；]",r["TCM Syndrome"]) if s.strip()} for r in external]
    results = {}
    for method, model in (("frozen_unweighted_v2",baseline), ("validation_selected",selected)):
        pred = rank(model,matrix)
        precision, recall, f1, support = precision_recall_fscore_support(gold,[p[0] for p in pred],labels=labels,zero_division=0)
        by_label = [{"label": name, "train": counts[name], "test": int(support[i]), "precision": float(precision[i]),
                     "recall": float(recall[i]), "f1": float(f1[i])} for i,name in enumerate(labels)]
        bands = []
        for name,lower,upper in (("<20",0,20),("20-99",20,100),("100-999",100,1000),(">=1000",1000,float("inf"))):
            members = [r for r in by_label if lower <= r["train"] < upper]
            index = [i for i,g in enumerate(gold) if lower <= counts[g] < upper]
            bands.append({"training_frequency": name, "test_records": len(index), "labels": len(members),
                          "accuracy": float(np.mean([pred[i][0] == gold[i] for i in index])) if index else None,
                          "macro_f1": float(np.mean([r["f1"] for r in members])) if members else None})
        external_prediction = [[norm(s) for s in row] for row in rank(model,ex)]
        results[method] = {"official_test": scores(gold,pred,labels),
                           "unseen_text_test": scores([gold[i] for i in unseen],[pred[i] for i in unseen],labels),
                           "frequency_bands": bands, "by_label": by_label,
                           "external_transfer": {"records": len(ey), "top1": float(np.mean([p[0] in g for p,g in zip(external_prediction,ey)])),
                                                 "top3": float(np.mean([bool(set(p)&g) for p,g in zip(external_prediction,ey)]))}}
    result = {"selection": selection, "completed_at": datetime.now(timezone.utc).isoformat(), "results": results}
    dump(OUT/"imbalance.json", result)
    print(json.dumps({m:r["official_test"] for m,r in results.items()},ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()
