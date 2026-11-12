"""Author-partition extraction evaluation, without publishing medical narratives."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import subprocess
import tempfile
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from lmformatenforcer import JsonSchemaParser, CharacterLevelParserConfig
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn

from dmease.evidence_service import PublicEvidenceService
from dmease.local_extraction import build_messages, validate_output
from dmease.structured_extraction import chunks, prompt, output_schema, validate, normalize_items, normalize_term, set_counts, summarize, character_positions

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"experiments/v4"
RAW = ROOT/"data/raw/public/tcmeval_v4"
PRIVATE = ROOT/"outputs/research_v4/extraction"


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf8",newline="\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_records(split):
    records = read(RAW/f"{split}_TCM_Data_v1.json")
    answers = {}
    for line in (RAW/f"{split}_data_result.txt").read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            row = line.split("@",4)
            if len(row) != 5 or row[0].strip() in answers:
                raise ValueError("Invalid or duplicate expert answer")
            answers[row[0].strip()] = row
    assert len(records) == len(answers) == 50
    assert {r["Medical Record ID"].strip() for r in records} == answers.keys()
    return [{**r, "expert_information": answers[r["Medical Record ID"].strip()][1],
             "expert_syndrome_choices": answers[r["Medical Record ID"].strip()][3]} for r in records]


def prepare(protocol):
    source_repo = ROOT/"outputs/tcmeval_source"
    if not source_repo.exists():
        subprocess.run(["git","clone","--filter=blob:none","--no-checkout",protocol["source_repository"],str(source_repo)],check=True)
    RAW.mkdir(parents=True,exist_ok=True)
    files = [f"evaluation/TCMEval-SDT/{folder}/{name}" for folder,names in {
        "data": ["Validation_TCM_Data_v1.json","Test_TCM_Data_v1.json"],
        "Results": ["Validation_data_result.txt","Test_data_result.txt"]}.items() for name in names]
    manifest = {}
    for name in files:
        object_id = subprocess.check_output(["git","-C",str(source_repo),"rev-parse",f"{protocol['source_commit']}:{name}"],text=True).strip()
        destination = RAW/Path(name).name
        if not destination.exists():
            destination.write_bytes(subprocess.check_output(["git","-c","http.sslBackend=openssl","-C",str(source_repo),"show",f"{protocol['source_commit']}:{name}"]))
        content = destination.read_bytes()
        assert hashlib.sha1(f"blob {len(content)}\0".encode()+content).hexdigest() == object_id
        manifest[name] = {"git_blob": object_id,"sha256":sha(destination),"bytes":len(content)}
    train = read(ROOT/"data/raw/public/tcmeval_sdt/Train_TCM_Data_v1.json")
    partitions = {"train":train, "validation":load_records("Validation"), "test":load_records("Test")}
    ids = {k:{normalize_term(r["Clinical Data"]) for r in rows} for k,rows in partitions.items()}
    overlaps = {f"{a}_{b}":len(ids[a]&ids[b]) for a,b in (("train","validation"),("train","test"),("validation","test"))}
    assert not any(overlaps.values())
    save(OUT/"extraction_sources.json",{"repository":protocol["source_repository"],"commit":protocol["source_commit"],
        "files":manifest,"normalized_text_overlap":overlaps,"counts":{k:len(v) for k,v in partitions.items()},
        "correction":protocol["correction"],"license":"TCMEval-SDT author README: CC BY 4.0"})


def score_record(record, terms, positive, service, metadata):
    source = normalize_term(record["Clinical Data"])
    gold = {normalize_term(x) for x in re.split("[;；]",record["expert_information"]) if x.strip()}
    predicted = {normalize_term(x) for x in terms if x.strip()}
    locatable = {g for g in gold if g in source}
    gold_concepts = {s.canonical for s in service.parser.parse(record["expert_information"]) if s.present}
    return {"row_hash":hashlib.sha256(record["Clinical Data"].encode()).hexdigest(),
            "surface":set_counts(gold,predicted), "characters":set_counts(character_positions(source,locatable),character_positions(source,predicted)),
            "projected_secondary":set_counts(gold_concepts,positive), "locatable_gold":len(locatable), "total_gold":len(gold),
            "author_task1_recall":len(gold&predicted)/len(gold) if gold else 0., **metadata}


def aggregate(rows):
    rejected=Counter()
    for row in rows: rejected.update(row.get("rejected",{}))
    return {"records":len(rows), **{key:summarize([r[key] for r in rows]) for key in ("surface","characters","projected_secondary")},
            "author_task1_recall":float(np.mean([r["author_task1_recall"] for r in rows])),
            "gold_verbatim_fraction":sum(r["locatable_gold"] for r in rows)/sum(r["total_gold"] for r in rows),
            "invalid_json_chunks":sum(r.get("invalid_json_chunks",0) for r in rows),
            "invalid_schema_chunks":sum(r.get("invalid_schema_chunks",0) for r in rows),
            "cases_with_invalid_chunks":sum(bool(r.get("invalid_json_chunks",0) or r.get("invalid_schema_chunks",0)) for r in rows),
            "chunks":sum(r.get("chunks",0) for r in rows), "output_limits":sum(r.get("output_limits",0) for r in rows),
            "rejected_items":dict(rejected),"proposed_items":sum(r.get("proposed_items",0) for r in rows),
            "median_seconds":float(np.median([r.get("seconds",0) for r in rows])),"total_seconds":sum(r.get("seconds",0) for r in rows)}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--prepare",action="store_true");parser.add_argument("--phase",choices=["validation","test"],default="validation")
    args=parser.parse_args();protocol=read(OUT/"extraction_protocol.json")
    if args.prepare:prepare(protocol);return
    sources=["scripts/extraction_v4.py","src/dmease/structured_extraction.py","src/dmease/local_extraction.py","src/dmease/symptom_parser.py"]
    stamp={"protocol":protocol,"source_lf_sha256":{s:hashlib.sha256((ROOT/s).read_text(encoding="utf8").encode()).hexdigest() for s in sources},
           "sources_sha256":sha(OUT/"extraction_sources.json")}
    if args.phase == "test":
        selection=read(OUT/"extraction_selection.json");assert selection["provenance"]==stamp
    records=load_records(args.phase.capitalize())
    model_path=ROOT/"outputs/research_v3/local_model";manifest=read(model_path/"verified_model.json")
    assert manifest["revision"]==protocol["revision"]
    with (model_path/"model.safetensors").open("rb") as handle:assert hashlib.file_digest(handle,"sha256").hexdigest()==manifest["weights_sha256"]
    tokenizer=AutoTokenizer.from_pretrained(str(model_path),local_files_only=True)
    model=None;config=protocol["decoding"]
    with sqlite3.connect(f"file:{ROOT/'data/processed/public_evidence.sqlite'}?mode=ro",uri=True) as db:
        vocab=sorted({normalize_term(p["TCM_symptom_name"]) for (value,) in db.execute("SELECT payload FROM entities WHERE source='SymMap' AND kind='SMTS'")
                      if not (p:=json.loads(value)).get("Suppress") and p.get("TCM_symptom_name")})
    with tempfile.TemporaryDirectory(prefix="dmease-v4-extract-") as temp:
        service=PublicEvidenceService(ROOT/"data/processed/evidence_bundle.json",Path(temp)/"records.db",None)
        methods={m:[] for m in ["rules24","symmap_literal","copy_clauses",*protocol["variants"]]}
        if args.phase=="test":methods["v3_strict"]=[]
        for i,record in enumerate(records):
            source=record["Clinical Data"]; rowhash=hashlib.sha256(source.encode()).hexdigest()
            parsed=service.parser.parse(source)
            direct=[s.original for s in parsed]
            positive={s.canonical for s in parsed if s.present}
            methods["rules24"].append(score_record(record,direct,positive,service,{}))
            for method,terms in (("symmap_literal",[w for w in vocab if w in normalize_term(source)]),
                                 ("copy_clauses",[s.strip() for s in re.split(r"[，,。；;！？!?]",source) if s.strip()])):
                normalized={s.canonical for t in terms for s in service.parser.parse(t) if s.present}
                methods[method].append(score_record(record,terms,normalized,service,{}))
            for variant in [*protocol["variants"],*( ["v3_strict"] if args.phase=="test" else [])]:
                segments=[(0,source)] if variant=="v3_strict" else chunks(source,config["chunk_chars"])
                accepted=[];quality=Counter();reasons=Counter();duration=0
                for j,(offset,text) in enumerate(segments):
                    cache=PRIVATE/args.phase/variant/f"{i:03d}_{j:02d}.json"
                    if cache.exists():
                        generated=read(cache);assert generated["provenance"]==stamp and generated["row_hash"]==rowhash
                    else:
                        if model is None:
                            torch.manual_seed(config["seed"])
                            model=AutoModelForCausalLM.from_pretrained(str(model_path),local_files_only=True,dtype=torch.float16,attn_implementation="sdpa").to("cuda").eval()
                        messages=build_messages(text,service.graph.symptoms) if variant=="v3_strict" else prompt(text)
                        inputs=tokenizer.apply_chat_template(messages,add_generation_prompt=True,return_tensors="pt",return_dict=True)
                        assert inputs["input_ids"].shape[1]<4096,"Unexpected context length; do not silently truncate"
                        inputs={k:v.to("cuda") for k,v in inputs.items()}; kwargs={}
                        if variant in ("schema","source_schema"):
                            schema=output_schema(text if variant=="source_schema" else None,config["max_term_chars"],config["max_items"])
                            alphabet="".join(sorted(set(CharacterLevelParserConfig().alphabet+text)))
                            grammar=JsonSchemaParser(schema,config=CharacterLevelParserConfig(alphabet=alphabet,max_consecutive_whitespaces=1,force_json_field_order=True))
                            kwargs["prefix_allowed_tokens_fn"]=build_transformers_prefix_allowed_tokens_fn(tokenizer,grammar)
                        limit=384 if variant=="v3_strict" else config["max_new_tokens"]
                        torch.cuda.synchronize();start=time.perf_counter()
                        with torch.inference_mode():
                            result=model.generate(**inputs,max_new_tokens=limit,do_sample=False,pad_token_id=tokenizer.eos_token_id,**kwargs)
                        torch.cuda.synchronize();elapsed=time.perf_counter()-start
                        output=result[0,inputs["input_ids"].shape[1]:]
                        generated={"provenance":stamp,"row_hash":rowhash,"offset":offset,"raw":tokenizer.decode(output,skip_special_tokens=True),
                                   "seconds":elapsed,"output_limit":len(output)>=limit,"output_tokens":len(output),"input_tokens":int(inputs["input_ids"].shape[1])}
                        save(cache,generated)
                    duration+=generated["seconds"];quality["output_limits"]+=generated["output_limit"]
                    if variant=="v3_strict":
                        check=validate_output(generated["raw"],text,{s["name"] for s in service.graph.symptoms})
                        accepted.extend({"text":x["term"],"present":x["present"],"canonical":x["canonical"]} for x in check["items"])
                        quality["proposed_items"]+=check["proposed_items"]
                    else:
                        check=validate(generated["raw"],text,config["max_term_chars"],config["max_items"])
                        accepted.extend(check["items"]);quality["proposed_items"]+=check["proposed"]
                    quality["invalid_json_chunks"]+=check["invalid_json"];quality["invalid_schema_chunks"]+=check.get("invalid_schema",False)
                    reasons.update(check["rejected"])
                mapped=normalize_items(accepted,service.parser)
                positive={name for name,items in mapped.items() if any(x["present"] for x in items)} if variant!="v3_strict" else {x["canonical"] for x in accepted if x["present"] and x["canonical"]}
                methods[variant].append(score_record(record,[x["text"] for x in accepted],positive,service,
                    {**quality,"rejected":dict(reasons),"seconds":duration,"chunks":len(segments)}))
            if (i+1)%5==0:print(f"{args.phase}: {i+1}/50 completed for all variants",flush=True)
        summary={method:aggregate(rows) for method,rows in methods.items()}
        report={"provenance":stamp,"phase":args.phase,"model_manifest":manifest,"evidence_fingerprint":service.evidence.fingerprint,
                "symmap_vocabulary_size":len(vocab),"summary":summary,"methods":methods,"completed_at":datetime.now(timezone.utc).isoformat()}
        save(OUT/f"extraction_{args.phase}.json",report)
        if args.phase=="validation":
            selected=max(protocol["variants"],key=lambda m:summary[m]["surface"]["f1"])
            save(OUT/"extraction_selection.json",{"provenance":stamp,"selected":selected,"validation":{m:summary[m] for m in protocol["variants"]},
                "frozen_before_test_at":datetime.now(timezone.utc).isoformat()})
        print(json.dumps({m:{"surface_f1":r["surface"]["f1"],"invalid_json":r["invalid_json_chunks"]} for m,r in summary.items()},ensure_ascii=False),flush=True)


if __name__=="__main__":main()
