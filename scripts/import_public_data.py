"""Download official catalogues and human STRING v12.0, retaining provenance.

Run from repository root: python scripts/import_public_data.py
Raw data and the SQLite index stay in ignored data/raw and data/processed.
Only the provenance/count manifest is intended for Git.
"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import sqlite3
import time
from urllib.parse import unquote, urljoin, urlencode, urlparse, parse_qs

import openpyxl
import requests

from dmease.public_data import SCHEMA, embedded_tables


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == "a" and values.get("href"):
            self.links.append(values["href"])


class Importer:
    def __init__(self, root: Path):
        self.root = root
        self.raw = root / "data/raw/public"
        self.raw.mkdir(parents=True, exist_ok=True)
        self.files = []
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "DMease-Research-Importer/1.0"

    def _download(self, url: str, path: Path):
        with self.session.get(url, timeout=(15, 90), stream=True) as response:
            response.raise_for_status()
            size = int(response.headers.get("Content-Length", 0))
            etag = response.headers.get("ETag")
            if size < 8 * 1024 * 1024:
                with path.open("wb") as output:
                    for chunk in response.iter_content(256 * 1024):
                        output.write(chunk)
                return
        # Bounded range downloads avoid long unresponsive transfers of large files.
        block = 4 * 1024 * 1024
        def download_range(start):
            end = min(start + block, size) - 1
            headers = {"Range": f"bytes={start}-{end}", "User-Agent": "DMease-Research-Importer/1.0"}
            if etag:
                headers["If-Range"] = etag
            with requests.get(url, headers=headers, timeout=(15, 90)) as part:
                part.raise_for_status()
                if part.status_code != 206 or part.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                    raise ValueError("Server did not preserve the requested byte range/version")
                if len(part.content) != end - start + 1:
                    raise ValueError("Incomplete byte range")
                return start, part.content
        with path.open("wb") as output, ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(download_range, start) for start in range(0, size, block)]
            for future in as_completed(futures):
                start, content = future.result()
                output.seek(start)
                output.write(content)
                print(f"Received {start}-{start + len(content) - 1} / {size}", flush=True)

    def fetch(self, url: str, relative: str) -> Path:
        path = self.raw / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        sidecar = path.with_suffix(path.suffix + ".provenance.json")
        cached = False
        if path.exists() and sidecar.exists():
            prior = json.loads(sidecar.read_text(encoding="utf8"))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            cached = prior.get("url") == url and prior.get("sha256") == digest
        if not cached:
            for attempt in range(3):
                try:
                    temporary = path.with_suffix(path.suffix + ".part")
                    self._download(url, temporary)
                    if temporary.stat().st_size == 0:
                        raise ValueError(f"Empty download: {url}")
                    temporary.replace(path)
                    break
                except (requests.RequestException, OSError, ValueError):
                    if attempt == 2:
                        raise
                    time.sleep(2 * (attempt + 1))
            prior = {
                "url": url, "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
            sidecar.write_text(json.dumps(prior, indent=2), encoding="utf8")
        self.files.append({"file": str(path.relative_to(self.root)).replace("\\", "/"), **prior})
        print(f"Verified {relative}: {prior['bytes']} bytes", flush=True)
        return path

    def symmap(self, connection):
        base = "http://www.symmap.org"
        page = self.fetch(base + "/download/", "symmap/download.html")
        parser = Links()
        parser.feed(page.read_text(encoding="utf8"))
        counts = {}
        for href in dict.fromkeys(parser.links):
            if not href.startswith("/static/download/") or not href.endswith(".xlsx"):
                continue
            filename = unquote(href.rsplit("/", 1)[-1])
            path = self.fetch(urljoin(base, href), "symmap/" + filename)
            match = re.search(r"v([\d.]+),\s+(SM[A-Z]+)\s+(key )?file", filename)
            if not match:
                raise ValueError(f"Unexpected SymMap filename: {filename}")
            version, kind, key = match.groups()
            workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
            count = 0
            for sheet in workbook:
                rows = iter(sheet.iter_rows(values_only=True))
                header = [str(x or f"column_{i}") for i, x in enumerate(next(rows))]
                batch = []
                for number, row in enumerate(rows, 1):
                    if not any(x is not None for x in row):
                        continue
                    payload = dict(zip(header, row))
                    entity_id = str(row[0])
                    if key:
                        # Identifier crosswalks have many rows per entity.
                        entity_id += f":{sheet.title}:{number}"
                    name = " | ".join(str(x) for x in row[1:4] if x is not None)
                    batch.append(("SymMap", version, kind + ("_keys" if key else ""),
                                  entity_id, name, json.dumps(payload, ensure_ascii=False, default=str)))
                    count += 1
                connection.executemany("INSERT INTO entities VALUES(?,?,?,?,?,?)", batch)
            workbook.close()
            counts[filename] = count
        if not counts:
            raise ValueError("No official SymMap downloads found")
        return {"status": "official_entity_catalogues_imported", "tables": counts,
                "scope": "All XLSX links exposed on the captured download page; entity and identifier tables, not a full relationship dump",
                "license": "No explicit redistribution license identified on download page; raw files are not republished",
                "transport": "Official HTTP endpoint; HTTPS failed during verification"}

    def tcmsp(self, connection):
        counts = {}
        herb_rows = []
        herb_page = ""
        for kind in ("herbs", "ingredients", "targets", "diseases"):
            url = "https://old.tcmsp-e.com/browse.php?qc=" + kind
            path = self.fetch(url, f"tcmsp/{kind}.html")
            tables = embedded_tables(path.read_text(encoding="utf8"))
            if len(tables) != 1 or not tables[0]:
                raise ValueError(f"Expected one nonempty TCMSP {kind} table")
            rows = tables[0]
            if kind == "herbs":
                herb_rows, herb_page = rows, path.read_text(encoding="utf8")
            for i, row in enumerate(rows):
                # Keep the complete source row; names/IDs are not assumed to be UniProt IDs.
                entity_id = str(row.get("molecule_ID") or row.get("target_ID") or
                                row.get("disease_ID") or row.get("herb_cn_name") or i)
                entity_id = "TCMSP:" + {"herbs": "herb", "ingredients": "molecule", "targets": "target", "diseases": "disease"}[kind] + ":" + entity_id
                name = " | ".join(str(value) for key, value in row.items()
                                  if value is not None and ("name" in key or key in ("MOL_ID", "TAR_ID", "DIS_ID", "herb_pinyin")))
                connection.execute("INSERT INTO entities VALUES(?,?,?,?,?,?)", (
                    "TCMSP", "2.3", kind, entity_id, name, json.dumps(row, ensure_ascii=False)))
            counts[kind] = len(rows)
            time.sleep(0.5)
        # Follow the query link published by TCMSP; use exact Chinese herb matches.
        template = re.search(r'template:\s*"<a href=\x27([^\x27]+)\x27', herb_page)
        if not template:
            raise ValueError("TCMSP herb detail link missing")
        parsed = urlparse(template.group(1))
        params = parse_qs(parsed.query)
        graph = json.loads((self.root / "data/knowledge_graph.json").read_text(encoding="utf8"))
        requested = [item["name"] for item in graph["herbs"]]
        by_name = {row["herb_cn_name"]: row for row in herb_rows}
        missing, details = [], {}
        for name in requested:
            if name not in by_name:
                missing.append(name)
                continue
            query = {key: values[0] for key, values in params.items()}
            query["qr"] = by_name[name]["herb_en_name"]
            url = "https://old.tcmsp-e.com/" + parsed.path + "?" + urlencode(query)
            path = self.fetch(url, "tcmsp/herbs/" + name + ".html")
            tables = embedded_tables(path.read_text(encoding="utf8"))
            ingredients = next((table for table in tables if table and "ob" in table[0]), [])
            associations = next((table for table in tables if table and "molecule_ID" in table[0]
                                 and "target_ID" in table[0]), [])
            if not ingredients:
                raise ValueError(f"No ingredient table returned for {name}")
            herb_id = "TCMSP:herb:" + name
            for row in ingredients:
                connection.execute("INSERT OR IGNORE INTO edges VALUES(?,?,?,?,?,?,?)", (
                    "TCMSP", "2.3", "contains", herb_id, "TCMSP:molecule:" + str(row["molecule_ID"]),
                    None, json.dumps({"url": url, "row": row}, ensure_ascii=False)))
            for row in associations:
                connection.execute("INSERT OR IGNORE INTO edges VALUES(?,?,?,?,?,?,?)", (
                    "TCMSP", "2.3", "compound_target_association", "TCMSP:molecule:" + str(row["molecule_ID"]),
                    "TCMSP:target:" + str(row["target_ID"]), None,
                    json.dumps({"url": url, "row": row,
                                "interpretation": "Database association; validation/SVM/RF fields preserved, not converted to binding probabilities"}, ensure_ascii=False)))
            details[name] = {"ingredients": len(ingredients), "compound_target_rows": len(associations)}
            time.sleep(0.5)
        return {"status": "official_browse_catalogues_imported", "tables": counts,
                "herb_details": details, "unmatched_application_herbs": missing,
                "scope": "Four complete public browse tables plus exact-matched application herbs' ingredient/target relations; not a complete relationship export",
                "license": "ODbL-1.0 / DbCL-1.0", "license_url": "https://old.tcmsp-e.com/tcmsp.php"}

    def string(self, connection, threshold=700):
        version = "12.0"
        root_url = "https://stringdb-downloads.org/download/"
        info = self.fetch(root_url + "protein.info.v12.0/9606.protein.info.v12.0.txt.gz",
                          "string/9606.protein.info.v12.0.txt.gz")
        links = self.fetch(root_url + "protein.links.v12.0/9606.protein.links.v12.0.txt.gz",
                           "string/9606.protein.links.v12.0.txt.gz")
        proteins = set()
        with gzip.open(info, "rt", encoding="utf8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                entity_id = row["#string_protein_id"]
                proteins.add(entity_id)
                connection.execute("INSERT INTO entities VALUES(?,?,?,?,?,?)", (
                    "STRING", version, "human_protein", entity_id, row["preferred_name"], json.dumps(row)))
        count = total = unknown = 0
        batch = []
        with gzip.open(links, "rt", encoding="utf8") as handle:
            if handle.readline().split() != ["protein1", "protein2", "combined_score"]:
                raise ValueError("Unexpected STRING link schema")
            for line in handle:
                a, b, score_text = line.split()
                total += 1
                score = int(score_text)
                if not 0 <= score <= 1000:
                    raise ValueError("STRING score outside 0..1000")
                if score < threshold or a >= b:
                    continue
                if a not in proteins or b not in proteins:
                    unknown += 1
                    continue
                batch.append(("STRING", version, "functional_association", a, b, score / 1000,
                              "9606.protein.links.v12.0.txt.gz; combined_score/1000; undirected pair stored once"))
                count += 1
                if len(batch) >= 10000:
                    connection.executemany("INSERT INTO edges VALUES(?,?,?,?,?,?,?)", batch)
                    batch.clear()
        connection.executemany("INSERT INTO edges VALUES(?,?,?,?,?,?,?)", batch)
        return {"status": "human_network_imported", "species": 9606, "version": version,
                "proteins": len(proteins), "source_rows": total, "undirected_edges": count,
                "unknown_endpoints": unknown, "minimum_combined_score": threshold,
                "scope": "Complete human links file, filtered >=700; functional associations include predicted and indirect evidence, not physical interactions only",
                "license": "CC-BY-4.0", "license_url": "https://string-db.org/cgi/access"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    importer = Importer(args.root)
    destination = args.root / "data/processed/public_evidence.sqlite"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".building.sqlite")
    if temporary.exists():
        temporary.unlink()
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(),
                "clinical_records": 0, "sources": {}, "files": importer.files,
                "recommendation_integration": "Public evidence search only; does not replace nonclinical training/recommendation data"}
    connection = sqlite3.connect(temporary)
    connection.executescript(SCHEMA)
    try:
        for source in ("string", "symmap", "tcmsp"):
            connection.execute("SAVEPOINT source_import")
            try:
                manifest["sources"][source] = getattr(importer, source)(connection)
                connection.execute("RELEASE source_import")
                print(json.dumps({source: manifest["sources"][source]}, ensure_ascii=False), flush=True)
            except Exception as error:
                connection.execute("ROLLBACK TO source_import")
                connection.execute("RELEASE source_import")
                manifest["sources"][source] = {"status": "failed", "error": str(error)}
                print(f"Import failed for {source}: {error}", flush=True)
        connection.commit()
        manifest["index_counts"] = {
            "entity_rows": [dict(zip(("source", "version", "kind", "rows"), row)) for row in
                            connection.execute("SELECT source,version,kind,count(*) FROM entities GROUP BY source,version,kind")],
            "relations": [dict(zip(("source", "version", "relation", "rows"), row)) for row in
                          connection.execute("SELECT source,version,relation,count(*) FROM edges GROUP BY source,version,relation")],
        }
    finally:
        connection.close()
    failed = any(row["status"] == "failed" for row in manifest["sources"].values())
    # Never silently replace a previously complete index with an incomplete run.
    if failed and destination.exists():
        manifest["index_promoted"] = False
        report = args.root / "data/processed/public_import_failure.json"
    else:
        temporary.replace(destination)
        manifest["index_promoted"] = True
        manifest["index_sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
        report = args.root / "data/public_registry.json"
    report.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf8")
    print(f"Manifest: {report}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
