import gzip
import importlib.util
import json
from pathlib import Path
import sqlite3

from dmease.public_data import PublicEvidenceStore, SCHEMA, embedded_tables

ROOT = Path(__file__).parents[1]


def test_website_tables_are_parsed_without_executing_script():
    html = 'data: [{"name":"黄芪","text":"] end"}], callback: malicious(); data: execute()'
    assert embedded_tables(html) == [[{"name": "黄芪", "text": "] end"}]]
    assert embedded_tables('data: [malicious()]') == []


def test_search_literal_text_and_both_directions(tmp_path):
    path = tmp_path / "evidence.sqlite"
    with sqlite3.connect(path) as connection:
        connection.executescript(SCHEMA)
        connection.execute("INSERT INTO entities VALUES(?,?,?,?,?,?)", ("STRING", "12.0", "protein", "9606.a", "AKT1", "{}"))
        connection.execute("INSERT INTO edges VALUES(?,?,?,?,?,?,?)", ("STRING", "12.0", "functional_association", "9606.a", "9606.b", .9, "source"))
    store = PublicEvidenceStore(path)
    assert store.search("akt1")[0]["id"] == "9606.a"
    assert store.search("AKT1", "TCMSP") == []
    assert store.search("%") == []
    assert store.search("' OR 1=1--") == []
    assert store.neighbors("9606.a") == store.neighbors("9606.b")
    assert not PublicEvidenceStore(tmp_path / "absent.sqlite").available


def test_string_species_threshold_ids_and_symmetric_deduplication(tmp_path):
    spec = importlib.util.spec_from_file_location("public_import", ROOT / "scripts/import_public_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    info, links = tmp_path / "info.gz", tmp_path / "links.gz"
    with gzip.open(info, "wt") as handle:
        handle.write("#string_protein_id\tpreferred_name\n9606.a\tA\n9606.b\tB\n9606.c\tC\n")
    with gzip.open(links, "wt") as handle:
        handle.write("protein1 protein2 combined_score\n9606.a 9606.b 700\n9606.b 9606.a 700\n9606.a 9606.c 699\n9606.a 9606.d 950\n")
    importer = module.Importer(tmp_path)
    importer.fetch = lambda url, relative: info if ".info." in relative else links
    with sqlite3.connect(":memory:") as connection:
        connection.executescript(SCHEMA)
        result = importer.string(connection)
        assert result["undirected_edges"] == 1
        assert result["unknown_endpoints"] == 1
        assert connection.execute("SELECT score FROM edges").fetchone()[0] == .7


def test_herb_evidence_requires_exact_herb_and_preserves_original_claim(tmp_path):
    path = tmp_path / "evidence.sqlite"
    with sqlite3.connect(path) as connection:
        connection.executescript(SCHEMA)
        connection.executemany("INSERT INTO entities VALUES(?,?,?,?,?,?)", [
            ("TCMSP", "2.3", "ingredients", "TCMSP:molecule:1", "compound", "{}"),
            ("TCMSP", "2.3", "targets", "TCMSP:target:2", "target", "{}"),
        ])
        connection.executemany("INSERT INTO edges VALUES(?,?,?,?,?,?,?)", [
            ("TCMSP", "2.3", "contains", "TCMSP:herb:地黄", "TCMSP:molecule:1", None, "source"),
            ("TCMSP", "2.3", "compound_target_association", "TCMSP:molecule:1", "TCMSP:target:2", None, '{"validated":"","RF_score":"0.8"}'),
        ])
    store = PublicEvidenceStore(path)
    assert store.herb_paths("生地黄") == []
    rows = store.herb_paths("地黄")
    assert len(rows) == 1
    assert json.loads(rows[0]["provenance"])["validated"] == ""
