import sqlite3
import time

import pytest

from dmease.auth import AuthStore
from dmease.exports import csv_export, report_html
from dmease.patient_db import PatientDatabase, SCHEMA
from dmease.types import AnalysisResult, PatientProfile


def result(name="P001", complaint="口渴"):
    return AnalysisResult(PatientProfile(name=name, complaint=complaint), [], [], [], {}, [])


def test_account_setup_verification_throttling_and_password_change(tmp_path):
    store = AuthStore(tmp_path / "accounts.db")
    password = "Test-password1234"
    with pytest.raises(ValueError):
        store.bootstrap("operator", "short")
    store.bootstrap("operator", password)
    with pytest.raises(ValueError):
        store.bootstrap("another", password)
    assert store.authenticate("operator", password) == 1
    with sqlite3.connect(store.path) as db:
        row = db.execute("SELECT salt,digest FROM accounts").fetchone()
        assert password not in row and len(row[0]) == 32
    for _ in range(5):
        assert store.authenticate("operator", "incorrect-password") is None
    assert store.authenticate("operator", password) is None
    with sqlite3.connect(store.path) as db:
        db.execute("UPDATE accounts SET locked_until=?", (time.time() - 1,))
    store.change_password("operator", password, "Updated-password1234")
    assert store.revision("operator") == 2
    assert store.authenticate("operator", password) is None
    assert store.authenticate("operator", "Updated-password1234") == 2


def test_old_database_migration_isolation_duplicate_save_and_version_link(tmp_path):
    path = tmp_path / "patients.db"
    with sqlite3.connect(path) as db:
        db.executescript(SCHEMA)
        db.execute("INSERT INTO patient_records(patient_name,created_at,complaint,payload_json) VALUES ('legacy','2020','','{}')")
    store = PatientDatabase(path)
    assert store.count("operator") == 0
    sample = result()
    first = store.save(sample, "operator")
    assert store.save(sample, "operator") == first
    assert store.count("operator") == 1
    assert store.get(first, owner="other") is None
    assert not store.delete(first, owner="other")
    with pytest.raises(ValueError):
        store.save(result("P002"), "other", first)
    second = store.save(result(complaint="口干"), "operator", first)
    assert store.list(owner="operator", query="口干")[0]["parent_record_id"] == first
    assert store.get(first, owner="operator")["patient"]["complaint"] == "口渴"
    assert store.list(owner="operator", query="' OR 1=1 --") == []
    assert store.delete(second, owner="operator")
    assert store.get(first, owner="operator") is not None


def test_export_escapes_markup_and_spreadsheet_formulas():
    payload = result(name='<script>alert(1)</script>', complaint='=HYPERLINK("x")').to_dict()
    report = report_html(payload)
    assert "<script>" not in report and "&lt;script&gt;" in report
    text = csv_export([{"患者": "=cmd", "分数": -0.2}]).decode("utf-8-sig")
    assert "'=cmd" in text
    assert "-0.2" in text
