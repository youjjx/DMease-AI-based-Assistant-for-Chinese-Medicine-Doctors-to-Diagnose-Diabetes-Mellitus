"""Read-only queries and parsers for separately stored public research evidence."""
from __future__ import annotations

import json
from pathlib import Path
import re
import sqlite3


def embedded_tables(html: str) -> list[list[dict]]:
    """Decode literal JSON tables; never execute a website's JavaScript."""
    tables = []
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\bdata\s*:\s*(\[)", html):
        try:
            value, _ = decoder.raw_decode(html[match.start(1):])
        except ValueError:
            continue
        if isinstance(value, list) and all(isinstance(row, dict) for row in value):
            tables.append(value)
    return tables


SCHEMA = """
CREATE TABLE entities(source TEXT, version TEXT, kind TEXT, id TEXT, name TEXT,
                      payload TEXT, PRIMARY KEY(source,version,kind,id));
CREATE INDEX entity_lookup ON entities(source,kind,name);
CREATE INDEX entity_id_lookup ON entities(id,source,version);
CREATE TABLE edges(source TEXT, version TEXT, relation TEXT, subject TEXT, object TEXT,
                   score REAL, provenance TEXT,
                   PRIMARY KEY(source,version,relation,subject,object));
CREATE INDEX edge_object ON edges(object);
CREATE INDEX edge_subject ON edges(subject);
"""


class PublicEvidenceStore:
    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()

    @property
    def available(self) -> bool:
        return self.path.is_file()

    def _connect(self):
        return sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)

    def search(self, query: str, source: str = "", limit: int = 30) -> list[dict]:
        if not self.available or not query.strip():
            return []
        # instr treats %, _, quotes and user text literally.
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT source,version,kind,id,name,payload FROM entities "
                "WHERE (instr(lower(name),lower(?))>0 OR id=?) "
                "AND (?='' OR source=?) "
                "ORDER BY CASE WHEN lower(name)=lower(?) OR id=? THEN 0 ELSE 1 END,source,kind,id LIMIT ?",
                (query.strip(), query.strip(), source, source, query.strip(), query.strip(), min(100, max(1, limit))),
            ).fetchall()
            return [dict(row) for row in rows]

    def neighbors(self, entity_id: str, limit: int = 30) -> list[dict]:
        if not self.available:
            return []
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT * FROM edges WHERE subject=? OR object=? "
                "ORDER BY score DESC,subject,object LIMIT ?",
                (entity_id, entity_id, min(100, max(1, limit))),
            ).fetchall()
            return [dict(row) for row in rows]

    def herb_paths(self, herb: str, limit: int = 20) -> list[dict]:
        """Exact TCMSP herb name only; no guessed synonym or efficacy conversion."""
        if not self.available:
            return []
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT a.subject AS herb_id,c.id AS compound_id,c.name AS compound,"
                "t.id AS target_id,t.name AS target,b.source,b.version,b.provenance "
                "FROM edges a JOIN edges b ON a.object=b.subject "
                "JOIN entities c ON c.id=a.object AND c.source='TCMSP' AND c.version=a.version "
                "JOIN entities t ON t.id=b.object AND t.source='TCMSP' AND t.version=b.version "
                "WHERE a.source='TCMSP' AND b.source='TCMSP' AND a.version=b.version "
                "AND a.relation='contains' AND b.relation='compound_target_association' "
                "AND a.subject=? ORDER BY c.id,t.id LIMIT ?",
                ("TCMSP:herb:" + herb, min(100, max(1, limit))),
            ).fetchall()
            return [dict(row) for row in rows]
