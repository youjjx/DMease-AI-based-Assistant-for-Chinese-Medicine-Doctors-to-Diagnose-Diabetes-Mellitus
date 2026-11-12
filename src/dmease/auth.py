"""Local account authentication without bundled credentials."""
from __future__ import annotations

from contextlib import closing

import hashlib
import hmac
import re
import secrets
import sqlite3
import time
from pathlib import Path


class AuthStore:
    iterations = 600_000

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db, db:
            db.execute("""CREATE TABLE IF NOT EXISTS accounts (
                username TEXT PRIMARY KEY, salt TEXT NOT NULL, digest TEXT NOT NULL,
                iterations INTEGER NOT NULL, failures INTEGER NOT NULL DEFAULT 0,
                locked_until REAL NOT NULL DEFAULT 0, revision INTEGER NOT NULL DEFAULT 1
            )""")

    def _connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        return db

    @staticmethod
    def validate_password(password: str) -> None:
        if not 12 <= len(password) <= 128 or not re.search(r"[A-Za-z]", password) or not re.search(r"\d", password):
            raise ValueError("密码须为12—128位，且同时包含英文字母和数字。")

    @staticmethod
    def _digest(password: str, salt: str, iterations: int) -> str:
        return hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), iterations).hex()

    def initialized(self) -> bool:
        with closing(self._connect()) as db, db:
            return db.execute("SELECT 1 FROM accounts LIMIT 1").fetchone() is not None

    def bootstrap(self, username: str, password: str) -> None:
        """Create the first account atomically; setup cannot run again after initialization."""
        username = username.strip()
        if not re.fullmatch(r"[A-Za-z0-9_]{3,32}", username):
            raise ValueError("账号须为3—32位英文字母、数字或下划线。")
        self.validate_password(password)
        salt = secrets.token_hex(16)
        digest = self._digest(password, salt, self.iterations)
        with closing(self._connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT 1 FROM accounts LIMIT 1").fetchone():
                raise ValueError("系统已完成初始化，请返回登录。")
            db.execute("INSERT INTO accounts(username,salt,digest,iterations) VALUES (?,?,?,?)",
                       (username, salt, digest, self.iterations))

    def authenticate(self, username: str, password: str) -> int | None:
        """Return credential revision on success; persist failed-attempt throttling."""
        username = username.strip()
        with closing(self._connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM accounts WHERE username=?", (username,)).fetchone()
            if row is None:
                self._digest(password[:128], "00" * 16, self.iterations)
                return None
            if row["locked_until"] > time.time():
                return None
            valid = len(password) <= 128 and hmac.compare_digest(
                self._digest(password, row["salt"], row["iterations"]), row["digest"])
            if not valid:
                failures = row["failures"] + 1
                locked = time.time() + 60 if failures >= 5 else 0
                db.execute("UPDATE accounts SET failures=?,locked_until=? WHERE username=?",
                           (0 if locked else failures, locked, username))
                return None
            db.execute("UPDATE accounts SET failures=0,locked_until=0 WHERE username=?", (username,))
            return int(row["revision"])

    def revision(self, username: str) -> int | None:
        with closing(self._connect()) as db, db:
            row = db.execute("SELECT revision FROM accounts WHERE username=?", (username,)).fetchone()
        return int(row[0]) if row else None

    def change_password(self, username: str, old: str, new: str) -> None:
        self.validate_password(new)
        revision = self.authenticate(username, old)
        if revision is None:
            raise ValueError("当前密码不正确或账号暂时锁定。")
        salt = secrets.token_hex(16)
        digest = self._digest(new, salt, self.iterations)
        with closing(self._connect()) as db, db:
            cur = db.execute("UPDATE accounts SET salt=?,digest=?,iterations=?,revision=revision+1 "
                             "WHERE username=? AND revision=?",
                             (salt, digest, self.iterations, username, revision))
            if not cur.rowcount:
                raise ValueError("账号状态已变化，请重新登录。")
