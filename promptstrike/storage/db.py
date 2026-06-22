"""SQLite persistence layer — stores campaigns, attack runs, and per-iteration data."""

import sqlite3
import json
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from promptstrike.core.pair import PairResult, AttackStatus


_SCHEMA = """
CREATE TABLE IF NOT EXISTS campaigns (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL,
    target      TEXT    NOT NULL,
    algorithm   TEXT    NOT NULL DEFAULT 'pair',
    created_at  TEXT    NOT NULL,
    total_goals INTEGER NOT NULL DEFAULT 0,
    succeeded   INTEGER NOT NULL DEFAULT 0,
    asr         REAL    NOT NULL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS attack_runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    campaign_id  INTEGER NOT NULL REFERENCES campaigns(id),
    behavior_id  INTEGER NOT NULL,
    goal         TEXT    NOT NULL,
    category     TEXT    NOT NULL DEFAULT '',
    owasp        TEXT    NOT NULL DEFAULT '',
    target_model TEXT    NOT NULL,
    algorithm    TEXT    NOT NULL DEFAULT 'pair',
    status       TEXT    NOT NULL,
    final_score  INTEGER NOT NULL DEFAULT 0,
    calls_used   INTEGER NOT NULL DEFAULT 0,
    winning_prompt TEXT  NOT NULL DEFAULT '',
    created_at   TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS iterations (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id           INTEGER NOT NULL REFERENCES attack_runs(id),
    iteration_number INTEGER NOT NULL,
    attacker_prompt  TEXT    NOT NULL,
    target_response  TEXT    NOT NULL,
    judge_score      INTEGER NOT NULL,
    is_jailbroken    INTEGER NOT NULL DEFAULT 0
);
"""


class Database:
    def __init__(self, path: str | Path = "promptstrike.db") -> None:
        self._path = str(path)
        self._init()

    def _init(self) -> None:
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def create_campaign(self, name: str, target: str, algorithm: str = "pair") -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO campaigns (name, target, algorithm, created_at) VALUES (?,?,?,?)",
                (name, target, algorithm, _now()),
            )
            return cur.lastrowid

    def save_run(
        self,
        campaign_id: int,
        result: PairResult,
        category: str = "",
        owasp: str = "",
    ) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO attack_runs
                   (campaign_id, behavior_id, goal, category, owasp, target_model,
                    algorithm, status, final_score, calls_used, winning_prompt, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    campaign_id,
                    result.behavior_id,
                    result.goal,
                    category,
                    owasp,
                    result.target_model,
                    "pair",
                    result.status.value,
                    result.final_score,
                    result.calls_used,
                    result.winning_prompt,
                    _now(),
                ),
            )
            run_id = cur.lastrowid

            for it in result.iterations:
                conn.execute(
                    """INSERT INTO iterations
                       (run_id, iteration_number, attacker_prompt,
                        target_response, judge_score, is_jailbroken)
                       VALUES (?,?,?,?,?,?)""",
                    (
                        run_id,
                        it.number,
                        it.attacker_prompt,
                        it.target_response,
                        it.judge.score,
                        int(it.judge.is_jailbroken),
                    ),
                )
            return run_id

    def finalize_campaign(self, campaign_id: int) -> None:
        with self._conn() as conn:
            conn.execute(
                """UPDATE campaigns
                   SET total_goals = (SELECT COUNT(*) FROM attack_runs WHERE campaign_id=?),
                       succeeded   = (SELECT COUNT(*) FROM attack_runs
                                      WHERE campaign_id=? AND status='success'),
                       asr         = CAST((SELECT COUNT(*) FROM attack_runs
                                           WHERE campaign_id=? AND status='success') AS REAL)
                                     / MAX(1, (SELECT COUNT(*) FROM attack_runs WHERE campaign_id=?))
                   WHERE id=?""",
                (campaign_id,) * 5,
            )

    def get_campaign_runs(self, campaign_id: int) -> list[sqlite3.Row]:
        with self._conn() as conn:
            return conn.execute(
                "SELECT * FROM attack_runs WHERE campaign_id=? ORDER BY id",
                (campaign_id,),
            ).fetchall()

    def get_campaign(self, campaign_id: int) -> sqlite3.Row | None:
        with self._conn() as conn:
            return conn.execute(
                "SELECT * FROM campaigns WHERE id=?", (campaign_id,)
            ).fetchone()

    def list_campaigns(self) -> list[sqlite3.Row]:
        with self._conn() as conn:
            return conn.execute(
                "SELECT * FROM campaigns ORDER BY created_at DESC"
            ).fetchall()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
