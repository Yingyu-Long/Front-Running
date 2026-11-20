"""SQLite storage for analysis snapshots, week-over-week comparison, caches, and trade journal."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class TradingDB:
    """Lightweight local SQLite database for on-demand trading desk workflows."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS run_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    run_label TEXT,
                    device_info TEXT,
                    llm_model TEXT,
                    regime_score REAL,
                    regime_label TEXT,
                    payload_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS kv_cache (
                    cache_key TEXT PRIMARY KEY,
                    updated_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS trade_journal (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    shares REAL NOT NULL DEFAULT 0,
                    price REAL NOT NULL DEFAULT 0,
                    stop_loss REAL,
                    target_price REAL,
                    setup_tag TEXT,
                    conviction INTEGER DEFAULT 3,
                    notes TEXT,
                    outcome_pnl REAL,
                    status TEXT NOT NULL DEFAULT 'OPEN'
                );

                CREATE INDEX IF NOT EXISTS idx_snapshots_created
                    ON run_snapshots(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_journal_symbol
                    ON trade_journal(symbol, created_at DESC);
                """
            )

    # ------------------------------------------------------------------
    # Run snapshots (persist every full run so PC can shut down anytime)
    # ------------------------------------------------------------------
    def save_snapshot(
        self,
        payload: dict[str, Any],
        run_label: str = "full_scan",
        device_info: str = "",
        llm_model: str = "",
    ) -> int:
        now = datetime.now(timezone.utc).isoformat()
        regime = payload.get("regime") or {}
        regime_score = float(regime.get("regime_score", 0.0))
        regime_label = str(regime.get("regime_label", ""))

        serialized = json.dumps(payload, ensure_ascii=False, default=_json_default)
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO run_snapshots
                    (created_at, run_label, device_info, llm_model, regime_score, regime_label, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (now, run_label, device_info, llm_model, regime_score, regime_label, serialized),
            )
            return int(cur.lastrowid or 0)

    def list_snapshots(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, created_at, run_label, device_info, llm_model, regime_score, regime_label
                FROM run_snapshots
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_snapshot(self, snapshot_id: int | None = None) -> dict[str, Any] | None:
        with self._connect() as conn:
            if snapshot_id is None:
                row = conn.execute(
                    "SELECT * FROM run_snapshots ORDER BY id DESC LIMIT 1"
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM run_snapshots WHERE id = ?", (snapshot_id,)
                ).fetchone()
        if not row:
            return None
        record = dict(row)
        record["payload"] = json.loads(record.pop("payload_json"))
        return record

    def get_previous_snapshot(self, before_id: int) -> dict[str, Any] | None:
        """Fetch the immediately preceding snapshot for week-over-week signal diffs."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM run_snapshots WHERE id < ? ORDER BY id DESC LIMIT 1",
                (before_id,),
            ).fetchone()
        if not row:
            return None
        record = dict(row)
        record["payload"] = json.loads(record.pop("payload_json"))
        return record

    # ------------------------------------------------------------------
    # Generic JSON cache (SEC CIK maps, filings, etc.)
    # ------------------------------------------------------------------
    def get_cache(self, key: str, max_age_hours: float = 24.0) -> Any | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT updated_at, payload_json FROM kv_cache WHERE cache_key = ?",
                (key,),
            ).fetchone()
        if not row:
            return None
        updated = datetime.fromisoformat(row["updated_at"])
        age_hours = (datetime.now(timezone.utc) - updated).total_seconds() / 3600.0
        if age_hours > max_age_hours:
            return None
        return json.loads(row["payload_json"])

    def set_cache(self, key: str, value: Any) -> None:
        now = datetime.now(timezone.utc).isoformat()
        serialized = json.dumps(value, ensure_ascii=False, default=_json_default)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO kv_cache (cache_key, updated_at, payload_json)
                VALUES (?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    updated_at = excluded.updated_at,
                    payload_json = excluded.payload_json
                """,
                (key, now, serialized),
            )

    # ------------------------------------------------------------------
    # Trade Journal (log decisions, entries, exits, and post-mortems)
    # ------------------------------------------------------------------
    def add_journal_entry(
        self,
        symbol: str,
        action: str,
        shares: float,
        price: float,
        stop_loss: float | None = None,
        target_price: float | None = None,
        setup_tag: str = "",
        conviction: int = 3,
        notes: str = "",
        status: str = "OPEN",
        outcome_pnl: float | None = None,
    ) -> int:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO trade_journal
                    (created_at, symbol, action, shares, price, stop_loss, target_price,
                     setup_tag, conviction, notes, outcome_pnl, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now,
                    symbol.upper().strip(),
                    action.upper().strip(),
                    float(shares),
                    float(price),
                    stop_loss,
                    target_price,
                    setup_tag,
                    int(conviction),
                    notes,
                    outcome_pnl,
                    status.upper().strip(),
                ),
            )
            return int(cur.lastrowid or 0)

    def list_journal_entries(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM trade_journal ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def close_journal_entry(
        self, entry_id: int, outcome_pnl: float, notes_append: str = ""
    ) -> None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT notes FROM trade_journal WHERE id = ?", (entry_id,)
            ).fetchone()
            existing = (row["notes"] or "") if row else ""
            new_notes = f"{existing}\n[平仓复盘] {notes_append}".strip() if notes_append else existing
            conn.execute(
                """
                UPDATE trade_journal
                SET status = 'CLOSED', outcome_pnl = ?, notes = ?
                WHERE id = ?
                """,
                (float(outcome_pnl), new_notes, int(entry_id)),
            )

    def delete_journal_entry(self, entry_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM trade_journal WHERE id = ?", (int(entry_id),))


def _json_default(obj: Any) -> Any:
    """Safe JSON fallback for numpy/pandas/datetime/Path objects."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            pass
    return str(obj)
