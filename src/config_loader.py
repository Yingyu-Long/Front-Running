"""Configuration loader for universe, portfolio, and runtime settings."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional in bare test env
    def load_dotenv(*_args: Any, **_kwargs: Any) -> bool:
        return False

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"


@dataclass
class AppSettings:
    llm_base_url: str = "http://127.0.0.1:11434/v1"
    llm_api_key: str = "ollama"
    llm_model: str = "qwen3:14b"
    llm_timeout_sec: int = 180
    llm_temperature: float = 0.2
    compute_device: str = "auto"
    mc_simulations: int = 100_000
    sec_user_agent: str = "FrontRunningQuantResearch/1.0 (trader@local-research.dev)"
    history_years: int = 10
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    reports_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "reports")
    db_path: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "trading_desk.db")

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)


def load_settings() -> AppSettings:
    """Load runtime configuration from .env and environment variables."""
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=False)

    def _resolve_path(val: str) -> Path:
        p = Path(val)
        return p if p.is_absolute() else PROJECT_ROOT / p

    settings = AppSettings(
        llm_base_url=os.getenv("LLM_BASE_URL", "http://127.0.0.1:11434/v1").strip(),
        llm_api_key=os.getenv("LLM_API_KEY", "ollama").strip(),
        llm_model=os.getenv("LLM_MODEL", "qwen3:14b").strip(),
        llm_timeout_sec=int(os.getenv("LLM_TIMEOUT_SEC", "180")),
        llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
        compute_device=os.getenv("COMPUTE_DEVICE", "auto").strip().lower(),
        mc_simulations=int(os.getenv("MC_SIMULATIONS", "100000")),
        sec_user_agent=os.getenv(
            "SEC_USER_AGENT",
            "FrontRunningQuantResearch/1.0 (trader@local-research.dev)",
        ).strip(),
        history_years=int(os.getenv("HISTORY_YEARS", "10")),
        data_dir=_resolve_path(os.getenv("DATA_DIR", "data")),
        reports_dir=_resolve_path(os.getenv("REPORTS_DIR", "reports")),
        db_path=_resolve_path(os.getenv("DB_PATH", "data/trading_desk.db")),
    )
    settings.ensure_dirs()
    return settings


def _require_yaml() -> Any:
    if yaml is None:
        raise RuntimeError("PyYAML is required. Install with: pip install PyYAML")
    return yaml


def load_universe(path: Path | None = None) -> dict[str, Any]:
    """Load stock universe, sector definitions, supply-chain graph, and rotation pairs."""
    target = path or (CONFIG_DIR / "universe.yaml")
    y = _require_yaml()
    with target.open("r", encoding="utf-8") as f:
        data = y.safe_load(f) or {}
    return data


def save_universe(universe_data: dict[str, Any], path: Path | None = None) -> None:
    """Persist updated watchlist/universe back to universe.yaml."""
    target = path or (CONFIG_DIR / "universe.yaml")
    y = _require_yaml()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        y.safe_dump(universe_data, f, allow_unicode=True, sort_keys=False, indent=2)


def load_portfolio(path: Path | None = None) -> dict[str, Any]:
    """Load user's account settings and current holdings."""
    target = path or (CONFIG_DIR / "portfolio.yaml")
    y = _require_yaml()
    with target.open("r", encoding="utf-8") as f:
        data = y.safe_load(f) or {}
    return data


def save_portfolio(portfolio_data: dict[str, Any], path: Path | None = None) -> None:
    """Persist updated account settings and holdings back to portfolio.yaml."""
    target = path or (CONFIG_DIR / "portfolio.yaml")
    y = _require_yaml()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        y.safe_dump(portfolio_data, f, allow_unicode=True, sort_keys=False, indent=2)


def get_all_equity_symbols(universe_cfg: dict[str, Any]) -> list[str]:
    """Return ordered list of all tracked stocks (holdings + watchlist + bellwethers)."""
    return list((universe_cfg.get("universe") or {}).keys())


def get_benchmark_symbols(universe_cfg: dict[str, Any]) -> list[str]:
    """Return ordered list of macro/sector benchmark tickers."""
    return list((universe_cfg.get("benchmarks") or {}).keys())
