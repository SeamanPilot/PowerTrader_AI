from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
from typing import Dict, Any


CONFIG_PATH = Path("stockbot_config.json")


@dataclass
class BrokerCredentials:
    broker: str
    api_key: str = ""
    api_secret: str = ""
    endpoint: str = ""
    extra: Dict[str, Any] | None = None


@dataclass
class BotSettings:
    symbols: list[str]
    initial_cash: float = 100_000
    position_size_pct: float = 0.1
    timeframe: str = "1d"
    lookback: str = "2y"


def save_config(credentials: BrokerCredentials, settings: BotSettings, path: Path = CONFIG_PATH) -> None:
    payload = {
        "credentials": asdict(credentials),
        "settings": asdict(settings),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_config(path: Path = CONFIG_PATH) -> tuple[BrokerCredentials, BotSettings]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cred = BrokerCredentials(**data["credentials"])
    settings = BotSettings(**data["settings"])
    return cred, settings
