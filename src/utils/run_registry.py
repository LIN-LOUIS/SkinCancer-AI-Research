from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


REQUIRED_FIELDS = frozenset(
    {
        "run_id", "source_domain", "tests", "model", "weights_provenance",
        "data_version", "protocol_version", "source_split_hash", "commit_sha",
        "config_hash", "seed", "environment", "epoch", "checkpoint",
        "threshold_origin", "threshold", "timestamp", "metrics",
        "artifact_paths", "state",
    }
)
RUN_STATES = frozenset({"PENDING", "RUNNING", "PASS", "FAILED", "BLOCKED"})


def append_run_record(path: Path, record: Mapping[str, Any]) -> None:
    missing = REQUIRED_FIELDS - set(record)
    if missing:
        raise ValueError(f"run registry missing required fields: {sorted(missing)}")
    if record["state"] not in RUN_STATES:
        raise ValueError("invalid run state")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True))
        handle.write("\n")
