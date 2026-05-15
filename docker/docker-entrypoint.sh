#!/usr/bin/env bash
set -euo pipefail
cd /app
export PYTHONPATH="${PYTHONPATH:-/app}"

python3 <<'PY'
import os
import socket
import time


def wait_tcp(host: str, port: int, timeout_sec: float) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=3):
                print(f"[entrypoint] {host}:{port} ready")
                return
        except OSError:
            time.sleep(2)
    raise SystemExit(f"[entrypoint] timeout waiting for {host}:{port}")


wait_tcp(os.environ.get("WAIT_NEO4J_HOST", "neo4j"), int(os.environ.get("WAIT_NEO4J_PORT", "7687")), 180)
wait_tcp(os.environ.get("WAIT_MILVUS_HOST", "milvus"), int(os.environ.get("WAIT_MILVUS_PORT", "19530")), 300)
PY

mkdir -p data/raw data/processed data/meta data/logs data/lightrag_workdir

HOST="${SERVICE_HOST:-0.0.0.0}"
PORT="${SERVICE_PORT:-8000}"
exec python3 -m uvicorn src.service.api:app --host "$HOST" --port "$PORT"
