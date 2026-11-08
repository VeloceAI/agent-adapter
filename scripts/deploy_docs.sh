#!/usr/bin/env bash
# Build the docs and publish them to https://agent-adapter.veloceai.in/ (nginx on the veloceai.in VPS).
# Needs the SSH host alias Hitesh_hostinger in ~/.ssh/config. Usage: bash scripts/deploy_docs.sh
set -euo pipefail

HOST="${DOCS_HOST:-Hitesh_hostinger}"
REMOTE=/root/agent-adapter/docs
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT

PY="$ROOT/.venv/Scripts/python"
[ -x "$PY" ] || PY="$ROOT/.venv/bin/python"
"$PY" -m mkdocs build --strict -q -f "$ROOT/mkdocs.yml" -d "$BUILD/site"

# Unpack beside the live folder, then swap, so readers never see a half-copied site.
tar -czf "$BUILD/site.tgz" -C "$BUILD/site" .
ssh "$HOST" "set -e; rm -rf $REMOTE.new; mkdir -p $REMOTE.new; tar -xzf - -C $REMOTE.new;
  chmod -R a+rX $REMOTE.new; rm -rf $REMOTE.old; [ -d $REMOTE ] && mv $REMOTE $REMOTE.old;
  mv $REMOTE.new $REMOTE; rm -rf $REMOTE.old" < "$BUILD/site.tgz"

ssh "$HOST" "curl -fsS -o /dev/null -w 'live: %{http_code}\n' https://agent-adapter.veloceai.in/"
