#!/bin/sh
set -eu

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    for cfg in \
        "${OPENROUTER_CONFIG_DIR:-/config}/secrets.toml" \
        "${OPENROUTER_CONFIG_DIR:-/config}/openrouter.toml"
    do
        [ -f "$cfg" ] || continue
        OPENROUTER_API_KEY="$(python - "$cfg" <<'PY'
import sys

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

with open(sys.argv[1], "rb") as fh:
    data = tomllib.load(fh)

key = data.get("OPENROUTER_API_KEY") or data.get("openrouter_api_key")
if not key:
    for section in ("openrouter", "api"):
        if isinstance(data.get(section), dict):
            key = data[section].get("api_key") or data[section].get("key")
            if key:
                break

print(key or "", end="")
PY
)"
        export OPENROUTER_API_KEY
        break
    done
fi

exec "$@"
