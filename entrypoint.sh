#!/bin/sh
set -eu

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    for cfg in \
        "${OPENROUTER_CONFIG_DIR:-/config}/secrets.toml" \
        "${OPENROUTER_CONFIG_DIR:-/config}/openrouter.toml"
    do
        [ -f "$cfg" ] || continue
        if ! OPENROUTER_API_KEY="$(python - "$cfg" <<'PY'
import sys

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

try:
    with open(sys.argv[1], "rb") as fh:
        data = tomllib.load(fh)
except Exception as exc:
    print(f"warning: cannot parse {sys.argv[1]}: {exc}", file=sys.stderr)
    raise SystemExit(1)

key = data.get("OPENROUTER_API_KEY") or data.get("openrouter_api_key")
if not key:
    for section in ("openrouter", "api"):
        if isinstance(data.get(section), dict):
            key = data[section].get("api_key") or data[section].get("key")
            if key:
                break

print(str(key).strip() if key else "", end="")
PY
        )"; then
            echo "warning: skipping unreadable API key file $cfg" >&2
            OPENROUTER_API_KEY=""
            continue
        fi
        if [ -n "$OPENROUTER_API_KEY" ]; then
            echo "Loaded OpenRouter API key from $cfg" >&2
        else
            echo "warning: no API key found in $cfg" >&2
        fi
        export OPENROUTER_API_KEY
        break
    done
fi

exec "$@"
