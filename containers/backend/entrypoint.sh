#!/bin/sh
set -e
# The guardrail classifier is loaded from the local cache only. Fetch it on first start
# (the cache is a volume, so later starts skip the download). Never block startup on it:
# without the model the guardrails fall back to regexes and log a warning.
if [ "${GUARDRAIL_CLASSIFIER:-true}" != "false" ] && [ "${USE_GUARDRAILS:-true}" != "false" ]; then
    python -m src.guardrails.download || echo "WARNING: guardrail model download failed; classifier disabled"
fi
exec "$@"
