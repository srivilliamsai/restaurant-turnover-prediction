"""Revoke the current CLI OAuth grant without printing credentials."""
import json
import logging
from pathlib import Path
import sys

import requests

logging.disable(logging.CRITICAL)

if sys.argv[1:] != ["--apply"]:
    raise SystemExit("Use --apply to disconnect the current Colab CLI account.")

credential_file = Path.home() / ".config/colab-cli/token.json"
if not credential_file.exists():
    print(json.dumps({"status": "already_signed_out"}))
    raise SystemExit(0)

try:
    credentials = json.loads(credential_file.read_text())
    token = credentials.get("refresh_token") or credentials.get("token")
    if not token:
        raise ValueError("Missing token")
    response = requests.post(
        "https://oauth2.googleapis.com/revoke",
        data={"token": token},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=(10, 30),
        allow_redirects=False,
    )
    if response.status_code != 200:
        print(json.dumps({"status": "revocation_not_confirmed", "http_status": response.status_code}))
        raise SystemExit(1)
    credential_file.unlink()
    print(json.dumps({"status": "revoked_and_local_login_removed", "credential_file_exists": credential_file.exists()}))
except SystemExit:
    raise
except Exception as error:
    # Exceptions from requests can include request data; print only the type.
    print(json.dumps({"status": "disconnect_failed", "error_type": type(error).__name__}))
    raise SystemExit(1)
