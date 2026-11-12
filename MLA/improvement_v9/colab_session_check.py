"""Inspect and register an existing Colab runtime without starting a kernel.

Authentication stays in Google's normal CLI credential store. This script never
prints access tokens, proxy URLs, authorization codes, or credential files.
"""
import json
import os
from pathlib import Path

import requests
from colab_cli.common import state
from colab_cli.contents import ContentsClient
from colab_cli.state import SessionState

os.umask(0o077)
assignments = state.client.list_assignments()
assert len(assignments) == 1, "Expected one existing runtime; inspect before selecting."
assignment = assignments[0]
info = assignment.runtime_proxy_info
headers = {"X-Colab-Runtime-Proxy-Token": info.token,
           "X-Colab-Client-Agent": "colab-cli"}
params = {"authuser": "REDACTED", "colab-runtime-proxy-token": info.token}

def get_json(relative):
    response = requests.get(info.url.rstrip("/") + relative,
                            headers=headers, params=params, timeout=30)
    if not response.ok:
        raise RuntimeError(f"Runtime inspection HTTP {response.status_code}")
    return response.json()

kernels = get_json("/api/kernels")
sessions = get_json("/api/sessions")
session = SessionState(name="v9-existing-t4", endpoint=assignment.endpoint,
    token=info.token, url=info.url, token_expires_at=info.expires_at(),
    accelerator=assignment.accelerator.value, variant=assignment.variant.name,
    machine_shape=assignment.machine_shape.name)
if len(kernels) == 1:
    session.kernel_id = kernels[0]["id"]
    matching = [s for s in sessions if s.get("kernel", {}).get("id") == session.kernel_id]
    if len(matching) == 1:
        session.session_id = matching[0]["id"]
state.store.add(session)
listing = ContentsClient(session).list_dir("content")
record = {"session": session.name, "accelerator": session.accelerator,
    "kernels": [{k: v for k, v in kernel.items()
                 if k in ["id", "execution_state", "last_activity", "connections"]}
                for kernel in kernels],
    "notebooks": [{"name": s.get("name"), "path": s.get("path"),
                   "kernel_id": s.get("kernel", {}).get("id")} for s in sessions],
    "content": [{"name": p["name"], "type": p.get("type")}
                for p in listing.get("content", [])]}
destination = Path(__file__).resolve().parent / "results" / "colab_cli_session.json"
destination.write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
