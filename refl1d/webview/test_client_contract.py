"""
Check that the socket.io event names used by the Refl1D webview client match the server.

The client calls server handlers and listens for server events by name, through
an untyped socket, so a rename on either side is invisible to the type checker
and to the client unit tests (which use a fake socket). These tests scan the
client source for event names and compare them with what the server registers,
including the handlers Refl1D adds to bumps.
"""

import re
from dataclasses import fields
from pathlib import Path
from typing import get_args

import pytest

CLIENT_SRC = Path(__file__).parent / "client" / "src"

# Client files that nothing imports, so their event names don't matter.
UNUSED_FILES = {
    "ModelViewPlotly.vue",  # older Profile panel; calls get_profile_data, which the server no longer has
}


def _client_names(pattern):
    if not CLIENT_SRC.exists():
        pytest.skip("webview client source not available")
    sources = (p for p in CLIENT_SRC.rglob("*") if p.suffix in {".ts", ".vue", ".js"} and p.name not in UNUSED_FILES)
    text = "\n".join(p.read_text(encoding="utf-8") for p in sources)
    return set(re.findall(pattern, text))


def test_client_calls_exist_on_server():
    try:
        from bumps import api
        from bumps.webview import webserver  # noqa: F401 -- registers the bumps handlers
    except ImportError:
        # CRUFT: bumps < 1.1
        from bumps.webview.server import api
        from bumps.webview.server import webserver  # noqa: F401
    from refl1d.webview.server import api as refl1d_api  # noqa: F401 -- registers the refl1d handlers

    called = _client_names(r"""asyncEmit\(\s*["'`](\w+)["'`]""")
    assert called, "found no asyncEmit calls; has the client API changed?"
    missing = called - set(api.REGISTRY)
    assert not missing, f"client calls handlers the server does not define: {sorted(missing)}"


def test_client_listens_for_events_the_server_sends():
    try:
        from bumps.state import SharedState, TopicNameType
    except ImportError:
        # CRUFT: bumps < 1.1
        from bumps.webview.server.state_hdf5_backed import SharedState, TopicNameType

    listened = _client_names(r"""\.on\(\s*["'](\w+)["']""") | _client_names(r"""setupDrawLoop\(\s*["'](\w+)["']""")
    assert listened, "found no event listeners; has the client API changed?"
    # every shared-state change is broadcast under the field name (api.notify_shared_setting)
    shared = {f.name for f in fields(SharedState) if not f.name.startswith("_")}
    topics = set(get_args(TopicNameType))
    unknown = listened - shared - topics
    assert not unknown, f"client listens for events the server never sends: {sorted(unknown)}"
