import hashlib
import json

from agent_adapter.integrations.mcp import _scope
from agent_adapter.runtime.ids import scoped_id


def test_scoped_id_is_prefix_plus_32_hex_of_the_canonical_pair():
    payload = json.dumps({"k": "key", "p": "alice"}, sort_keys=True, separators=(",", ":"))
    expected = "call_" + hashlib.sha256(payload.encode()).hexdigest()[:32]
    assert scoped_id("call_", "alice", "key") == expected


def test_two_principals_never_share_an_id_for_one_key():
    ids = {scoped_id("run_", p, "same") for p in ("alice", "bob", None, "")}
    assert len(ids) == 4


def test_lone_surrogates_stay_distinct():
    assert scoped_id("run_", "a", "\ud800") != scoped_id("run_", "a", "\ud801")


def test_the_mcp_scope_module_reuses_the_shared_function():
    assert _scope.scoped_id is scoped_id
