"""Learning scope identity, and the migration invariant that protects it.

The point of these tests is not that `resolve_scope` returns a hash — it is that
the hash it returns is the *same one* every stored sample already has, and that
nothing in the runtime snapshot can quietly redefine it.
"""
from __future__ import annotations

from astrbot_plugin_dynamics_learning.core.samples import (
    SAMPLE_SCHEMA_VERSION, LearningSample, build_dataset, samples_from_annotation,
)
from astrbot_plugin_dynamics_learning.core.scope import (
    SCOPE_SOURCE_SESSION, SCOPE_SOURCE_SESSION_UMO_EQUAL, SCOPE_SOURCE_SESSION_UMO_MISMATCH,
    LearningScope, resolve_scope, scope_label, session_hash,
)

from .factories import make_record, make_trace

KEY = "aiocqhttp:GroupMessage:123456"


def _topic_record(msg_id: str = "m1"):
    return make_record(msg_id, trace=make_trace(topic_id="t1", topic_confidence=0.5),
                       predicted_topic="t1", expected_topic="t1")


# ---- the migration invariants ------------------------------------------

def test_scope_fallback_is_legacy_session_hash():
    scope = resolve_scope(KEY, {})

    assert scope.scope_hash == session_hash(KEY)


def test_equal_umo_does_not_change_identity():
    a = resolve_scope(KEY, {})
    b = resolve_scope(KEY, {"umo": KEY})

    assert a.scope_hash == b.scope_hash
    assert a.scope_hash == session_hash(KEY)


def test_no_runtime_field_can_redefine_the_scope():
    """The snapshot may describe the scope; it may never decide it."""
    baseline = session_hash(KEY)
    for runtime in ({}, {"umo": KEY}, {"group_id": "123456"}, {"bot_id": "9"},
                    {"umo": KEY, "group_id": "123456", "bot_id": "9"},
                    {"umo": "somewhere:else:999"}):
        assert resolve_scope(KEY, runtime).scope_hash == baseline


def test_schema_one_row_keeps_its_session_identity():
    sample = build_dataset([(KEY, _topic_record())])[0]
    legacy = sample.as_dict(include_trace=True)
    for key in ("scope_hash", "scope_source", "group_hint_hash", "sample_schema_version"):
        legacy.pop(key)

    restored = LearningSample.from_dict(legacy)

    assert restored is not None
    assert restored.sample_id == sample.sample_id
    assert restored.session_hash == sample.session_hash
    assert restored.scope_hash == sample.session_hash          # byte for byte
    assert restored.scope_source == SCOPE_SOURCE_SESSION
    assert restored.group_hint_hash == ""


def test_session_hash_stays_importable_from_samples():
    """`core/scope.py` owns the algorithm; the old import path keeps working."""
    from astrbot_plugin_dynamics_learning.core.samples import session_hash as from_samples

    assert from_samples is session_hash


# ---- what the host's group_id may and may not do ------------------------

def test_same_group_id_across_two_platforms_is_not_merged():
    """The host reports one group id; the contract still proves two sessions.

    ChatDynamics' own tests use exactly this pair, so merging on `group_id`
    would pool two unrelated conversations into a single profile.
    """
    left = "platform-a:GroupMessage:room"
    right = "platform-b:GroupMessage:room"
    meta = {
        left: {"session_key": left, "umo": left, "group_id": "room"},
        right: {"session_key": right, "umo": right, "group_id": "room"},
    }

    first = resolve_scope(left, meta[left])
    second = resolve_scope(right, meta[right])

    assert first.group_hint_hash == second.group_hint_hash      # the host says "same group"
    assert first.scope_hash != second.scope_hash                # and we still refuse to merge
    assert first.scope_hash != first.group_hint_hash

    samples = build_dataset([(left, _topic_record("m1")), (right, _topic_record("m2"))],
                            session_meta=meta)
    assert {sample.scope_hash for sample in samples} == {first.scope_hash, second.scope_hash}


def test_group_id_that_is_only_the_session_key_is_not_a_hint():
    """The contract writes the session key here when the event had no group id."""
    scope = resolve_scope(KEY, {"umo": KEY, "group_id": KEY})

    assert scope.group_hint_hash == ""


def test_umo_mismatch_is_reported_but_never_decides_identity():
    """The host's restore path treats this as a violation; so do we — visibly."""
    scope = resolve_scope(KEY, {"umo": "other:GroupMessage:1"})

    assert scope.scope_source == SCOPE_SOURCE_SESSION_UMO_MISMATCH
    assert scope.scope_hash == session_hash(KEY)


# ---- plumbing -----------------------------------------------------------

def test_build_dataset_records_which_host_facts_backed_the_scope():
    plain = build_dataset([(KEY, _topic_record())])[0]
    confirmed = build_dataset([(KEY, _topic_record())],
                              session_meta={KEY: {"session_key": KEY, "umo": KEY}})[0]

    assert plain.scope_source == SCOPE_SOURCE_SESSION
    assert confirmed.scope_source == SCOPE_SOURCE_SESSION_UMO_EQUAL
    assert plain.scope_hash == confirmed.scope_hash


def test_every_built_sample_keeps_scope_equal_to_its_session():
    rows = [{"session_key": f"umo:group:{index}", "umo": f"umo:group:{index}",
             "group_id": str(index)} for index in range(3)]
    meta = {row["session_key"]: row for row in rows}
    records = [(row["session_key"], _topic_record(f"m{index}"))
               for index, row in enumerate(rows)]

    samples = build_dataset(records, session_meta=meta)

    assert samples
    assert all(sample.scope_hash == sample.session_hash for sample in samples)
    assert all(sample.scope_source == SCOPE_SOURCE_SESSION_UMO_EQUAL for sample in samples)


def test_samples_from_annotation_accepts_an_explicit_scope():
    scope = LearningScope(scope_hash=session_hash(KEY), scope_source=SCOPE_SOURCE_SESSION,
                          group_hint_hash="h")
    sample = samples_from_annotation(_topic_record(), KEY, scope=scope)[0]

    assert sample.scope_hash == session_hash(KEY)
    assert sample.group_hint_hash == "h"
    restored = LearningSample.from_dict(sample.as_dict())
    assert restored is not None
    assert (restored.scope_hash, restored.scope_source, restored.group_hint_hash) == \
        (sample.scope_hash, sample.scope_source, sample.group_hint_hash)


def test_sample_schema_version_is_persisted_not_just_declared():
    sample = build_dataset([(KEY, _topic_record())])[0]

    payload = sample.as_dict()

    assert SAMPLE_SCHEMA_VERSION == 3
    assert payload["sample_schema_version"] == SAMPLE_SCHEMA_VERSION
    assert payload["scope_hash"] == payload["session_hash"]


def test_the_label_is_shorter_than_the_identity_and_never_used_as_one():
    scope = resolve_scope(KEY, {})

    assert LearningScope(scope_hash=scope.scope_hash).label == scope_label(scope.scope_hash)
    assert len(scope.label) == 12
    assert scope.label != scope.scope_hash
    assert scope.scope_hash.startswith(scope.label)
