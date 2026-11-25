"""Lean, deterministic model-facing payload projections for ANALYZE and REVIEW.

Never mutates or drops persisted SQLite fields; projects compact views strictly
at LLM-invocation time to eliminate repeated per-event boilerplate (such as
identical persona_summary, policy.context, character_scope, and bookkeeping
metadata) while preserving 100% of verbatim text, IDs, attribution, and review
context needed by the model.
"""

from __future__ import annotations

import json

from .contract import PERSONALITY_POLICY, digest

DEFER_FINGERPRINT = "review_defer_fingerprint"
ANALYZE_PAYLOAD_BUDGET = 22_000
ANALYZE_CURRENT_BUDGET = 8_000
ANALYZE_SINGLE_EVENT_MAX = 112_000
REVIEW_BATCH_PAYLOAD_BUDGET = 24_000
_REPLY_TEXT_CAP = 800


def encode_prompt_payload(value) -> str:
    """Serialize LLM prompt payloads preserving static-first dict key order (sort_keys=False)."""
    return json.dumps(value, ensure_ascii=False, sort_keys=False, separators=(",", ":"), allow_nan=False)


def utf8_len(value) -> int:
    """Return UTF-8 byte length of a string or compact JSON-encoded value."""
    text = value if isinstance(value, str) else encode_prompt_payload(value)
    return len(text.encode("utf-8"))


def _compact_interaction(interaction):
    if not isinstance(interaction, dict):
        return None
    out = {}
    for key in ("communication_intent", "expression_mode", "turn_update", "speaker_stance"):
        val = interaction.get(key)
        choice = val.get("choice") if isinstance(val, dict) else val
        if isinstance(choice, str) and choice and choice != "unclear":
            out[key] = choice
    reply_focus = interaction.get("reply_focus")
    if isinstance(reply_focus, list) and reply_focus:
        out["reply_focus"] = reply_focus[:3]
    return out or None


def extract_baseline_threshold(events) -> float | None:
    """Extract per-group baseline_threshold for inclusion in user JSON payloads (never system_prompt)."""
    for ev in events or ():
        if isinstance(ev, dict) and ev.get("baseline_threshold") is not None:
            try:
                return float(ev["baseline_threshold"])
            except (TypeError, ValueError):
                continue
    return None


def extract_character_context(events, *, scope="", persona="", include_scope=False):
    """Hoist static persona-level character summary once.

    Omits per-group `scope` and `baseline_threshold` so multiple groups sharing the
    same persona produce 100% byte-identical system_prompt prefixes and implicit-cache
    anchors; both `scope` and `baseline_threshold` remain in user JSON payloads.
    """
    summary = ""
    for ev in events or ():
        if not isinstance(ev, dict):
            continue
        ctx = ev.get("context")
        if (
            not summary
            and isinstance(ctx, dict)
            and isinstance(ctx.get("persona_summary"), str)
            and ctx["persona_summary"].strip()
        ):
            summary = ctx["persona_summary"].strip()
            break
    if not summary:
        return None
    out = {"persona_summary": summary}
    if persona:
        out["persona"] = persona
    if include_scope and scope:
        out["scope"] = scope
    return out


def format_system_prompt(base_prompt, character_context=None):
    """Prepend deterministic persona-level character_context + policy before task rules.

    Placing the shared persona prefix first enables cross-group AND cross-task
    (ANALYZE / UNIFIED_REVIEW / ANALYZE_SCENES / ANALYZE_STYLE) prefix cache sharing.
    """
    if not character_context:
        return base_prompt
    return (
        "[character_context]\n"
        + encode_prompt_payload(character_context)
        + "\n\n[personality_policy]\n"
        + PERSONALITY_POLICY
        + "\n\n"
        + base_prompt
    )


def project_event_for_model(event, *, include_scope=True):
    """Project an event or native source into a lean model-facing dict.

    Preserves id, kind, member, message_id, turn_id, verbatim text, timestamp,
    concise decision/outcome/reply/policy metadata, and allowlisted context
    (omitting hoisted persona_summary, repeated policy.context, character_scope,
    identity_tokens, and internal bookkeeping hashes).
    """
    if not isinstance(event, dict):
        return event
    out = {}
    for key in ("id", "kind", "member", "message_id", "turn_id", "text", "timestamp"):
        if key in event and event[key] is not None:
            out[key] = event[key]
    if include_scope and event.get("scope"):
        out["scope"] = event["scope"]
    if event.get("mandatory") is True:
        out["mandatory"] = True
    for key in ("actor", "provenance"):
        if event.get(key):
            out[key] = event[key]

    reply = event.get("reply")
    if isinstance(reply, dict) and reply:
        r_out = {}
        if "delivered" in reply:
            r_out["delivered"] = reply["delivered"]
        if reply.get("message_id"):
            r_out["message_id"] = reply["message_id"]
        if reply.get("text"):
            text = str(reply["text"])
            r_out["text"] = text if len(text) <= _REPLY_TEXT_CAP else text[: _REPLY_TEXT_CAP - 3] + "..."
        if r_out:
            out["reply"] = r_out

    decision = event.get("decision")
    if isinstance(decision, dict) and decision:
        d_out = {k: decision[k] for k in ("action", "length", "state", "reason") if decision.get(k) not in (None, "")}
        answers = decision.get("answers")
        if isinstance(answers, dict) and isinstance(answers.get("join"), dict) and "noul" in answers["join"]:
            d_out["join"] = answers["join"]["noul"]
        if d_out:
            out["decision"] = d_out

    outcome = event.get("outcome")
    if isinstance(outcome, dict) and outcome:
        o_out = {
            k: outcome[k]
            for k in ("delivered", "final_outcome", "stage", "suppression_reason")
            if outcome.get(k) not in (None, "")
        }
        if o_out:
            out["outcome"] = o_out

    policy = event.get("policy")
    if isinstance(policy, dict) and policy:
        p_out = {}
        if policy.get("threshold") is not None:
            p_out["threshold"] = policy["threshold"]
        if policy.get("versions"):
            p_out["versions"] = policy["versions"]
        if p_out:
            out["policy"] = p_out

    ctx = event.get("context")
    if isinstance(ctx, dict) and ctx:
        c_out = {}
        if ctx.get("source_verified") is True:
            c_out["source_verified"] = True
        for key in ("reply_to", "previous_id", "wake_kind"):
            if ctx.get(key):
                c_out[key] = ctx[key]
        reply_to_ids = [x for x in (ctx.get("reply_to_ids") or ()) if x]
        if reply_to_ids:
            c_out["reply_to_ids"] = reply_to_ids
        if ctx.get("mentions"):
            c_out["mentions"] = ctx["mentions"]
        mids = ctx.get("message_ids")
        if isinstance(mids, list) and mids and mids != [event.get("message_id")]:
            c_out["message_ids"] = mids
        if ctx.get("scene_feedback"):
            c_out["scene_feedback"] = ctx["scene_feedback"]
        inter = _compact_interaction(ctx.get("interaction"))
        if inter:
            c_out["interaction"] = inter
        if c_out:
            out["context"] = c_out
    return out


def project_current_entry_for_model(entry):
    """Project an active entry into a stable, timestamp-free view for ANALYZE."""
    return {k: entry[k] for k in ("id", "category", "key", "member", "value", "revision") if k in entry}


def project_before_for_review(before):
    if not isinstance(before, dict):
        return None
    return {k: before[k] for k in ("value", "revision", "active") if k in before}


def project_proposal_for_review(
    proposal,
    *,
    include_scope_persona=True,
    include_evidence=True,
    evidence_ids=None,
    target_material=...,
):
    """Project a proposal into a lean model-facing view for REVIEW."""
    out = {
        k: proposal[k]
        for k in ("id", "category", "key", "member", "value", "reason", "mode", "source", "days", "base_revision")
        if k in proposal
    }
    if include_scope_persona:
        if "scope" in proposal:
            out["scope"] = proposal["scope"]
        if "persona" in proposal:
            out["persona"] = proposal["persona"]
    out["before"] = project_before_for_review(proposal.get("before"))
    if include_evidence:
        out["evidence"] = [project_event_for_model(ev, include_scope=False) for ev in (proposal.get("evidence") or ())]
    elif evidence_ids is not None:
        out["evidence_ids"] = evidence_ids
    if target_material is not ...:
        out["target_material"] = target_material
    return out


def build_analyze_payload(
    scope,
    persona,
    *,
    current,
    events,
    member_sources,
    character_context=None,
    baseline_threshold=None,
):
    """Assemble ANALYZE payload in cache-friendly static-before-volatile key order."""
    payload = {}
    if character_context:
        payload["character_context"] = character_context
    payload["current"] = current
    if baseline_threshold is not None:
        payload["baseline_threshold"] = float(baseline_threshold)
    payload["events"] = events
    payload["member_sources"] = member_sources
    return payload


def build_single_review_payload(proposal, *, target_material=..., include_character_context=False):
    """Assemble lean single-proposal REVIEW payload.

    By default `character_context` is placed in `system_prompt` by the caller so
    it is not duplicated in user JSON; per-group `baseline_threshold` lives here.
    """
    raw_evidence = proposal.get("evidence") or ()
    payload = {}
    if include_character_context:
        char_ctx = extract_character_context(
            raw_evidence, scope=proposal.get("scope", ""), persona=proposal.get("persona", "")
        )
        if char_ctx:
            payload["character_context"] = char_ctx
    payload["proposal"] = project_proposal_for_review(
        proposal, include_scope_persona=True, include_evidence=True
    )
    threshold = extract_baseline_threshold(raw_evidence)
    if threshold is not None:
        payload["baseline_threshold"] = threshold
    if target_material is not ...:
        payload["target_material"] = target_material
    return payload


def build_batch_review_payload(scope, persona, entries, *, resolve_target=None, include_character_context=False):
    """Assemble lean batched REVIEW payload with deduplicated evidence_pool.

    Each entry in `entries` may be a `(proposal, target_material_or_ellipsis)` tuple
    or a bare proposal dict. By default `character_context` lives in `system_prompt`.
    """
    evidence_pool = {}
    all_raw_evidence = []
    compact_proposals = []
    for entry in entries:
        if isinstance(entry, tuple):
            p, target_material = entry
        else:
            p, target_material = entry, ...
        if target_material is ... and resolve_target is not None and p.get("category") in {
            "expression_adjustment",
            "scene_adjustment",
        }:
            target_material = resolve_target(p)
        ev_ids = []
        for ev in p.get("evidence") or ():
            all_raw_evidence.append(ev)
            ev_id = str(ev.get("id") or "") if isinstance(ev, dict) else ""
            if not ev_id:
                ev_id = digest(ev)
            if ev_id not in evidence_pool:
                evidence_pool[ev_id] = project_event_for_model(ev, include_scope=False)
            ev_ids.append(ev_id)
        compact_proposals.append(
            project_proposal_for_review(
                p,
                include_scope_persona=False,
                include_evidence=False,
                evidence_ids=ev_ids,
                target_material=target_material,
            )
        )
    payload = {}
    if include_character_context:
        char_ctx = extract_character_context(all_raw_evidence, scope=scope, persona=persona)
        if char_ctx:
            payload["character_context"] = char_ctx
    payload["scope"] = scope
    payload["persona"] = persona
    threshold = extract_baseline_threshold(all_raw_evidence)
    if threshold is not None:
        payload["baseline_threshold"] = threshold
    payload["evidence_pool"] = {k: evidence_pool[k] for k in sorted(evidence_pool)}
    payload["proposals"] = compact_proposals
    return payload
