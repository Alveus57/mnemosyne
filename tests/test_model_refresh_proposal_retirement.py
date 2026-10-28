"""Resolved model-refresh proposals must stop being recall candidates.

Sleep writes every model-refresh proposal as a ``working_memory`` row with
``source='sleep_model_refresh_proposal'`` and sets ``consolidated_at``
immediately, so consolidation never retires it. The resolution paths only
flipped ``metadata["status"]``; ``superseded_by`` stayed NULL, and recall
selects on ``superseded_by IS NULL``. The result: every applied or rejected
proposal was recalled forever -- on a real bank 998 of 6,493 working rows
(15%) were resolved proposals still competing for recall slots.

These tests pin both halves of the contract: a resolved proposal is retired
from recall while the row survives for audit, and a pending proposal is
still surfaced (the negative control that keeps the first assertion from
passing vacuously).
"""

from __future__ import annotations

import json

from mnemosyne.core.beam import BeamMemory
from mnemosyne.core import model_refresh


def _proposal(token: str, **overrides):
    base = {
        "category": "model:user",
        "name": f"retirement_{token}",
        "body": f"Durable preference about {token} reporting.",
        "confidence": 0.95,
        "evidence_ids": [f"wm-{token}-a", f"wm-{token}-b"],
        "action": "update",
        "reason": "Repeated durable preference.",
    }
    base.update(overrides)
    return base


def _store_proposal(beam, token: str):
    proposal = _proposal(token)
    metadata = model_refresh.prepare_proposal_metadata(
        proposal, source_wm_ids=proposal["evidence_ids"]
    )
    proposal_id = beam.remember(
        model_refresh.proposal_to_memory_content(proposal),
        source=model_refresh.PROPOSAL_SOURCE,
        metadata=metadata,
    )
    assert proposal_id, "beam.remember must return the stored proposal id"
    return proposal_id


def _recalled_ids(beam, token: str):
    return {
        row.get("id")
        for row in beam.recall(f"retirement_{token} reporting preference", top_k=10)
    }


def _superseded_by(beam, proposal_id: str):
    row = beam.conn.execute(
        "SELECT superseded_by FROM working_memory WHERE id = ?", (proposal_id,)
    ).fetchone()
    assert row is not None, "proposal row must survive for audit"
    return row["superseded_by"]


class TestResolvedProposalRetirement:

    def test_applied_proposal_is_retired_from_recall(self, tmp_path):
        beam = BeamMemory(session_id="retire", db_path=tmp_path / "mnemo.db")
        proposal_id = _store_proposal(beam, "applied")

        # Negative control: while pending, the proposal is a recall candidate.
        assert proposal_id in _recalled_ids(beam, "applied")
        assert _superseded_by(beam, proposal_id) is None

        model_refresh.apply_model_refresh_proposal(beam, proposal_id)

        assert _superseded_by(beam, proposal_id) == "model_refresh:applied"
        assert proposal_id not in _recalled_ids(beam, "applied")

    def test_rejected_proposal_is_retired_from_recall(self, tmp_path):
        beam = BeamMemory(session_id="retire", db_path=tmp_path / "mnemo.db")
        proposal_id = _store_proposal(beam, "rejected")

        assert proposal_id in _recalled_ids(beam, "rejected")

        model_refresh.reject_model_refresh_proposal(beam, proposal_id, reason="stale")

        assert _superseded_by(beam, proposal_id) == "model_refresh:rejected"
        assert proposal_id not in _recalled_ids(beam, "rejected")

    def test_row_and_metadata_are_kept_for_audit(self, tmp_path):
        beam = BeamMemory(session_id="retire", db_path=tmp_path / "mnemo.db")
        proposal_id = _store_proposal(beam, "audit")

        model_refresh.reject_model_refresh_proposal(beam, proposal_id, reason="noise")

        row = beam.conn.execute(
            "SELECT content, metadata_json, source FROM working_memory WHERE id = ?",
            (proposal_id,),
        ).fetchone()
        assert row is not None
        assert row["source"] == model_refresh.PROPOSAL_SOURCE
        assert row["content"]
        assert json.loads(row["metadata_json"])["status"] == "rejected"

    def test_pending_proposal_stays_recall_eligible(self, tmp_path):
        """A proposal nobody resolved must not be hidden by this fix."""
        beam = BeamMemory(session_id="retire", db_path=tmp_path / "mnemo.db")
        proposal_id = _store_proposal(beam, "pending")

        assert _superseded_by(beam, proposal_id) is None
        assert proposal_id in _recalled_ids(beam, "pending")

    def test_retirement_is_idempotent(self, tmp_path):
        beam = BeamMemory(session_id="retire", db_path=tmp_path / "mnemo.db")
        proposal_id = _store_proposal(beam, "twice")

        model_refresh.reject_model_refresh_proposal(beam, proposal_id, reason="first")
        model_refresh.reject_model_refresh_proposal(beam, proposal_id, reason="second")

        assert _superseded_by(beam, proposal_id) == "model_refresh:rejected"
