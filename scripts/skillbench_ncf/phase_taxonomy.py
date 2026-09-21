#!/usr/bin/env python3
"""Frozen, task-independent phase taxonomy shared by V8 data and runtime.

The classifier only reads the visible query text and structural metadata.  It
must never read SkillsBench gold labels, trajectories, rewards, or task IDs.
"""

from __future__ import annotations

import re
from typing import Any


TAXONOMY_VERSION = "skillsbench_generic_operation_phase_v2"
PHASES = ("discover", "prepare", "execute", "verify", "deliver", "workflow")

# Order is not precedence: the earliest operative expression wins.  Status and
# delivery objects receive explicit semantic overrides below.
PHASE_PATTERNS = {
    "verify": (
        r"\bverify\b", r"\bvalidate\b", r"\bcheck\b", r"\bstatus\b",
        r"\bconfirm\b", r"\binspect(?:ion)?\b", r"\bmonitor\b",
        r"\baudit\b", r"\bdiagnos(?:e|is|tic)\b", r"\bhealth\b",
    ),
    "deliver": (
        r"\bdownload\b", r"\bexport\b", r"\bpublish\b", r"\bshare\b",
        r"\bdeliver\b", r"\bnotify\b", r"\bdispatch\b", r"\bsave\b",
    ),
    "discover": (
        r"\bsearch\b", r"\bfind\b", r"\blist\b", r"\bretrieve\b",
        r"\bfetch\b", r"\bget\b", r"\bread\b", r"\bquery\b",
        r"\blook up\b", r"\blookup\b", r"\bbrowse\b", r"\bview\b",
        r"\benumerate\b", r"\blocate\b", r"\bidentify\b", r"\bshow\b",
    ),
    "prepare": (
        r"\bsetup\b", r"\bset up\b", r"\bconfigure\b",
        r"\bauthenticate\b", r"\bconnect\b", r"\binitialize\b",
        r"\bupload\b", r"\bcreate\b", r"\badd\b", r"\bregister\b",
        r"\bschedule\b", r"\bprepare\b", r"\bselect\b", r"\bchoose\b",
    ),
    "execute": (
        r"\bupdate\b", r"\bdelete\b", r"\bremove\b", r"\bsend\b",
        r"\brun\b", r"\banalyze\b", r"\bgenerate\b", r"\bconvert\b",
        r"\btransform\b", r"\bcalculate\b", r"\bwrite\b", r"\bedit\b",
        r"\bprocess\b", r"\bexecute\b", r"\bmanage\b", r"\bmark\b",
        r"\bcancel\b", r"\bstandardize\b", r"\brecognize\b",
        r"\brecord\b", r"\bset\b", r"\bdetach\b", r"\bgrant\b",
        r"\bextract\b", r"\bcompare\b", r"\bautomate\b", r"\btrain\b",
        r"\bevaluate\b", r"\bdetect\b", r"\bbuild\b", r"\bfill\b",
        r"\bmove\b", r"\bplace\b", r"\boperate\b", r"\bscan\b",
        r"\breceive\b", r"\bupsert\b", r"\bmerge\b", r"\bsubmit\b",
    ),
}


def classify_phase(row: dict[str, Any]) -> tuple[str, str, str]:
    """Return ``(phase, basis, confidence)`` under the frozen V2 policy."""

    if len(row.get("positive_skill_ids", [])) > 1 or row.get("stratum") == "workflow":
        return "workflow", "explicit_joint_positive_query", "high"

    text = str(row.get("query_text", "")).lower().strip()
    matches: list[tuple[int, int, str, str]] = []
    for phase, patterns in PHASE_PATTERNS.items():
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                matches.append((match.start(), match.end(), phase, pattern))

    # A request to retrieve/check status is verification even when its first
    # verb is a generic retrieval verb.
    semantic_status = re.search(
        r"\b(?:status|progress|health|quota usage|remaining credits?|plan limits?|"
        r"resource usage|test results?|run details?)\b",
        text,
    )
    if semantic_status:
        return "verify", "semantic_status_object_override", "high"

    semantic_delivery = re.search(r"\b(?:download|export)\b", text)
    if semantic_delivery:
        return "deliver", "semantic_delivery_override", "high"

    if not matches:
        return "execute", "no_frozen_lexical_match_default_core_operation", "low"

    matches.sort(key=lambda item: (item[0], item[1], item[2]))
    start, _, phase, pattern = matches[0]
    # A verb in the opening phrase is a decisive action.  Later words such as
    # "updated at" or "test suite" do not make that action ambiguous.
    if start <= 24:
        return phase, f"opening_operative_rule:{pattern}", "high"
    return phase, f"embedded_operative_rule:{pattern}", "medium"
