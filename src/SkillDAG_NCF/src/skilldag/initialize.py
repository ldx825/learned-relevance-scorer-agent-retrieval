"""Cold-start graph initialization.

Two-view candidate seeding:
  * `e_self`  — embedding of (description + body preview). Captures
                "what this skill DOES". Anchors `similar_to / specializes`
                (same-meaning relations).
  * `e_needs` — embedding of an LLM-imagined common-prerequisites sentence
                ("imagine 2-3 tasks using this skill; what must be true
                BEFORE invoking it?"). Captures "what this skill REQUIRES".
                Anchors `depends_on / composes_with` (cross-meaning chain
                relations) which top-K cosine on `e_self` systematically
                misses.

Pipeline:
  1. Embed each skill's (description + body preview) → `e_self`.
  2. Per-skill LLM call: imagined-tasks → common prerequisites text →
     embed → `e_needs`. Skills with "self-contained" output get no
     `e_needs` and contribute no second-pass candidates.
  3. Rank candidate pairs from BOTH views:
       a. cos(e_self_i, e_self_j)  — adaptive threshold (mean + 1σ,
          clipped [0.35, 0.75]) + top-K=12 + floor=3 per anchor.
       b. cos(e_needs_i, e_self_j) — same ranking strategy.
     Union both pair sets.
  4. Bucket pairs by lower-index anchor; one LLM call per anchor
     classifies all of its candidates into one of
     {similar_to, specializes, depends_on, composes_with, none}.
  5. Return edge dicts with origin="cold_start".

Environment variables (all required when initialization is enabled):
  SKILLDAG_EMBEDDING_API_KEY   — API key for embedding endpoint
  SKILLDAG_LLM_API_KEY         — API key for chat LLM endpoint

Optional (defaults filled in):
  SKILLDAG_EMBEDDING_BASE      default "https://api.openai.com/v1"
  SKILLDAG_EMBEDDING_MODEL     default "text-embedding-3-large"
  SKILLDAG_INITIALIZE_EMBEDDINGS_CACHE
                                optional path to existing e_self embeddings
                                cache; when set, every skill must hit cache
  SKILLDAG_LLM_BASE            default "https://api.openai.com/v1"
  SKILLDAG_LLM_MODEL           default "gpt-5-nano"
  SKILLDAG_LLM_CONCURRENCY     default 10
  SKILLDAG_LLM_TIMEOUT_S       default 30
"""

from __future__ import annotations

import json
import logging
import os
import re
import socket
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class _HTTPTimeout(Exception):
    """Stdlib equivalent of requests.exceptions.Timeout."""


def _http_post_json(
    url: str,
    headers: dict[str, str],
    payload: dict,
    timeout: int,
) -> tuple[int, str]:
    """POST JSON via stdlib urllib. Returns (status, body_text).

    Raises ``_HTTPTimeout`` on socket timeout, ``RuntimeError`` on any other
    transport-level failure. Caller handles status-code retries.
    """
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={**headers, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8")
    except (socket.timeout, TimeoutError) as e:
        raise _HTTPTimeout(str(e)) from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"network error: {e}") from e


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_EMBEDDING_BASE = "https://api.openai.com/v1"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
INITIALIZE_EMBEDDINGS_CACHE_ENV = "SKILLDAG_INITIALIZE_EMBEDDINGS_CACHE"
DEFAULT_LLM_BASE = "https://api.openai.com/v1"
DEFAULT_LLM_MODEL = "gpt-5-nano"

SIM_THRESHOLD_MIN = 0.35
SIM_THRESHOLD_MAX = 0.75
TOP_K_MAX = 12
TOP_K_MIN = 3     # floor: guarantee each skill has at least this many candidate neighbors
BODY_PREVIEW_CHARS = 1200
EMBED_BATCH_SIZE = 100
DEFAULT_LLM_TIMEOUT_S = 60
LLM_MAX_RETRIES = 3
LLM_RETRY_BACKOFF = [2, 5, 10]  # seconds before retry 1, 2, 3
# Cold-start emits only positive structural edge types. `conflicts_with` is
# reserved for online execution evidence because static SKILL.md text cannot
# establish that co-use hurts task success.
VALID_COLD_START_TYPES = {
    "similar_to",
    "specializes",
    "depends_on",
    "composes_with",
}


INITIALIZE_PROMPT = """Classify the relationship between the anchor skill A and EACH candidate skill (B1, B2, ...) based on static text in their SKILL.md (description + body preview).

Valid types:
- similar_to     : A and Bi are alternative ways to solve substantially the SAME
                   subtask under similar prerequisites and with substitutable
                   outputs. Selecting BOTH is redundant, not harmful; one can
                   replace the other. Use this for redundancy / interchangeability
                   (e.g., two web-search backends, two PDF text extractors,
                   multiple TTS APIs — "pick one"). If one is clearly broader
                   or narrower, use `specializes` instead.

- specializes    : one is a narrower version of the other; the broader one's
                   capability is a SUPERSET of the narrower's. Direction:
                   source = narrower, target = broader. Example: "tesseract-ocr"
                   specializes "general-vision-analysis" because vision-analysis
                   does OCR *and* more.

- depends_on     : A REQUIRES B's setup or output to be in place before A can
                   succeed. The requirement must be stated or directly
                   derivable from A's "Prerequisites" / "Inputs" / "Verify
                   Context" / "Process" section in SKILL.md. Direction:
                   source = A (the dependent skill), target = B (the
                   prerequisite skill). Example: an "object-cooler" skill
                   whose Prerequisites include "Object in Hand" depends_on
                   an "object-picker" skill — cooler needs picker's output
                   first. Not symmetric: cooler depends_on picker, NOT the
                   reverse.

- composes_with  : A and B chain together in a typical workflow — A's output
                   directly feeds B's input, or both are explicit steps of a
                   stated multi-step pipeline. Use ONLY when A's "Next Steps"
                   / "Output" / workflow narrative names B (or vice versa) by
                   role, AND there is no asymmetric prerequisite (otherwise
                   depends_on is more informative). Direction is symmetric;
                   put source = {a_id}.

- none           : no static structural relationship. Skills are complementary
                   but neither requires nor produces the other, OR they are
                   unrelated, OR a possible conflict / dependency is implied
                   but not textually evidenced. When in doubt, prefer none
                   over a weak label.

Per-pair single label:
- Each pair (A, Bi) gets EXACTLY ONE type. Pick the most structurally
  informative one that fits.
- Tie-break: depends_on > composes_with > specializes > similar_to > none.
  (depends_on wins over specializes if both fit.)

For `specializes` / `similar_to` / `depends_on`, if the description or body
preview clearly establishes the relationship, LABEL IT — do not default to
"none".

Treat each candidate Bi INDEPENDENTLY — presence of other candidates in this
batch must not influence the judgement for Bi.

Anchor Skill A:
  id: {a_id}
  description: {a_desc}
  body_preview: {a_body}

Candidate Skills (classify A vs each of these):
{candidates_block}

Output exactly one JSON array of length {n_candidates} (no markdown, no prose).
Each element is:
{{"target_ref": "B<number>", "type": "similar_to|specializes|depends_on|composes_with|none", "source_ref": "A|B<number>", "reason": "<<=25 words>"}}

Field rules:
- `target_ref`: MUST be the candidate label, exactly one of B1..B{n_candidates}
  and never "A". All {n_candidates} candidate labels must appear exactly once,
  in the order listed above. Do not copy candidate ids into `target_ref`.
- `source_ref`: identifies direction within the pair (A, Bi).
    * For `specializes`: source_ref = the NARROWER skill (A or the same B label)
      whose capability is a
      strict subset of the other).
    * For `depends_on`: source_ref = the DEPENDENT skill (A or the same B label)
      whose
      prerequisites name the other).
    * For `similar_to` / `composes_with` / `none`: put "A"
      (direction is symmetric or unused).
"""


NEEDS_PROMPT = """You are seeding a skill dependency graph. Your job is to identify what other capabilities or states must be in place BEFORE this skill can be invoked.

Skill:
  id: {sid}
  description: {description}

Imagine 2-3 concrete tasks (≤1 sentence each) in which an agent invokes
this skill as part of a longer action chain. For each task, describe what
the agent must have accomplished BEFORE invoking this skill — held
artifacts, prior outputs, environment state, installed tools, file
formats, prior capability invocations, anything that must be true.

Then summarise the common prerequisites across the imagined tasks in ONE
sentence. If no clear prerequisite holds across the imagined tasks (the
skill is self-contained or accepts raw input), output exactly the literal
string "self-contained" for the common_prerequisites field.

Output exactly one JSON object (no markdown, no prose):
{{"imagined_tasks": ["...", "...", "..."], "common_prerequisites": "..."}}
"""


class InitializationError(Exception):
    pass


def _require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise InitializationError(
            f"Initialization requires {name}; set it and retry. "
            f"Or set SKILLDAG_INITIALIZE_ENABLED=0 to skip initialization and fail load() instead."
        )
    return val


# ---------------------------------------------------------------------------
# SKILL.md body preview
# ---------------------------------------------------------------------------

def _read_body_preview(skill_path: str, max_chars: int = BODY_PREVIEW_CHARS) -> str:
    """Return first max_chars of SKILL.md body (after frontmatter, skipping headings)."""
    try:
        md_path = Path(skill_path) / "SKILL.md"
        if not md_path.exists():
            return ""
        text = md_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

    lines = text.splitlines()
    # Strip YAML frontmatter if present
    if lines and lines[0].strip() == "---":
        try:
            end = lines.index("---", 1)
            lines = lines[end + 1:]
        except ValueError:
            pass

    # Drop headings and blank lines for a denser preview
    body_lines = [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]
    body = " ".join(body_lines)
    # Collapse repeated whitespace
    body = re.sub(r"\s+", " ", body)
    return body[:max_chars]


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def _embed_batch(texts: list[str]) -> list[list[float]]:
    """Call embedding API in batches; return list aligned with input order.

    Per-batch retries on Timeout / 429 / 5xx with exponential backoff.
    """
    import time

    base = os.environ.get("SKILLDAG_EMBEDDING_BASE", DEFAULT_EMBEDDING_BASE).rstrip("/")
    model = os.environ.get("SKILLDAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    api_key = _require_env("SKILLDAG_EMBEDDING_API_KEY")
    timeout = int(os.environ.get("SKILLDAG_EMBEDDING_TIMEOUT_S", "120"))

    out: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i:i + EMBED_BATCH_SIZE]
        data = None
        last_err: Exception | None = None
        for attempt in range(LLM_MAX_RETRIES + 1):
            try:
                status, body = _http_post_json(
                    f"{base}/embeddings",
                    headers={"Authorization": f"Bearer {api_key}"},
                    payload={"model": model, "input": batch},
                    timeout=timeout,
                )
                if status in (502, 503, 504, 429):
                    last_err = RuntimeError(f"HTTP {status}: {body[:120]}")
                    if attempt < LLM_MAX_RETRIES:
                        time.sleep(LLM_RETRY_BACKOFF[attempt])
                        continue
                    raise last_err
                if status >= 400:
                    raise RuntimeError(f"HTTP {status}: {body[:200]}")
                data = json.loads(body)["data"]
                break
            except _HTTPTimeout as e:
                last_err = e
                if attempt < LLM_MAX_RETRIES:
                    logger.warning("embed batch %d/%d timeout (attempt %d); retrying",
                                   i // EMBED_BATCH_SIZE + 1,
                                   (len(texts) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE,
                                   attempt + 1)
                    time.sleep(LLM_RETRY_BACKOFF[attempt])
                    continue
                raise
        if data is None:
            raise last_err or RuntimeError("embed batch failed with no error captured")
        # Sort by index to be safe (API spec says they come in order, but let's not trust)
        data_sorted = sorted(data, key=lambda x: x["index"])
        out.extend(item["embedding"] for item in data_sorted)
    return out


# ---------------------------------------------------------------------------
# Similarity + candidate generation (pure Python; fine for N ≤ few thousand)
# ---------------------------------------------------------------------------

def _normalize(vec: list[float]) -> list[float]:
    s = sum(x * x for x in vec) ** 0.5 or 1.0
    return [x / s for x in vec]


def _cos_matrix(embeddings: list[list[float]]) -> list[list[float]]:
    """Symmetric cos-sim matrix; diagonal left as 0 (we skip self anyway)."""
    E = [_normalize(v) for v in embeddings]
    n = len(E)
    dim = len(E[0]) if E else 0
    M = [[0.0] * n for _ in range(n)]
    for i in range(n):
        ei = E[i]
        for j in range(i + 1, n):
            ej = E[j]
            s = 0.0
            for k in range(dim):
                s += ei[k] * ej[k]
            M[i][j] = s
            M[j][i] = s
    return M


def _rank_candidates(sim: list[list[float]]) -> set[frozenset]:
    """For each row, pick neighbors: adaptive threshold (mean+1σ clipped) with top-K cap.

    Floor: each skill gets at least TOP_K_MIN neighbors even if threshold excludes them —
    this avoids empty candidate sets for small N where std is compressed.
    """
    n = len(sim)
    pairs: set[frozenset] = set()
    for i in range(n):
        row_full = [(j, sim[i][j]) for j in range(n) if j != i]
        if not row_full:
            continue
        sims = [s for _, s in row_full]
        mu = sum(sims) / len(sims)
        var = sum((s - mu) ** 2 for s in sims) / len(sims)
        std = var ** 0.5
        thresh = max(SIM_THRESHOLD_MIN, min(SIM_THRESHOLD_MAX, mu + std))
        ranked = sorted(row_full, key=lambda x: -x[1])
        above = [(j, s) for j, s in ranked if s > thresh][:TOP_K_MAX]
        # Floor: if adaptive filter returns < TOP_K_MIN, take top TOP_K_MIN by raw similarity
        if len(above) < TOP_K_MIN:
            above = ranked[:min(TOP_K_MIN, n - 1)]
        for j, _ in above:
            pairs.add(frozenset({i, j}))
    return pairs


def _rank_needs_candidates(
    e_needs: list[list[float] | None],
    e_self: list[list[float]],
) -> set[frozenset]:
    """Asymmetric cosine: cos(anchor.e_needs, other.e_self).

    For each anchor i with non-None e_needs, score every other skill j by
    cos(e_needs[i], e_self[j]) and pick top-K above adaptive threshold,
    same as `_rank_candidates`. Returns frozensets so they merge cleanly
    with the e_self candidate set; the LLM classification step decides
    direction per pair.
    """
    n = len(e_self)
    pairs: set[frozenset] = set()
    e_self_norm = [_normalize(v) for v in e_self]

    for i, needs_vec in enumerate(e_needs):
        if needs_vec is None:
            continue
        nn = _normalize(needs_vec)
        row_full = []
        for j in range(n):
            if j == i:
                continue
            sj = e_self_norm[j]
            s = sum(a * b for a, b in zip(nn, sj))
            row_full.append((j, s))
        if not row_full:
            continue
        sims = [s for _, s in row_full]
        mu = sum(sims) / len(sims)
        var = sum((s - mu) ** 2 for s in sims) / len(sims)
        std = var ** 0.5
        thresh = max(SIM_THRESHOLD_MIN, min(SIM_THRESHOLD_MAX, mu + std))
        ranked = sorted(row_full, key=lambda x: -x[1])
        above = [(j, s) for j, s in ranked if s > thresh][:TOP_K_MAX]
        if len(above) < TOP_K_MIN:
            above = ranked[:min(TOP_K_MIN, n - 1)]
        for j, _ in above:
            pairs.add(frozenset({i, j}))
    return pairs


# ---------------------------------------------------------------------------
# LLM pair classification
# ---------------------------------------------------------------------------

def _strip_json_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text).strip()
    return text


def _extract_imagined_needs(node: dict[str, str]) -> str:
    """Per-skill LLM call: imagined-tasks → common prerequisites sentence.

    Returns the `common_prerequisites` field from the LLM JSON output.
    Returns the literal string "self-contained" when the LLM finds no
    cross-task prerequisite. Returns "" on transport / parse failure
    (caller treats empty same as self-contained → no second-pass
    candidate contribution).
    """
    import time

    base = os.environ.get("SKILLDAG_LLM_BASE", DEFAULT_LLM_BASE).rstrip("/")
    model = os.environ.get("SKILLDAG_LLM_MODEL", DEFAULT_LLM_MODEL)
    api_key = _require_env("SKILLDAG_LLM_API_KEY")
    timeout = int(os.environ.get("SKILLDAG_LLM_TIMEOUT_S", str(DEFAULT_LLM_TIMEOUT_S)))

    prompt = NEEDS_PROMPT.format(
        sid=node["id"],
        description=node.get("description", ""),
    )

    reasoning_effort = os.environ.get("SKILLDAG_LLM_REASONING_EFFORT", "")
    payload: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    else:
        payload["temperature"] = 0.0

    sid = node["id"]
    content = None
    last_err: Exception | None = None

    for attempt in range(LLM_MAX_RETRIES + 1):
        try:
            status, body = _http_post_json(
                f"{base}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                payload=payload,
                timeout=timeout,
            )
            if status in (502, 503, 504, 429):
                last_err = RuntimeError(f"HTTP {status}: {body[:120]}")
                if attempt < LLM_MAX_RETRIES:
                    time.sleep(LLM_RETRY_BACKOFF[attempt])
                    continue
                raise last_err
            if status >= 400:
                raise RuntimeError(f"HTTP {status}: {body[:200]}")
            content = json.loads(body)["choices"][0]["message"]["content"]
            break
        except _HTTPTimeout as e:
            last_err = e
            if attempt < LLM_MAX_RETRIES:
                time.sleep(LLM_RETRY_BACKOFF[attempt])
                continue
            logger.warning("imagined_needs(%s) timeout: %s", sid, e)
            return ""
        except Exception as e:
            logger.warning("imagined_needs(%s) failed: %s", sid, e)
            return ""

    if content is None:
        logger.warning("imagined_needs(%s) all retries exhausted: %s", sid, last_err)
        return ""

    try:
        parsed = json.loads(_strip_json_fence(content))
    except Exception as e:
        logger.warning("imagined_needs(%s) JSON parse failed: %s | raw=%s",
                       sid, e, content[:200])
        return ""

    if not isinstance(parsed, dict):
        return ""
    cp = parsed.get("common_prerequisites", "")
    if not isinstance(cp, str):
        return ""
    return cp.strip()


def _classify_anchor_batch(
    anchor: dict[str, str],
    candidates: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Classify anchor vs every candidate in one LLM call.

    Returns a list of edge dicts {source, target, type, reason} — only non-"none"
    results. Invalid / missing entries are dropped with a warning. Returns [] on
    total failure (transport / parse).
    """
    import time

    if not candidates:
        return []

    base = os.environ.get("SKILLDAG_LLM_BASE", DEFAULT_LLM_BASE).rstrip("/")
    model = os.environ.get("SKILLDAG_LLM_MODEL", DEFAULT_LLM_MODEL)
    api_key = _require_env("SKILLDAG_LLM_API_KEY")
    timeout = int(os.environ.get("SKILLDAG_LLM_TIMEOUT_S", str(DEFAULT_LLM_TIMEOUT_S)))

    candidates_block = "\n\n".join(
        f"B{idx + 1}:\n  id: {c['id']}\n  description: {c['description']}\n  body_preview: {c.get('body_preview', '')}"
        for idx, c in enumerate(candidates)
    )
    prompt = INITIALIZE_PROMPT.format(
        a_id=anchor["id"],
        a_desc=anchor["description"],
        a_body=anchor.get("body_preview", ""),
        candidates_block=candidates_block,
        n_candidates=len(candidates),
    )

    reasoning_effort = os.environ.get("SKILLDAG_LLM_REASONING_EFFORT", "")
    payload: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    else:
        payload["temperature"] = 0.0

    content = None
    last_err: Exception | None = None
    anchor_id = anchor["id"]
    ref_to_id = {f"B{idx + 1}": c["id"] for idx, c in enumerate(candidates)}
    tag = f"{anchor_id} [+{len(candidates)}]"

    for attempt in range(LLM_MAX_RETRIES + 1):
        try:
            status, body = _http_post_json(
                f"{base}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                payload=payload,
                timeout=timeout,
            )
            if status in (502, 503, 504, 429):
                last_err = RuntimeError(f"HTTP {status}: {body[:120]}")
                if attempt < LLM_MAX_RETRIES:
                    time.sleep(LLM_RETRY_BACKOFF[attempt])
                    continue
                raise last_err
            if status >= 400:
                raise RuntimeError(f"HTTP {status}: {body[:200]}")
            content = json.loads(body)["choices"][0]["message"]["content"]
            break
        except _HTTPTimeout as e:
            last_err = e
            if attempt < LLM_MAX_RETRIES:
                time.sleep(LLM_RETRY_BACKOFF[attempt])
                continue
            logger.warning("classify_anchor(%s) timeout after %d retries: %s",
                           tag, LLM_MAX_RETRIES, e)
            return []
        except Exception as e:
            last_err = e
            logger.warning("classify_anchor(%s) failed: %s", tag, e)
            return []

    if content is None:
        logger.warning("classify_anchor(%s) all retries exhausted: %s", tag, last_err)
        return []

    try:
        parsed = json.loads(_strip_json_fence(content))
    except Exception as e:
        logger.warning("classify_anchor(%s) JSON parse failed: %s | raw=%s",
                       tag, e, content[:200])
        return []

    if not isinstance(parsed, list):
        logger.warning("classify_anchor(%s) expected JSON array, got %s", tag, type(parsed).__name__)
        return []
    if len(parsed) != len(candidates):
        logger.warning("classify_anchor(%s) count mismatch: got %d, expected %d",
                       tag, len(parsed), len(candidates))

    edges: list[dict[str, str]] = []
    seen_targets: set[str] = set()
    for item in parsed:
        if not isinstance(item, dict):
            continue
        tgt_ref = item.get("target_ref")
        if tgt_ref not in ref_to_id or tgt_ref in seen_targets:
            logger.warning("classify_anchor(%s) invalid target_ref=%r; dropping", tag, tgt_ref)
            continue
        seen_targets.add(tgt_ref)
        tgt_id = ref_to_id[tgt_ref]

        t = item.get("type")
        if t not in VALID_COLD_START_TYPES:
            continue  # "none" or malformed → skip

        src_ref = item.get("source_ref")
        if src_ref == "A":
            src = anchor_id
        elif src_ref == tgt_ref:
            src = tgt_id
        else:
            logger.warning("classify_anchor(%s) invalid source_ref=%r for target %s; dropping",
                           tag, src_ref, tgt_ref)
            continue
        # Derive target from the pair (anchor_id, tgt_id) minus source.
        tgt = tgt_id if src == anchor_id else anchor_id

        edges.append({
            "source": src,
            "target": tgt,
            "type": t,
            "reason": (item.get("reason") or "").strip()[:200],
        })

    return edges


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _load_embeddings_cache(path: Path | None) -> dict[str, dict[str, Any]]:
    """Load embeddings cache file, failing loudly when a path is specified."""
    if path is None:
        return {}
    if not path.exists():
        raise InitializationError(f"e_self embeddings cache not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise InitializationError(f"failed to read e_self embeddings cache {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise InitializationError(f"e_self embeddings cache must be a JSON object: {path}")
    return data


def initialize_edges(
    nodes: dict[str, dict[str, Any]],
    embeddings_cache_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Build cold-start edges for the given nodes dict (skill_id -> node metadata).

    Args:
        nodes: {skill_id: {name, description, path, status, tags, ...}}
        embeddings_cache_path: path to existing e_self embeddings cache
            (same format as skillgraph.embeddings.json: {skill_id: {text_hash, model, embedding}}).
            When provided, every node must have a valid cache entry for the
            active embedding model; no e_self embedding API calls are made.

    Returns:
        List of edge dicts: {source, target, type, origin="cold_start", reason}
    """
    if not nodes:
        logger.info("initialize: no nodes → no edges")
        return []

    ids = sorted(nodes.keys())
    node_list: list[dict[str, str]] = []
    texts: list[str] = []

    for sid in ids:
        n = nodes[sid]
        desc = str(n.get("description", ""))
        body = _read_body_preview(str(n.get("path", "")))
        node_list.append({"id": sid, "description": desc, "body_preview": body})
        texts.append(f"{sid}: {desc}\n\n{body}".strip())

    concurrency = int(os.environ.get("SKILLDAG_LLM_CONCURRENCY", "10"))
    model = os.environ.get("SKILLDAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)

    cache_path_raw = embeddings_cache_path or os.environ.get(INITIALIZE_EMBEDDINGS_CACHE_ENV)
    cache_path = Path(cache_path_raw) if cache_path_raw else None

    # --- e_self: load from cache when explicitly supplied; otherwise embed.
    emb_cache = _load_embeddings_cache(cache_path)
    embeddings: list[list[float]] = [[] for _ in ids]
    to_embed_indices: list[int] = []
    cache_misses: list[str] = []

    for idx, sid in enumerate(ids):
        entry = emb_cache.get(sid)
        if (
            isinstance(entry, dict)
            and entry.get("model") == model
            and isinstance(entry.get("embedding"), list)
        ):
            embeddings[idx] = entry["embedding"]
        else:
            if cache_path:
                cache_misses.append(sid)
            else:
                to_embed_indices.append(idx)

    if cache_misses:
        preview = ", ".join(cache_misses[:8])
        if len(cache_misses) > 8:
            preview += f", ... (+{len(cache_misses) - 8} more)"
        raise InitializationError(
            f"e_self embeddings cache {cache_path} is incomplete for model {model}: "
            f"{len(cache_misses)}/{len(ids)} missing or invalid ({preview})"
        )

    if to_embed_indices:
        logger.info("initialize: embedding %d/%d skills via e_self (model=%s, %d cached)",
                    len(to_embed_indices), len(ids), model, len(ids) - len(to_embed_indices))
        fresh = _embed_batch([texts[i] for i in to_embed_indices])
        for i, emb in zip(to_embed_indices, fresh):
            embeddings[i] = emb
    else:
        logger.info("initialize: all %d e_self embeddings loaded from cache", len(ids))

    logger.info("initialize: computing e_self cosine matrix (%dx%d)", len(embeddings), len(embeddings))
    sim = _cos_matrix(embeddings)

    logger.info("initialize: ranking e_self candidate pairs (adaptive threshold, top-K=%d)", TOP_K_MAX)
    self_pairs = _rank_candidates(sim)
    logger.info("initialize: e_self produced %d candidate pairs", len(self_pairs))

    # ----- Second view: imagined-tasks → e_needs ---------------------------
    logger.info("initialize: extracting imagined needs for %d skills (concurrency=%d)",
                len(node_list), concurrency)
    needs_texts: list[str] = [""] * len(node_list)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        fut_to_idx = {pool.submit(_extract_imagined_needs, n): i for i, n in enumerate(node_list)}
        for fut in as_completed(fut_to_idx):
            i = fut_to_idx[fut]
            try:
                needs_texts[i] = fut.result() or ""
            except Exception as e:
                logger.warning("imagined_needs(%s) raised: %s", node_list[i]["id"], e)
                needs_texts[i] = ""

    non_sc_indices = [
        i for i, t in enumerate(needs_texts)
        if t and t.strip().lower() != "self-contained"
    ]
    n_with_needs = len(non_sc_indices)
    n_self_contained = len(node_list) - n_with_needs
    logger.info("initialize: needs extraction complete — %d with needs, %d self-contained",
                n_with_needs, n_self_contained)

    needs_pairs: set[frozenset] = set()
    if non_sc_indices:
        needs_to_embed = [needs_texts[i] for i in non_sc_indices]
        logger.info("initialize: embedding %d needs sentences", len(needs_to_embed))
        needs_embeds = _embed_batch(needs_to_embed)
        e_needs: list[list[float] | None] = [None] * len(node_list)
        for idx, emb in zip(non_sc_indices, needs_embeds):
            e_needs[idx] = emb
        logger.info("initialize: ranking e_needs candidate pairs")
        needs_pairs = _rank_needs_candidates(e_needs, embeddings)
        logger.info("initialize: e_needs produced %d candidate pairs", len(needs_pairs))

    idx_pairs = self_pairs | needs_pairs
    overlap = len(self_pairs & needs_pairs)
    logger.info(
        "initialize: %d total candidate pairs (e_self: %d, e_needs: %d, overlap: %d)",
        len(idx_pairs), len(self_pairs), len(needs_pairs), overlap,
    )

    if not idx_pairs:
        return []

    # Bucket candidate pairs by anchor: the lower-index endpoint owns each pair.
    # This ensures every pair is classified exactly once (no double-counting).
    anchor_buckets: dict[int, list[int]] = {}
    for fs in idx_pairs:
        i, j = sorted(fs)
        anchor_buckets.setdefault(i, []).append(j)

    batch_args: list[tuple[dict, list[dict]]] = []
    total_pairs = 0
    for anchor_idx, neighbor_idxs in anchor_buckets.items():
        neighbor_idxs_sorted = sorted(neighbor_idxs)
        anchor = node_list[anchor_idx]
        cands = [node_list[j] for j in neighbor_idxs_sorted]
        batch_args.append((anchor, cands))
        total_pairs += len(cands)

    logger.info("initialize: %d anchor batches covering %d pairs (concurrency=%d)",
                len(batch_args), total_pairs, concurrency)

    edges: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_classify_anchor_batch, a, c): a["id"] for a, c in batch_args}
        for idx, fut in enumerate(as_completed(futures)):
            try:
                result_edges = fut.result()
            except Exception as e:
                logger.warning("classify_anchor_batch(%s) raised: %s", futures[fut], e)
                result_edges = []
            for r in result_edges:
                edges.append({
                    "source": r["source"],
                    "target": r["target"],
                    "type": r["type"],
                    "origin": "cold_start",
                    "reason": r["reason"],
                })
            if (idx + 1) % 20 == 0:
                logger.info("initialize: finished %d/%d anchor batches", idx + 1, len(batch_args))

    logger.info("initialize: generated %d edges from %d anchor batches (%d pairs)",
                len(edges), len(batch_args), total_pairs)
    return edges
