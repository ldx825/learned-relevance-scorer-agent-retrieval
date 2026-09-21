import json
import os
import re
import asyncio
import inspect
import hashlib
from pathlib import Path
import yaml
from typing import Any
import sys


_SKILLS_REF_SRC = str(Path(__file__).resolve().parent)
if _SKILLS_REF_SRC not in sys.path:
    sys.path.insert(0, _SKILLS_REF_SRC)

# Try to import GoS engine
try:
    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from gos import SkillGraphRAG
    from gos.core.engine import build_default_embedding_service, build_default_llm_service
    from gos.core.schema import QuerySchema
except ImportError:
    SkillGraphRAG = None
    build_default_embedding_service = None
    build_default_llm_service = None
    QuerySchema = None

try:
    from gos.ncf.retrieval37 import ALFWorld37Retriever
except ImportError:
    ALFWorld37Retriever = None

try:
    from .utils import get_llm_response
    from .skills_ref import to_prompt as skills_ref_to_prompt
except ImportError:
    from utils import get_llm_response
    from skills_ref import to_prompt as skills_ref_to_prompt

class SkillModule:
    def __init__(self, **kwargs):
        self.skills_dir = Path(kwargs.get("skills_dir", "skills"))
        self.model = kwargs.get("model", "gpt-4o")
        self.mode = kwargs.get("mode", "gos") # "all_full", "gos", "vector", "none"
        self.gos_workspace = kwargs.get("gos_workspace", None)
        self.enable_alfworld_gating = bool(kwargs.get("enable_alfworld_gating", False))
        self.skill_graph = kwargs.get("skill_graph")
        self.skill_embeddings = kwargs.get("skill_embeddings")
        self.ncf_model = kwargs.get("ncf_model")
        self.embedding_cache = kwargs.get("embedding_cache")

        self.last_retrieval_result: Any = None
        self.last_retrieval_status = "NOT_RUN"
        self.last_retrieval_summary = ""
        self.last_retrieved_skill_names = []
        self.last_retrieval_query = ""
        self.last_ncf_ranking = []
        self.runtime_skill_events = []
        self.runtime_skill_count = 0
        self.runtime_last_injection_step = -999

        self.metadata = self._load_metadata()

        gos_modes = {"gos", "vector", "paper_gos", "paper_gos_ncf"}

        # Initialize GoS if needed
        if self.mode in gos_modes and not self.gos_workspace:
            raise ValueError(f"{self.mode} mode requires `gos_workspace`.")

        if self.mode in gos_modes and SkillGraphRAG is None:
            raise ImportError("Failed to import `gos.SkillGraphRAG`; retrieval is unavailable.")

        if self.mode in gos_modes and SkillGraphRAG and self.gos_workspace:
            gos_workspace = str(Path(self.gos_workspace).expanduser().resolve())
            self.gos_workspace = gos_workspace
            self.rag = SkillGraphRAG(
                working_dir=gos_workspace,
                config=SkillGraphRAG.Config(
                    working_dir=gos_workspace,
                    prebuilt_working_dir=gos_workspace,
                    llm_service=build_default_llm_service() if build_default_llm_service else None,
                    embedding_service=build_default_embedding_service() if build_default_embedding_service else None,
                    # ALFWorld already constructs a retrieval-oriented query.
                    # Skip GoS internal LLM rewrite here to avoid schema-format drift.
                    enable_query_rewrite=False,
                )
            )
        else:
            self.rag = None

        self.retriever37 = None
        if self.mode in {"graph37", "graph37_ncf", "paper_gos_ncf"}:
            if ALFWorld37Retriever is None:
                raise ImportError("Failed to import the 37-skill retriever.")
            if not self.skill_graph or not self.skill_embeddings:
                raise ValueError(
                    f"{self.mode} requires skill_graph and skill_embeddings."
                )
            if self.mode in {"graph37_ncf", "paper_gos_ncf"} and not self.ncf_model:
                raise ValueError(f"{self.mode} requires ncf_model.")
            self.retriever37 = ALFWorld37Retriever(
                graph_path=self.skill_graph,
                embeddings_path=self.skill_embeddings,
                model_path=(
                    self.ncf_model
                    if self.mode in {"graph37_ncf", "paper_gos_ncf"}
                    else None
                ),
            )

    def _log(self, message):
        print(f"[SkillModule] {message}")

    def _is_alfworld_task(self, task):
        task_lower = task.lower()
        return "your task is to:" in task_lower or "you are in the middle of a room" in task_lower

    def _extract_alfworld_goal(self, task):
        match = re.search(r"your task is to:\s*(.+)", task, re.IGNORECASE)
        if match:
            return match.group(1).splitlines()[0].strip().rstrip('.')
        return task.strip()

    def _infer_task_type(self, goal):
        goal_lower = goal.lower()
        if "look at" in goal_lower or "examine" in goal_lower:
            return "examine"
        if "find two" in goal_lower or "put two" in goal_lower:
            return "put_two"
        if "clean" in goal_lower:
            return "clean_and_place"
        if "cool" in goal_lower:
            return "cool_and_place"
        if "heat" in goal_lower or "hot " in goal_lower:
            return "heat_and_place"
        if "put" in goal_lower:
            return "put"
        return "other"

    def _extract_required_state(self, goal):
        goal_lower = goal.lower()
        if "clean" in goal_lower:
            return "clean"
        if "cool" in goal_lower:
            return "cool"
        if "heat" in goal_lower or "hot " in goal_lower:
            return "hot"
        return "none"

    def _extract_count(self, goal):
        goal_lower = goal.lower()
        if "find two" in goal_lower or "put two" in goal_lower:
            return "2"
        if re.search(r"\bsome\b", goal_lower):
            return "some"
        if re.search(r"\ban?\b", goal_lower):
            return "1"
        return "unspecified"

    def _extract_target_receptacle(self, goal):
        patterns = [
            r"\b(?:in|into|inside|on|onto|under)\s+([a-z0-9]+)",
            r"\bto\s+([a-z0-9]+)$",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, goal, re.IGNORECASE)
            if matches:
                return matches[-1].lower()
        return "unknown"

    def _extract_device(self, goal):
        devices = [
            "desklamp",
            "microwave",
            "fridge",
            "sinkbasin",
            "coffeemachine",
            "stoveburner",
            "cabinet",
            "drawer",
            "dresser",
            "garbagecan",
            "diningtable",
            "countertop",
            "desk",
            "sidetable",
            "table",
            "toilet",
        ]
        goal_lower = goal.lower()
        for device in devices:
            if device in goal_lower:
                return device
        return "none"

    def _extract_primary_object(self, goal):
        goal_lower = goal.lower().rstrip('.')
        patterns = [
            r"(?:look at|examine)\s+the\s+([a-z0-9]+)",
            r"(?:look at|examine)\s+([a-z0-9]+)",
            r"(?:put|find|clean|cool|heat)\s+(?:a|an|some|two)?\s*(?:clean|cool|hot|heated)?\s*([a-z0-9]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, goal_lower)
            if match:
                candidate = match.group(1).lower()
                if candidate not in {"clean", "cool", "hot", "heated"}:
                    return candidate
        tokens = re.findall(r"[a-z0-9]+", goal_lower)
        stop = {
            "put", "find", "clean", "cool", "heat", "hot", "heated", "look", "at", "examine",
            "the", "a", "an", "some", "two", "in", "into", "inside", "on", "onto",
            "under", "with", "and", "it", "them"
        }
        for token in tokens:
            if token not in stop:
                return token
        return "unknown"

    def _build_alfworld_structured_query(self, task):
        goal = self._extract_alfworld_goal(task)
        task_type = self._infer_task_type(goal)
        obj = self._extract_primary_object(goal)
        required_state = self._extract_required_state(goal)
        target_receptacle = self._extract_target_receptacle(goal)
        count = self._extract_count(goal)
        device = self._extract_device(goal)
        if QuerySchema is None:
            return (
                "environment=alfworld; "
                f"task_type={task_type}; "
                f"goal={goal}; "
                f"object={obj}; "
                f"required_state={required_state}; "
                f"target_receptacle={target_receptacle}; "
                f"count={count}; "
                f"device={device}; "
                "actions=navigate,take,move,open,close,heat,cool,clean,use,look; "
                "optimization=shortest_valid_action_sequence"
            )

        schema = QuerySchema(
            goal=goal,
            task_name=f"alfworld-{task_type}",
            domain=["alfworld", "household manipulation", "embodied task planning"],
            operations=[
                task_type,
                "navigate",
                "take",
                "move",
                "open",
                "close",
                "heat",
                "cool",
                "clean",
                "look",
            ],
            artifacts=[obj, target_receptacle, device],
            constraints=[
                f"required_state={required_state}",
                f"count={count}",
                "optimize for shortest valid action sequence",
            ],
            keywords=[
                "environment=alfworld",
                f"object={obj}",
                f"target_receptacle={target_receptacle}",
                f"device={device}",
            ],
        )
        return schema.to_query_text()

    def _build_targeted_retrieval_query(self, task):
        if self._is_alfworld_task(task):
            return self._build_alfworld_structured_query(task)

        return task.strip()

    def _embedding_cache_rows(self):
        if not self.embedding_cache:
            return {}
        path = Path(self.embedding_cache).expanduser()
        if not path.exists():
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    def _task_embedding(self, task):
        """Embed the raw goal, matching the text used to train NeuMF."""
        goal = self._extract_alfworld_goal(task)
        if goal and goal[-1] not in ".!?":
            goal += "."
        key = "sha256:" + hashlib.sha256(goal.encode("utf-8")).hexdigest()
        cache = self._embedding_cache_rows()
        cached = cache.get(key)
        if isinstance(cached, dict) and isinstance(cached.get("embedding"), list):
            self._log(f"37-skill embedding cache hit key={key[:19]}")
            return cached["embedding"]

        api_key = (
            os.environ.get("GOS37_EMBEDDING_API_KEY")
            or os.environ.get("SKILLDAG_EMBEDDING_API_KEY")
        )
        base_url = (
            os.environ.get("GOS37_EMBEDDING_BASE")
            or os.environ.get("SKILLDAG_EMBEDDING_BASE")
        )
        model = (
            os.environ.get("GOS37_EMBEDDING_MODEL")
            or os.environ.get("SKILLDAG_EMBEDDING_MODEL")
            or "text-embedding-3-large"
        )
        if not api_key:
            raise ValueError("GOS37_EMBEDDING_API_KEY is empty.")

        from openai import OpenAI

        response = OpenAI(api_key=api_key, base_url=base_url or None).embeddings.create(
            model=model,
            input=[goal],
        )
        embedding = response.data[0].embedding
        if self.embedding_cache:
            path = Path(self.embedding_cache).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            cache[key] = {"task_text": goal, "model": model, "embedding": embedding}
            path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        self._log(f"37-skill embedding API call finished model={model}")
        return embedding

    def _ncf_candidate_embeddings(self, skills):
        """Raw content embeddings for known and cold-start GoS candidates."""
        known = {
            skill_id: self.retriever37.skill_embeddings[index].tolist()
            for skill_id, index in self.retriever37.index.items()
        }
        result = {}
        cache = self._embedding_cache_rows()
        missing = []

        for skill in skills:
            if skill.name in known:
                result[skill.name] = known[skill.name]
                continue
            text = (
                f"{skill.name}: {skill.description}\n\n"
                f"{skill.payload}"
            ).strip()
            key = "skill-sha256:" + hashlib.sha256(
                text.encode("utf-8")
            ).hexdigest()
            cached = cache.get(key)
            if isinstance(cached, dict) and isinstance(cached.get("embedding"), list):
                result[skill.name] = cached["embedding"]
            else:
                missing.append((skill.name, text, key))

        if missing:
            api_key = (
                os.environ.get("GOS37_EMBEDDING_API_KEY")
                or os.environ.get("SKILLDAG_EMBEDDING_API_KEY")
            )
            base_url = (
                os.environ.get("GOS37_EMBEDDING_BASE")
                or os.environ.get("SKILLDAG_EMBEDDING_BASE")
            )
            model = (
                os.environ.get("GOS37_EMBEDDING_MODEL")
                or os.environ.get("SKILLDAG_EMBEDDING_MODEL")
                or "text-embedding-3-large"
            )
            if not api_key:
                raise ValueError("GOS37_EMBEDDING_API_KEY is empty.")
            from openai import OpenAI

            response = OpenAI(
                api_key=api_key, base_url=base_url or None
            ).embeddings.create(
                model=model,
                input=[text for _, text, _ in missing],
            )
            for (name, text, key), item in zip(missing, response.data):
                embedding = item.embedding
                result[name] = embedding
                cache[key] = {
                    "skill_id": name,
                    "text_hash": key,
                    "model": model,
                    "embedding": embedding,
                }
            if self.embedding_cache:
                path = Path(self.embedding_cache).expanduser()
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(cache, ensure_ascii=False),
                    encoding="utf-8",
                )
            self._log(
                f"embedded {len(missing)} unseen GoS candidates for "
                "content-based NeuMF cold start"
            )
        return result

    def _skill_confident_enough(self, skill):
        rerank_score = float(getattr(skill, "rerank_score", 0.0) or 0.0)
        score = float(getattr(skill, "score", 0.0) or 0.0)
        semantic_rank = getattr(skill, "semantic_rank", None)

        if rerank_score >= 0.60:
            return True
        if rerank_score >= 0.45 and semantic_rank is not None and semantic_rank <= 2:
            return True
        if score >= 0.30 and semantic_rank is not None and semantic_rank <= 2:
            return True
        return False

    def _effective_top_k(self, task, requested_top_k):
        if self.mode in {"paper_gos", "paper_gos_ncf"}:
            # GoS Appendix Table 7 fixes the hydrated bundle at top-N=8.
            return 8
        if self._is_alfworld_task(task):
            return min(requested_top_k, 4)
        return requested_top_k

    def _extract_vector_skill_payloads(self, result):
        skill_names = [skill.name for skill in result.skills]
        skill_payloads = [skill.payload for skill in result.skills]
        return skill_payloads, skill_names

    def _filter_skills_for_task(self, task, result, *, source_label="retrieval"):
        if not self._is_alfworld_task(task) or not self.enable_alfworld_gating:
            skill_names = [skill.name for skill in result.skills]
            skill_payloads = [skill.payload for skill in result.skills]
            return skill_payloads, skill_names

        confident_skills = [skill for skill in result.skills if self._skill_confident_enough(skill)]
        if not confident_skills:
            self._log(f"alfworld gating pruned all {source_label} results; returning NO_SKILL_HIT")
            return [], []

        selected_skills = confident_skills
        skill_names = [skill.name for skill in selected_skills]
        skill_payloads = [skill.payload for skill in selected_skills]

        self._log(
            f"alfworld gating kept {len(selected_skills)}/{len(result.skills)} {source_label} skills after confidence pruning"
        )
        return skill_payloads, skill_names

    def should_generate_procedure(self, task):
        return False

    def _reset_retrieval_state(self):
        self.last_retrieval_result = None
        self.last_retrieval_status = "NOT_RUN"
        self.last_retrieval_summary = ""
        self.last_retrieved_skill_names = []
        self.last_retrieval_query = ""
        self.last_ncf_ranking = []
        self.runtime_skill_events = []
        self.runtime_skill_count = 0
        self.runtime_last_injection_step = -999

    def _set_retrieval_state(self, status, summary="", skill_names=None, result=None):
        self.last_retrieval_status = status
        self.last_retrieval_summary = summary or ""
        self.last_retrieved_skill_names = list(skill_names or [])
        self.last_retrieval_result = result

    def _all_metadata_entries(self):
        return [
            {
                "name": name,
                "description": data.get("description", ""),
                "skill_dir": data.get("skill_dir", ""),
            }
            for name, data in sorted(self.metadata.items())
        ]

    def _all_metadata_context(self):
        lines = []
        for item in self._all_metadata_entries():
            lines.append(f"- {item['name']}: {item['description']}")
        return "\n".join(lines)

    def _all_metadata_skill_bundle(self):
        metadata_context = self._all_metadata_context()
        if not metadata_context:
            return []
        return [
            "=== Full Skill Library Metadata ===\n"
            "The following is the full available skill library. Treat it as capability exposure, not as a pre-filtered retrieval result.\n\n"
            f"{metadata_context}"
        ]

    def get_all_full_exposure_messages(self):
        if self.mode != "all_full":
            return []

        skill_dirs = [Path(item["skill_dir"]) for item in self._all_metadata_entries() if item.get("skill_dir")]
        if not skill_dirs:
            return []

        prompt_block = skills_ref_to_prompt(skill_dirs)
        return [
            "The following block lists the full available skill library in Anthropic skills-ref format. "
            "This is not a pre-filtered retrieval result. Use it as a catalog of available capabilities. "
            "If a skill looks relevant, prefer reading only the few most relevant skills by exact name.\n\n"
            f"{prompt_block}"
        ]

    def get_all_full_exposure_message(self):
        messages = self.get_all_full_exposure_messages()
        if not messages:
            return ""
        return messages[0]

    def get_agent_skill_request_message(self):
        if self.mode in {"none", "paper_gos", "paper_gos_ncf"}:
            return ""

        lines = [
            "Tool-style skill access is available in this run.",
            "Use it when you are blocked, the syntax is unclear, the retrieved skills look mismatched to the current blocker, or 1-2 actions already failed.",
            "Use skills conditionally, not by default: if the next environment action is already obvious from the current observation, act directly instead of retrieving.",
            "Prefer retrieval when the exact syntax is unclear, the task needs a multi-step procedure or tool setup, the current shortlist looks mismatched, or 1-2 recent actions failed.",
            "Prefer READ_SKILL when you already have a promising exact skill name. For measurement, electrical connection, conditional placement, or any unfamiliar procedure, do not guess the syntax twice in a row; retrieve first, then read the single best skill before continuing.",
            "Mirror the current benchmark vocabulary in retrieval queries. Reuse the task's own object, property, tool, room, and container words instead of naming a different environment.",
            "In a request turn, output exactly two lines: `Thought: ...` and `SkillRequest: ...`. Do not output an `Action:` line in the same turn.",
        ]

        if self.mode == "gos":
            lines.extend([
                "Available requests:",
                "- `SkillRequest: GOS_RETRIEVE <short focused query>` to search GoS again. Prefer this first when you are blocked or the current shortlist looks noisy, generic, or off-task.",
                "- `SkillRequest: READ_SKILL <exact skill name>` to read one concrete skill after GoS has surfaced a promising candidate.",
                "Examples:",
                "- `Thought: I already have a good shortlist and need the exact instructions from one candidate.`",
                "  `SkillRequest: READ_SKILL <exact shortlisted skill name>`",
                "- `Thought: The current shortlist looks noisy. I need a narrower retrieval grounded in the current task.`",
                "  `SkillRequest: GOS_RETRIEVE <target object> <property or subgoal> <tool if needed> <room> <destination container>`",
                "- `Thought: I failed twice and need retrieval that mirrors the current blocker instead of guessing again.`",
                "  `SkillRequest: GOS_RETRIEVE <task-specific keywords from the current benchmark only>`",
                "Use skill requests sparingly, only when they directly help the next action. Prefer a two-step pattern: `GOS_RETRIEVE` to shortlist candidates, then `READ_SKILL` for the single best candidate before guessing again.",
            ])
        elif self.mode == "vector":
            lines.extend([
                "Available requests:",
                "- `SkillRequest: VECTOR_RETRIEVE <short focused query>` to run vector-only retrieval again. This uses embedding similarity only, without graph propagation or lexical expansion.",
                "- `SkillRequest: READ_SKILL <exact skill name>` to read a known skill file. Use this only when you already know the exact skill you want.",
                "Examples:",
                "- `Thought: I already have a good shortlist and need the exact instructions from one candidate.`",
                "  `SkillRequest: READ_SKILL <exact shortlisted skill name>`",
                "- `Thought: The current shortlist looks noisy. I need a narrower vector retrieval grounded in the current task.`",
                "  `SkillRequest: VECTOR_RETRIEVE <target object> <property or subgoal> <tool if needed> <room> <destination container>`",
                "- `Thought: I failed twice and need vector retrieval that mirrors the current blocker instead of guessing again.`",
                "  `SkillRequest: VECTOR_RETRIEVE <task-specific keywords from the current benchmark only>`",
                "Use skill requests sparingly, only when they directly help the next action. In vector mode, prefer `VECTOR_RETRIEVE` before guessing again, and `READ_SKILL` only after a specific skill name looks relevant.",
            ])
        elif self.mode in {"graph37", "graph37_ncf"}:
            lines.extend([
                "Available requests:",
                "- `SkillRequest: READ_SKILL <exact skill name>` to read one concrete skill from the retrieved shortlist.",
                "This paired experiment freezes retrieval to one initial task-level recommendation. Use READ_SKILL only when one shortlisted skill directly helps the next action.",
            ])
        elif self.mode == "all_full":
            lines.extend([
                "Available requests:",
                "- `SkillRequest: READ_SKILL <exact skill name>` to read a known skill file.",
                "Examples:",
                "- `Thought: The full catalog already shows a likely match and I need its exact instructions.`",
                "  `SkillRequest: READ_SKILL <exact skill name already visible in the catalog>`",
                "Use skill requests sparingly, only when they directly help the next action. In all_full mode, do not attempt retrieval; read a specific skill only when the full catalog already reveals a directly relevant candidate.",
            ])

        else:
            return ""
        return "\n".join(lines)

    def _skill_catalog_entries(self, skill_names):
        entries = []
        for name in skill_names or []:
            meta = self.metadata.get(name, {})
            entries.append(
                {
                    "name": name,
                    "description": meta.get("description", ""),
                    "skill_dir": meta.get("skill_dir", ""),
                }
            )
        return entries

    def _format_retrieval_shortlist(self, header, query, skill_names, source_label):
        if not skill_names:
            return f"{header}\n\nNo relevant skills were retrieved."

        lines = [
            header,
            f"Query: {query}",
            f"Shortlisted {source_label} candidates:",
        ]
        for entry in self._skill_catalog_entries(skill_names[:3]):
            description = entry["description"] or "No description available."
            lines.append(f"- {entry['name']}: {description}")
            if entry["skill_dir"]:
                lines.append(f"  Source: {entry['skill_dir']}/SKILL.md")
        lines.extend([
            "Do not assume these summaries are enough to execute correctly.",
            "If one candidate looks directly relevant to the current blocker, issue `SkillRequest: READ_SKILL <exact skill name>` before trying another uncertain action.",
        ])
        return "\n".join(lines)

    def _load_metadata(self):
        """Load existing metadata from file."""
        metadata = {}
        if not self.skills_dir.exists():
            return metadata
            
        for skill_dir in self.skills_dir.iterdir():
            if skill_dir.is_dir():
                skill_md_path = skill_dir / "SKILL.md"
                if skill_md_path.exists():
                    try:
                        content = skill_md_path.read_text(encoding="utf-8")
                        if content.strip().startswith('---'):
                            parts = content.split('---', 2)
                            if len(parts) >= 3:
                                header_data = yaml.safe_load(parts[1])
                                if isinstance(header_data, dict) and header_data.get('name') and header_data.get('description'):
                                    metadata[header_data['name']] = {
                                        'description': header_data['description'],
                                        'skill_dir': str(skill_dir)
                                    }
                    except Exception as e:
                        print(f"[ERROR] Failed to parse SKILL.md for {skill_dir.name}: {e}")
        return metadata
    
    def retrieve_relevant_skills(self, task, top_k=15):
        self._reset_retrieval_state()
        effective_top_k = self._effective_top_k(task, top_k)
        retrieval_query = (
            self._extract_alfworld_goal(task)
            if self.mode in {"paper_gos", "paper_gos_ncf"}
            else self._build_targeted_retrieval_query(task)
        )
        self.last_retrieval_query = retrieval_query
        self._log(
            f"retrieve_relevant_skills start mode={self.mode} top_k={top_k} effective_top_k={effective_top_k} task_chars={len(task)} retrieval_query={retrieval_query!r}"
        )

        if self.mode == "none":
            self._log("mode=none, skipping retrieval")
            return []

        if self.mode in {"graph37", "graph37_ncf"} and self.retriever37:
            method = "graph_ncf" if self.mode == "graph37_ncf" else "graph"
            rows = self.retriever37.retrieve(
                self._task_embedding(task),
                method=method,
                top_k=min(effective_top_k, 3),
            )
            skill_names = [row.skill_id for row in rows]
            status = "SKILL_HIT" if skill_names else "NO_SKILL_HIT"
            summary = self._format_retrieval_shortlist(
                f"37-skill retrieval completed with method={method}.",
                self._extract_alfworld_goal(task),
                skill_names,
                method,
            )
            self._set_retrieval_state(
                status=status,
                summary=summary,
                skill_names=skill_names,
                result={
                    "method": method,
                    "skills": [
                        {
                            "skill_id": row.skill_id,
                            "score": row.score,
                            "graph_score": row.graph_score,
                            "ncf_score": row.ncf_score,
                        }
                        for row in rows
                    ],
                },
            )
            self._log(
                f"37-skill retrieval finished method={method} skills={skill_names}"
            )
            return []
            
        if self.mode in {"paper_gos", "paper_gos_ncf"} and self.rag:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # The paper uses top-N=8, 2,400 chars/skill and 12,000 chars total.
            # For the NCF arm only, expose a hidden GoS top-12 candidate pool,
            # rerank it, then apply the original top-N and context budgets.
            candidate_top_n = 12 if self.mode == "paper_gos_ncf" else 8
            candidate_context_chars = (
                candidate_top_n * 2400
                if self.mode == "paper_gos_ncf"
                else 12000
            )
            result = loop.run_until_complete(
                self.rag.async_retrieve(
                    retrieval_query,
                    top_n=candidate_top_n,
                    max_chars_per_skill=2400,
                    max_context_chars=candidate_context_chars,
                )
            )

            if self.mode == "paper_gos_ncf":
                original_rank = {
                    skill.name: rank
                    for rank, skill in enumerate(result.skills, start=1)
                }
                task_embedding = self._task_embedding(retrieval_query)
                candidate_embeddings = self._ncf_candidate_embeddings(
                    result.skills
                )
                scores = self.retriever37.score_embedding_subset(
                    task_embedding,
                    candidate_embeddings,
                )
                ranked = sorted(
                    result.skills,
                    key=lambda skill: (
                        scores[skill.name],
                        skill.rerank_score,
                        skill.score,
                        skill.name,
                    ),
                    reverse=True,
                )[:8]
                budgeted = self.rag._fit_skills_to_context_budget(
                    retrieval_query, ranked, 12000
                )
                self.last_ncf_ranking = [
                    {
                        "skill_id": skill.name,
                        "gos_rank": original_rank[skill.name],
                        "gos_score": float(skill.rerank_score),
                        "ncf_score": float(scores[skill.name]),
                        "final_rank": final_rank,
                    }
                    for final_rank, skill in enumerate(budgeted, start=1)
                ]
                names = {skill.name for skill in budgeted}
                relations = [
                    relation
                    for relation in result.relations
                    if relation.source in names and relation.target in names
                ]
                result = result.model_copy(
                    update={
                        "budget": result.budget.model_copy(
                            update={"top_n": 8, "max_context_chars": 12000}
                        ),
                        "skills": budgeted,
                        "relations": relations,
                        "rendered_context": self.rag._render_context(
                            retrieval_query,
                            budgeted,
                            relations,
                            max_chars=12000,
                        ),
                        "summary": (
                            result.summary
                            + "\nNeuMF reranked the hidden GoS top-12 candidate "
                            "pool before the original top-8/context budgets."
                        ),
                    }
                )

            skill_names = [skill.name for skill in result.skills]
            skill_payloads = [skill.payload for skill in result.skills]
            self._set_retrieval_state(
                status="SKILL_HIT" if skill_names else "NO_SKILL_HIT",
                summary=result.summary,
                skill_names=skill_names,
                result=result,
            )
            self._log(
                f"{self.mode} finished n_skills={len(skill_names)} "
                f"query={retrieval_query!r}"
            )
            return skill_payloads

        if self.mode in {"gos", "vector"} and self.rag:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            if self.mode == "gos":
                self._log(f"starting GoS async_retrieve workspace={self.gos_workspace}")
                result = loop.run_until_complete(self.rag.async_retrieve(retrieval_query, top_n=effective_top_k))
                skill_payloads, skill_names = self._filter_skills_for_task(task, result, source_label="gos")
                status = "SKILL_HIT" if skill_names else "NO_SKILL_HIT"
                summary = result.summary
                if not skill_names:
                    status = "NO_SKILL_HIT"
                    summary = "ALFWorld retrieval gating pruned all retrieved skills; proceeding without injected skills."
                self._log(f"GoS async_retrieve finished status={status} n_skills={len(skill_names)}")
            else:
                self._log(f"starting vector async_retrieve workspace={self.gos_workspace}")
                result = loop.run_until_complete(self.rag.async_retrieve_vector(retrieval_query, top_n=effective_top_k))
                skill_payloads, skill_names = self._filter_skills_for_task(task, result, source_label="vector")
                status = "SKILL_HIT" if skill_names else "NO_SKILL_HIT"
                summary = result.summary
                self._log(f"vector async_retrieve finished status={status} n_skills={len(skill_names)}")

            self._set_retrieval_state(
                status=status,
                summary=summary,
                skill_names=skill_names,
                result=result,
            )
            return skill_payloads

        if self.mode == "all_full":
            metadata_entries = self._all_metadata_entries()
            skill_names = [entry["name"] for entry in metadata_entries]
            status = "SKILL_HIT" if skill_names else "NO_SKILL_HIT"
            summary = (
                f"Exposed full skill metadata library in a single initial dialogue message ({len(skill_names)} skills). "
                "This matches the all-skills capability-exposure baseline rather than retrieval-time shortlisting."
            )
            self._set_retrieval_state(
                status=status,
                summary=summary,
                skill_names=skill_names,
                result={"skill_names": skill_names, "mode": "all_full"},
            )
            self._log(f"all_full exposure finished status={status} n_skills={len(skill_names)}")
            return []

        self._set_retrieval_state(status="NO_SKILL_HIT", summary="No retrieval configured for this mode.")
        return []

    def get_paper_retrieval_bundle(self):
        """Return the exact hydrated context used by paper-aligned GoS modes."""
        if self.mode not in {"paper_gos", "paper_gos_ncf"}:
            return ""
        if self.last_retrieval_status != "SKILL_HIT":
            return ""
        return str(getattr(self.last_retrieval_result, "rendered_context", "") or "")

    def get_retrieval_guidance(self):
        if self.mode not in {"gos", "vector", "graph37", "graph37_ncf"} or self.last_retrieval_result is None:
            return ""

        if self.last_retrieval_status != "SKILL_HIT" or not self.last_retrieved_skill_names:
            return ""

        top_skills = self.last_retrieved_skill_names[:3]
        titles = {
            "gos": "Graph of Skills retrieval guidance:",
            "vector": "Vector-skills retrieval guidance:",
            "graph37": "Shared 37-skill graph retrieval guidance:",
            "graph37_ncf": "Shared 37-skill graph + NeuMF retrieval guidance:",
        }
        title = titles[self.mode]
        content_parts = [
            title,
            f"Retrieval Status: {self.last_retrieval_status}",
        ]
        if top_skills:
            content_parts.append("Top retrieved skills: " + ", ".join(top_skills))
            skill_lines = ["Retrieved skill summaries:"]
            for entry in self._skill_catalog_entries(top_skills):
                description = entry["description"] or "No description available."
                skill_lines.append(f"- {entry['name']}: {description}")
            content_parts.append("\n".join(skill_lines))
        content_parts.append(
            "Use retrieval only as weak high-level guidance. Prioritize the shortest path from current observation to task completion."
        )
        content_parts.append(
            "Do not follow a rigid room-wide search checklist if the current observation already reveals the target object or target receptacle."
        )
        content_parts.append(
            "If the environment feedback or reward indicates the task is complete, stop issuing new actions immediately."
        )
        content_parts.append(
            "For ALFWorld action syntax: first navigate to the destination receptacle, then use the exact action form 'move {obj} to {recep}'."
        )
        content_parts.append(
            "Do not use 'use {obj}' unless the task explicitly requires turning on, heating, cooling, or cleaning something."
        )
        if self.mode == "gos":
            content_parts.append(
                "If the current retrieved skills look mismatched to the blocker, or 1-2 actions already failed, issue `SkillRequest: GOS_RETRIEVE <short focused query>`. Treat retrieval as a shortlist step and prefer `READ_SKILL` for the single best candidate before another uncertain action."
            )
        elif self.mode == "vector":
            content_parts.append(
                "If the current retrieved skills look mismatched to the blocker, or 1-2 actions already failed, issue `SkillRequest: VECTOR_RETRIEVE <short focused query>`. After vector retrieval surfaces a plausible exact skill name, prefer `READ_SKILL` for that single candidate before another uncertain action."
            )
        return "\n\n".join(part for part in content_parts if part)

    def _get_skill_contents(self, skill_names):
        skill_contents = []
        for name in skill_names:
            if name in self.metadata:
                skill_dir = Path(self.metadata[name]['skill_dir'])
                combined_text = f"=== Skill: {name} ===\n"
                for file_path in skill_dir.rglob('*'):
                    if file_path.is_file():
                        try:
                            content = file_path.read_text(encoding='utf-8')
                            combined_text += f"\n[File: {file_path.name}]\n{content}\n"
                        except: continue
                skill_contents.append(combined_text)
        return skill_contents

    def _parse_skill_request(self, response):
        if not isinstance(response, str):
            return None, ""

        patterns = [
            r"^SkillRequest:\s*(.+)$",
            r"^Action:\s*SkillRequest:\s*(.+)$",
        ]
        match = None
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                break
        if not match:
            return None, ""

        payload = match.group(1).strip()
        if not payload:
            return None, ""

        upper = payload.upper()
        if upper.startswith("READ_SKILL "):
            return "read_skill", payload[len("READ_SKILL "):].strip()
        if upper.startswith("GOS_RETRIEVE "):
            return "gos_retrieve", payload[len("GOS_RETRIEVE "):].strip()
        if upper.startswith("VECTOR_RETRIEVE "):
            return "vector_retrieve", payload[len("VECTOR_RETRIEVE "):].strip()
        return None, payload

    def _record_runtime_skill_event(self, step, trigger, query, skill_names):
        self.runtime_skill_count += 1
        self.runtime_last_injection_step = step
        self.runtime_skill_events.append(
            {
                "step": step,
                "trigger": trigger,
                "query": query,
                "skill_names": list(skill_names or []),
            }
        )

    def _format_agent_skill_response(self, header, skill_names, skill_payloads):
        if not skill_payloads:
            return ""
        clipped_payloads = [self._clip_text(payload, 1200) for payload in skill_payloads[:2]]
        lines = [header, "Use this only if it directly improves the next action."]
        if skill_names:
            lines.append("Selected skills: " + ", ".join(skill_names[:2]))
        return "\n\n".join(lines + clipped_payloads)

    def handle_agent_skill_request(self, task, response, current_step):
        request_type, payload = self._parse_skill_request(response)
        if not request_type:
            return ""

        if request_type == "read_skill":
            skill_name = payload
            skill_payloads = self._get_skill_contents([skill_name])[:1]
            if not skill_payloads:
                return (
                    f"Skill request could not be fulfilled: skill `{skill_name}` was not found. "
                    "Use an exact skill name from the available skill list or retrieval results."
                )
            self._record_runtime_skill_event(current_step, "agent_request:read_skill", skill_name, [skill_name])
            return self._format_agent_skill_response(
                f"Skill request fulfilled: READ_SKILL {skill_name}",
                [skill_name],
                skill_payloads,
            )

        if request_type == "gos_retrieve":
            if self.mode != "gos" or not self.rag:
                return "Skill request could not be fulfilled: GOS_RETRIEVE is only available in gos mode."
            query = payload
            if not query:
                return "Skill request could not be fulfilled: empty GOS_RETRIEVE query."
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self.rag.async_retrieve(query, top_n=2))
            skill_payloads, skill_names = self._filter_skills_for_task(task, result, source_label="gos")
            skill_names = skill_names[:2]
            if not skill_names:
                return f"Skill request fulfilled: GOS_RETRIEVE {query}\n\nNo relevant skills were retrieved."
            self._record_runtime_skill_event(current_step, "agent_request:gos_retrieve", query, skill_names)
            return self._format_retrieval_shortlist(
                f"Skill request fulfilled: GOS_RETRIEVE {query}",
                query,
                skill_names,
                "GoS",
            )

        if request_type == "vector_retrieve":
            if self.mode != "vector" or not self.rag:
                return "Skill request could not be fulfilled: VECTOR_RETRIEVE is only available in vector mode."
            query = payload
            if not query:
                return "Skill request could not be fulfilled: empty VECTOR_RETRIEVE query."
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self.rag.async_retrieve_vector(query, top_n=2))
            skill_payloads, skill_names = self._filter_skills_for_task(task, result, source_label="vector")
            skill_payloads = skill_payloads[:2]
            skill_names = skill_names[:2]
            if not skill_names:
                return f"Skill request fulfilled: VECTOR_RETRIEVE {query}\n\nNo relevant skills were retrieved."
            self._record_runtime_skill_event(current_step, "agent_request:vector_retrieve", query, skill_names)
            return self._format_retrieval_shortlist(
                f"Skill request fulfilled: VECTOR_RETRIEVE {query}",
                query,
                skill_names,
                "vector",
            )

        return ""

    @staticmethod
    def _clip_text(text, max_chars=1800):
        if not text or len(text) <= max_chars:
            return text
        return text[: max_chars - 3].rstrip() + "..."

    @staticmethod
    def _recent_actions(messages, limit=2):
        actions = []
        for message in reversed(messages or []):
            if message.get("role") != "assistant":
                continue
            content = message.get("content", "")
            if not isinstance(content, str):
                continue
            match = re.search(r"Action:\s*(.+)", content, re.IGNORECASE)
            if match:
                actions.append(match.group(1).strip())
            if len(actions) >= limit:
                break
        actions.reverse()
        return actions

    def _runtime_trigger_reason(self, observation, current_step):
        observation_lower = (observation or "").lower()
        if current_step - self.runtime_last_injection_step < 3:
            return ""
        if self.runtime_skill_count >= 2:
            return ""
        failure_markers = [
            "nothing happens",
            "nothing happened",
            "you can't",
            "cannot",
            "can't",
            "not found",
            "don't see",
            "do not see",
        ]
        for marker in failure_markers:
            if marker in observation_lower:
                return f"runtime_failure:{marker}"
        return ""

    def _build_runtime_retrieval_query(self, task, messages, observation):
        base_query = self._build_targeted_retrieval_query(task)
        recent_actions = self._recent_actions(messages)
        parts = [base_query]
        if recent_actions:
            parts.append("recent_actions=" + ", ".join(recent_actions))
        compact_observation = " ".join((observation or "").split())
        if compact_observation:
            parts.append("runtime_observation=" + compact_observation[:400])
        return "\n".join(part for part in parts if part)

    def _format_runtime_skill_hint(self, skill_names, skill_payloads, trigger):
        if not skill_payloads:
            return ""
        clipped_payloads = [self._clip_text(payload, 1200) for payload in skill_payloads[:2]]
        header = [
            f"Additional runtime skill support was injected because: {trigger}.",
            "Use the following skill details only if they directly help recover and reach the shortest path to completion.",
        ]
        if skill_names:
            header.append("Selected skills: " + ", ".join(skill_names[:2]))
        return "\n\n".join(header + clipped_payloads)

    def maybe_get_runtime_skill_hint(self, task, messages, observation, current_step):
        trigger = self._runtime_trigger_reason(observation, current_step)
        if not trigger:
            return ""

        dynamic_query = self._build_runtime_retrieval_query(task, messages, observation)
        skill_names = []
        skill_payloads = []

        if self.mode in {"gos", "vector"} and self.rag:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            if self.mode == "gos":
                result = loop.run_until_complete(self.rag.async_retrieve(dynamic_query, top_n=2))
                skill_payloads, skill_names = self._filter_skills_for_task(task, result, source_label="gos")
            else:
                result = loop.run_until_complete(self.rag.async_retrieve_vector(dynamic_query, top_n=2))
                skill_payloads, skill_names = self._filter_skills_for_task(task, result, source_label="vector")

            skill_payloads = skill_payloads[:2]
            skill_names = skill_names[:2]
        if not skill_payloads:
            return ""

        self._record_runtime_skill_event(current_step, trigger, dynamic_query, skill_names)
        self._log(
            f"runtime skill injection triggered step={current_step} trigger={trigger} n_skills={len(skill_names)}"
        )
        return self._format_runtime_skill_hint(skill_names, skill_payloads, trigger)

    def get_runtime_skill_events(self):
        return list(self.runtime_skill_events)
