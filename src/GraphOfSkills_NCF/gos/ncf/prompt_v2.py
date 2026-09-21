"""Domain-agnostic prompt for groupwise task-skill supervision."""

GENERIC_GROUPWISE_JUDGE_PROMPT = """You annotate task-skill retrieval data.
Use only the raw task and the candidate skill descriptions. Do not assume hidden plans,
domain-specific conventions, or capabilities that are not stated in the evidence.

First summarize the task signature:
- core objective;
- explicitly requested actions or state changes;
- relevant inputs, entities, artifacts, and desired outputs;
- explicit constraints, quantities, ordering, destinations, tools, or resources;
- the requirements that most distinguish this task from nearby tasks.

Then select a minimal, non-redundant skill set jointly:
- primary: a capability directly grounded in a distinguishing task requirement;
- support: useful prerequisite or follow-up capability, but not task-discriminative;
- harmful: following the skill would contradict an explicit task requirement;
- uncertain: the available text is insufficient to decide;
- unlisted candidates default to irrelevant.

General reasoning constraints:
1. Do not promote a broadly useful skill to primary merely because many tasks need it.
2. Mentioning an entity, tool, location, format, or destination does not by itself imply
   invoking every operation associated with it.
3. Do not infer an unstated transformation or side effect.
4. Prefer the most specific candidate when candidates are redundant or nested.
5. Select at most 3 primary skills.

For every primary skill, copy a short exact quote from the raw task as task_evidence.
Do not paraphrase this quote. A primary without a verifiable quote will be excluded locally.

Return compact JSON only:
{"task_signature":{"core_objective":"","explicit_actions_or_changes":[],
"entities_and_artifacts":[],"constraints":[],"tools_or_resources":[],
"destination_or_output":"","distinguishing_requirements":[]},
"primary_skills":[{"skill_id":"","task_evidence":"exact quote from raw task"}],
"support_skill_ids":[],"harmful_skill_ids":[],
"uncertain_skill_ids":[],"minimal_set_reason":"short"}"""
