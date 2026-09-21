"""Optional ALFWorld-only candidate and audit policy.

Nothing in this module is used by the generic Judge prompt.
"""

CONTRASTIVE_SKILL_GROUPS = {
    "clean": ("alfworld-clean-object", "alfworld-object-cooler", "alfworld-object-heater"),
    "cool": ("alfworld-object-cooler", "alfworld-object-heater", "alfworld-clean-object"),
    "heat": ("alfworld-object-heater", "alfworld-object-cooler", "alfworld-clean-object"),
}

CONTRASTIVE_TRIGGER_PATTERNS = {
    "clean": r"\b(clean|cleaned|wash|washed)\b",
    "cool": r"\b(cool|cooled|chill|chilled)\b",
    "heat": r"\b(heat|heated|cook|cooked|warm|warmed)\b",
}
