# Memory Consolidation Prompt

You are maintaining the classification memory for sortAI, a system that automatically sorts PDF documents into a folder tree based on their content.
Documents are placed in the correct folder by an LLM that navigates through the archive hierarchy step-by-step, choosing one subfolder at a time.

A new rule has just been learned from a user correction and added to the memory.
Review all rules and produce a clean, consolidated list:
- Merge rules that say the same thing or overlap significantly
- Resolve contradictions (keep the more specific or more recent one)
- Remove rules that are too vague or obvious to be useful
- Generalize where multiple specific rules share a common pattern
- Keep the list under 25 rules

**New rule (just learned — pay special attention):**
{{new_rule}}

**Full memory including the new rule:**
{{current_memory}}

Reply as JSON: {"rules": ["rule 1", "rule 2", ...]}
The rules array must contain only plain-English rule strings, without numbering.
