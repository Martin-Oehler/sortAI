# Memory Consolidation Prompt

You are maintaining the classification memory for sortAI, a system that automatically sorts PDF documents into a folder tree based on their content.
Documents are placed in the correct folder by an LLM that navigates through the archive hierarchy step-by-step, choosing one subfolder at a time.

A new rule has just been learned from a user correction and added to the memory.
Review all rules and produce a clean, consolidated list:
- Extract the core meaning of learned rules, then produce a completely new, consolidated list that maintains these core concepts while:
    - Merging rules that say the same thing or overlap significantly
    - Resolving contradictions (prioritize the more recent rule)
    - Removing rules that are too vague, specific, or obvious to be useful
- Generalize where multiple specific rules share a common pattern
- Keep the list under 25 rules

**New rule (just learned — pay special attention):**
{{new_rule}}

**Full memory including the new rule:**
{{current_memory}}

Reply as JSON with these fields in order:
- `reasoning`: your step-by-step thinking about which rules to merge, drop, or keep
- `rules`: the consolidated list as an array of plain-English strings, without numbering
