DECISION_INSTRUCTIONS = """You are looking at seven views of one original CAD model (front, back, left, right, top, bottom, and isometric) and a text editing request.

Identify:
1. The intended edit.
2. A step by step plan for the edit using a CAD modeling software like CadQuery. Do not write CAD code, just describe the steps in natural language.
"""
#2. The region to modify.
#3. Geometry that should be preserved.

def decision_prompt(instruction: str, analysis: str | None = None, analysis_label: str | None = None) -> str:
    blocks = [
        DECISION_INSTRUCTIONS,
        "Editing request:",
        instruction.strip(),
    ]
    if analysis:
        blocks.extend(
            [
                "",
                f"The following {analysis_label or 'analysis'} was generated in a previous stage.",
                "Treat it as model-inferred background, not as supplied expert fact.",
                analysis.strip(),
            ]
        )
    blocks.append("\nWrite the two sections requested above.")
    return "\n".join(blocks)


PLAN_PROMPT = """You are looking at seven views of one original CAD model (front, back, left, right, top, bottom, and isometric) and a text editing request.

Write a short generic plan before any final decision:
1. What you can see in the images.
2. What the request appears to ask.
3. What should be checked before editing.

Editing request:
{instruction}
"""


FBS_PROMPT = """You are looking at seven views of one original CAD model (front, back, left, right, top, bottom, and isometric) and a text editing request.

Generate a brief function-behavior-structure (FBS) background analysis:
- Function: Deduce the likely purpose of the whole part and the likely functions of its sections.  Also, predict the functional purpose of the requested change.
- Behavior: Deduce how the part is expected to work before and after the change.
- Structure: identify the geometry features corresponding to the part's functions and sections relevant to the change.

This analysis is inferred from the images and request only. Label uncertain items as assumptions.
Editing request:
{instruction}
"""

CONDITIONS = {
    "direct": {
        "label": "Direct decision",
        "analysis_stage": None,
    },
    "plan": {
        "label": "Generic planning followed by decision",
        "analysis_stage": "generic_plan",
        "analysis_prompt": PLAN_PROMPT,
        "analysis_label": "generic plan",
    },
    "fbs": {
        "label": "Generated FBS/background analysis followed by decision",
        "analysis_stage": "generated_fbs",
        "analysis_prompt": FBS_PROMPT,
        "analysis_label": "generated FBS analysis",
    },
}
