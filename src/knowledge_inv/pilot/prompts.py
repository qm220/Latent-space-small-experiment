DECISION_INSTRUCTIONS = """You are looking at seven views of one original CAD model (front, back, left, right, top, bottom, and isometric) and a text editing request: {instruction}

Identify and generate:
1. The intended edit.
2. A step by step plan for the edit using a CAD modeling software like CadQuery. Do not write CAD code, just describe the steps in natural language.
"""
#2. The region to modify.
#3. Geometry that should be preserved.

def decision_prompt(instruction: str, analysis: str | None = None, analysis_label: str | None = None) -> str:
    blocks = [
        DECISION_INSTRUCTIONS.replace("{instruction}", instruction.strip()),
    ]
    if analysis:
        blocks.extend(
            [
                "",
                f"Use the following {analysis_label or 'analysis'} to help you with the decision.",
                analysis.strip(),
            ]
        )
    return "\n".join(blocks)


PLAN_PROMPT = """You are looking at seven views of one original CAD model (front, back, left, right, top, bottom, and isometric) and a text editing request: {instruction}

Write a short generic plan before any final decision:
1. What you can see in the images.
2. What the request appears to ask.
3. What should be checked before editing.
"""


FBS_PROMPT = """You are looking at seven views of one original CAD model (front, back, left, right, top, bottom, and isometric) and a text editing request: {instruction}

Generate a brief function-behavior-structure (FBS) analysis of this component and the requested change:
- For function: Deduce the likely purpose of the whole component. Then, decompose the component into multiple sections and deduce the likely function of each section. Lastly, identify the functional purpose of the requested change.
- For behavior: Deduce how the component and each section is expected to work to achieve its function.
- For structure: Describe the geometry features corresponding to the component's functions and sections' functions.
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
