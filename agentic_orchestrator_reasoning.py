"""
agentic_orchestrator.py

Shared agentic orchestration layer for compositional spatial reasoning.

Agent 1: ContextRouterAgent
    Selects the task profile and rule bank .

Agent 2: RuleBankPromptAgent
    Builds the reasoning prompt from a task-independent spatial rule bank.

Agent 3: DisambiguationRepairAgent
    Parses, validates, filters, and optionally repairs the model output.

The agents are implemented as fixed modules in one orchestration workflow.
The LLM is used as the reasoning backend; the workflow itself defines when
each agent is called and how information is handed over.
"""

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from transformers import AutoTokenizer


# ============================================================
# Shared labels
# ============================================================

DIRECTION_LABELS = ["front", "back", "left", "right"]

COMPOSITIONAL_LABELS = [
    "left", "right", "above", "below",
    "behind", "in front", "near", "far",
    "outside", "outside and touching", "partially overlapping",
    "inside and touching", "inside",
    "contains and touches", "contains", "overlapping",
]

ALL_TASKS = {"directional_frame", "compositional_multilabel"}


# ============================================================
# Agent 1: routing / context profile
# ============================================================

@dataclass
class TaskProfile:
    task_name: str
    allowed_labels: List[str]
    answer_format: str
    multilabel: bool


class ContextRouterAgent:
    """
    Routes each input to the correct rule bank.

    this is the orchestration/context agent:
    it identifies the spatial reasoning setting and selects the relevant
    rule bank without asking the LLM to know dataset names.
    """

    def route(self, task: str, example: Optional[Dict[str, Any]] = None) -> TaskProfile:
        if task not in ALL_TASKS:
            raise ValueError(
                f"Unknown task '{task}'. Expected one of: {sorted(ALL_TASKS)}"
            )

        if task == "directional_frame":
            return TaskProfile(
                task_name=task,
                allowed_labels=DIRECTION_LABELS,
                answer_format='Explanation: ... Answer: {front|back|left|right}',
                multilabel=False,
            )

        return TaskProfile(
            task_name=task,
            allowed_labels=COMPOSITIONAL_LABELS,
            answer_format="JSON list of spatial relation labels",
            multilabel=True,
        )


# ============================================================
# Rule bank A: frame-of-reference directional reasoning
# ============================================================

FRAME_RULE_BANK = """You are a useful question-answering system, especially in language and spatial relations. You will be given the textual description of the scene and the corresponding question in the format "Context: sentence(s). Question: sentence.”
Answer the question with a reasonable explanation.
Also, you need to consider the information regrading topological, distance, direction, and frame of reference category.
There are four classes of the frame of reference (external intrinsic, internal intrinsic, external relative, internal relative). Note that the intrinsic direction refers to whether the model has the front/back by itself. (Example: a bird, human. Counter Example: a ball, a box).
External intrinsic. The spatial description of an entity A relative to another entity B, where (1) A is not contained by B, (2) the spatial relation is based on B's facing orientation (intrinsic direction) if B has one.
Internal intrinsic. The spatial description of an entity A relative to another entity B, where (1) A is contained by B, (2) the spatial relation is based on B's facing orientation (intrinsic direction) if B has one.
External relative. The spatial description of an entity A relative to another entity B, where (1) A is not contained by B, (2) The spatial relation is relative to entity C, where C is not involved in the spatial relation and only provides the perspective. An example of C is the camera.
Internal relative. The spatial description of an entity A relative to another entity B, where (1) A is contained by B, (2) The spatial relation is relative to entity C, where C is not involved in the spatial relation and only provides the perspective. An example of C is the camera.
There are four answer candidates {front, back, left, right} indicate the relation asked in the question. The answer is in the format of "Explanation: sentence(s). Answer: candidate."
Rule hints:
1) Determine the SOURCE perspective: if the context uses B's own facing direction → intrinsic (B's frame). If the context uses the camera/observer → relative (camera frame).
2) Determine the TARGET perspective: if the question asks "from B's perspective / based on B's angle" → intrinsic. If the question asks "from the camera/observer perspective" → relative (camera frame).
3) If SOURCE and TARGET are the same perspective → output the relation exactly as stated in the context. Do NOT rotate.
4) If SOURCE and TARGET differ, use the conversion table below. Look up B's facing direction and the given relation to find the converted relation.

   CONVERSION TABLE (works in both directions: intrinsic↔camera):
   Use it intrinsic→camera OR camera→intrinsic — the mapping is the same table either way:
   find B's facing row and locate the given direction to read off the converted direction.

   B faces TOWARD camera:
     front↔front  |  back↔back  |  left↔right  |  right↔left

   B faces AWAY from camera:
     front↔back  |  back↔front  |  left↔left  |  right↔right

   B faces camera's LEFT:
     front↔left  |  back↔right  |  left↔front  |  right↔back

   B faces camera's RIGHT:
     front↔right  |  back↔left  |  left↔back  |  right↔front

5) If no facing direction is given for B (e.g. "nan" / not mentioned), assume the relation is already in the camera/observer frame and output it unchanged.
6) Always output exactly: "Explanation: … Answer: {front|back|left|right}".
"""

FRAME_FEW_SHOTS = [
    {
        "role": "user",
        "content": "Context: The bird is outside the car and in front of the car relative to the car. The car is facing toward the camera.\nQuestion: Based on the camera's perspective, where is the bird from the car position in the scene?"
    },
    {
        "role": "assistant",
        "content": "Explanation: Topological: The bird is outside the car. Distance: The bird is in front of the car. Direction: The bird is in the front direction of the car. Frame of Reference: external intrinsic. The context states that the bird is in front of the car relative to the car. Because the car is facing toward the camera, the car's front direction aligns with the camera's front direction. Therefore, from the camera's perspective, the bird is in front of the car. Answer: front"
    },
    {
        "role": "user",
        "content": "Context: The bird is inside the car and left of the car from the car's perspective. The car is facing to the right relative to the camera.\nQuestion: Based on the camera's perspective, where is the bird from the car's position?"
    },
    {
        "role": "assistant",
        "content": "Explanation: Topological: The bird is inside the car. Distance: The bird is close to the car. Direction: The bird is in the left direction of the car. Frame of Reference: internal intrinsic. The context states that the bird is to the left of the car from the car's perspective. Because the car is facing to the right relative to the camera, the car's left direction maps to the camera's back direction. Therefore, from the camera's perspective, the bird is behind the car. Answer: back"
    },
    {
        "role": "user",
        "content": "Context: The box is inside and to the right of the room from the observer's perspective.\nQuestion: From the observer's perspective, what is the spatial relation of the box to the room?"
    },
    {
        "role": "assistant",
        "content": "Explanation: Topological: The box is inside the room. Distance: The box is close to the room. Direction: From the observer's perspective, the box is to the right of the room. Frame of Reference: internal relative. The context already gives the relation from the observer's perspective. Because this is a relative frame of reference, no rotation is needed. Therefore, the box is to the right of the room from the observer's perspective. Answer: right"
    },
    {
        "role": "user",
        "content": "Context: A phone is to the left of a tablet from my perspective. The tablet is facing to the right.\nQuestion: From my perspective, what is the spatial relation of the phone to the tablet?"
    },
    {
        "role": "assistant",
        "content": "Explanation: Topological: The phone is outside the tablet. Distance: There is some distance between the phone and the tablet. Direction: From the observer's perspective, the phone is to the left of the tablet. Frame of Reference: external relative. The context explicitly states that the relation is from the observer's perspective. Because this is a relative frame of reference, the tablet's facing direction does not change the answer. Therefore, the phone is to the left of the tablet from the observer's perspective. Answer: left"
    },
    {
        "role": "user",
        "content": "Context: A ball is behind a robot relative to the robot. The robot is facing away from the camera.\nQuestion: Based on the camera's perspective, where is the ball from the robot's position?"
    },
    {
        "role": "assistant",
        "content": "Explanation: Topological: The ball is outside the robot. Distance: There is some distance between the ball and the robot. Direction: The ball is in the back direction of the robot. Frame of Reference: external intrinsic. The context states that the ball is behind the robot relative to the robot. Because the robot is facing away from the camera, the robot's back direction corresponds to the camera's front direction. Therefore, from the camera's perspective, the ball is in front of the robot. Answer: front"
    },
]


# ============================================================
# Rule bank B: compositional spatial relation reasoning
# ============================================================

COMPOSITIONAL_RULE_BANK = """
Apply the spatial composition rules conservatively. The goal is not to output many labels; the goal is to output only labels that are clearly supported.

First identify the exact ordered relation from the text.

Important wording patterns:
- "X is left/right/above/below/behind/in front of Y" means X -> Y.
- "left/right/above/below/behind/in front of Y is X" means X -> Y.
- "Below Y is X" means X is below Y.
- "Above Y is X" means X is above Y.
- "In front of Y is X" means X is in front of Y.
- "Behind Y is X" means X is behind Y.
- "At 3:00 position relative to Y there is X" means X is right of Y.
- "At 9:00 position relative to Y there is X" means X is left of Y.
- "At 12:00 position relative to Y there is X" means X is above Y.
- "At 6:00 position relative to Y there is X" means X is below Y.
- "X is in/inside/within/covered by Y" means X is inside Y.
- "X has/contains/covers Y" means X contains Y.
- Treat "front" as "in front".

Relation groups:
- Directional relations: left, right, above, below, behind, in front.
- Distance relations: near, far.
- Proper-part relations: inside, inside and touching, contains, contains and touches.
- Symmetric topological relations: outside, outside and touching, partially overlapping, overlapping.

Rule 1 — Inverse rule:
Use for directional and proper-part relations when the known relation is in the opposite order.

Examples:
- If Y is below X, then X is above Y.
- If Y is left of X, then X is right of Y.
- If Y is in front of X, then X is behind Y.
- If X is in front of Y, then Y is behind X.
- If Y contains X, then X is inside Y.
- If Y covers X, then X is inside and touching Y.

Rule 2 — Symmetry rule:
Use only for distance and symmetric topological relations.

Examples:
- If X is far from Y, then Y is far from X.
- If X is near Y, then Y is near X.
- If X is outside Y, then Y is outside X.

Rule 3 — Directional propagation through containers:
If X is inside container A, A has a directional relation to container B, and B contains Y, transfer the same directional relation to X -> Y.

Examples:
- X inside A, A right of B, B contains Y -> X right of Y.
- X inside A, A below B, B contains Y -> X below Y.
- X inside A, A in front of B, B contains Y -> X in front of Y.

Rule 4 — Transitivity:
Use only when the same directional relation repeats.

Examples:
- X left of A and A left of Y -> X left of Y.
- X below A and A below Y -> X below Y.
- X behind A and A behind Y -> X behind Y.

If different directions appear in a path, keep them as separate labels.
Example:
- X right of A and A below Y -> X is right and below Y.

Rule 5 — Not rule:
Never output opposite pairs together:
- left and right
- above and below
- behind and in front
- inside and contains
- inside and touching and contains and touches

Very important filtering rules:
- Do NOT output "outside" just because objects are in different blocks or containers.
- Do NOT output "outside" only because a container is disconnected somewhere in the path.
- Output "outside" only if the final relation between the target side and reference side is clearly supported as disconnected/outside.
- If the reasoning chain strongly supports directional labels but outside is uncertain, omit outside.
- Do NOT output "far" unless far is directly preserved along the relevant path to the final reference.
- If the final answer is mainly directional, prefer only the directional labels.
- Keep fewer labels rather than adding uncertain labels.

After the reasoning, output the final answer as a valid JSON list using only the allowed labels.
"""

COMPOSITIONAL_FEW_SHOT_BLOCK = """
Example 1:
Context:
There exist three blocks, called AAA, BBB and CCC. Block AAA contains a big blue circle. This block is inside block BBB. A medium yellow circle and a big black circle are inside block BBB. Block BBB covers a medium blue square. The medium yellow circle is behind the big black circle and is above the medium blue square. Block BBB is disconnected from and in front of block CCC. A medium blue triangle is inside block CCC.

Question:
Where is the medium blue triangle regarding the big blue circle?

Reasoning:
Step 1: Target X is the medium blue triangle. Reference Y is the big blue circle.
Step 2: Build the relevant chain: triangle -> CCC -> BBB -> AAA -> big blue circle.
Step 3: The triangle is inside CCC. This is a PP relation.
Step 4: The context says BBB is in front of CCC. This is a Dir relation in the opposite order, so apply inverse reasoning: CCC is behind BBB.
Step 5: The context says BBB is disconnected from CCC. This explicitly supports an outside relation between CCC and BBB through symmetry reasoning.
Step 6: Apply combination reasoning carefully: the triangle is inside CCC, and CCC has explicitly supported behind and outside relations with BBB, so the triangle inherits these relations with BBB.
Step 7: AAA is inside BBB. This is PP. Apply inverse reasoning: BBB contains AAA.
Step 8: Apply combination reasoning again: the triangle is behind/outside BBB, and BBB contains AAA, so the triangle is behind and outside AAA.
Step 9: AAA contains the big blue circle. Apply combination reasoning: the triangle is behind and outside the big blue circle.
Step 10: Apply contradiction filtering: do not add in front because behind is supported, and do not add inside because outside is supported.

Final answer:
[
"behind",
"outside"
]

Example 2:
Context:
Three blocks, named AAA, BBB and CCC exist. Block BBB is in block CCC. Block AAA with a medium yellow square is disconnected from and to the right of block CCC. Block CCC is under this block. A medium blue square is within block AAA. Block BBB covers a medium yellow square. A medium black square is inside this block.

Question:
Where is the medium yellow square in AAA relative to the medium black square?

Reasoning:
Step 1: Target X is the medium yellow square in AAA. Reference Y is the medium black square.
Step 2: Build the relevant chain: yellow square in AAA -> AAA -> CCC -> BBB -> black square.
Step 3: The yellow square is inside AAA. This is a PP relation.
Step 4: AAA is right of CCC. This is a Dir relation and already in the useful direction.
Step 5: CCC is under AAA. This is a Dir relation in the opposite order, so apply inverse reasoning: AAA is above CCC.
Step 6: AAA is disconnected from CCC. This explicitly supports an outside relation between the two blocks.
Step 7: Apply combination reasoning carefully: the yellow square is inside AAA, and AAA has explicitly supported outside relations with CCC together with directional relations, so the yellow square inherits right, above, and outside relative to CCC.
Step 8: BBB is inside CCC. This is PP. Apply inverse reasoning: CCC contains BBB.
Step 9: The black square is inside BBB. This is PP. Apply inverse reasoning: BBB contains the black square.
Step 10: Apply combination reasoning through CCC and BBB: the yellow square remains right, above, and outside the black square.
Step 11: Apply contradiction filtering: do not output left with right or below with above.

Final answer:
[
"right",
"above",
"outside"
]

Example 3:
Context:
Three blocks, called HHH, LLL and KKK exist in the image. Block KKK is under and to the right-hand side of block LLL. Block LLL has block HHH. A large purple pentagon is within block HHH. Block HHH cover a large red hexagon and a medium purple hexagon. Under the large purple pentagon is the large red hexagon. Under the medium purple hexagon there is the large purple pentagon. Block LLL has a large purple pentagon. A medium red hexagon and a medium grey pentagon are in block KKK. Block KKK has a large grey pentagon.

Question:
What is the position of the large grey pentagon regarding the large purple pentagon in HHH?

Reasoning:
Step 1: Target X is the large grey pentagon. Reference Y is the large purple pentagon in HHH.
Step 2: Build the relevant chain: large grey pentagon -> KKK -> LLL -> HHH -> large purple pentagon in HHH.
Step 3: The large grey pentagon is inside KKK. This is a PP relation.
Step 4: KKK is below LLL and right of LLL. These are Dir relations.
Step 5: Apply combination reasoning: the large grey pentagon is inside KKK, and KKK is below/right of LLL, so the large grey pentagon is below and right of LLL.
Step 6: LLL has HHH. This is PP, meaning LLL contains HHH.
Step 7: Apply combination reasoning: the large grey pentagon is below/right of LLL, and LLL contains HHH, so the large grey pentagon is below and right of HHH.
Step 8: The large purple pentagon in HHH is within HHH. This is PP. Apply inverse reasoning: HHH contains the large purple pentagon.
Step 9: Apply combination reasoning: the large grey pentagon is below/right of HHH, and HHH contains the reference object, so the large grey pentagon is below and right of the reference object.
Step 10: Apply contradiction filtering: do not output above with below or left with right.

Final answer:
[
"right",
"below"
]

Example 4:
Context:
Three boxes, named DDD, EEE and JJJ exist. Box EEE is at 6:00 position relative to box DDD. This box is at 12:00 position regarding to and in front of box JJJ. In front of this box there is box DDD with a midsize orange rectangle. This box covers a midsize white thing. A midsize green rectangle and another midsize orange rectangle are within this box. At 6:00 position regarding to midsize orange rectangle number two is the white shape. At 6 o'clock position regarding to midsize orange rectangle number one is midsize orange rectangle number two. Midsize orange rectangle number one is at 6:00 position relative to the midsize green rectangle. A midsize orange rectangle is within box JJJ. Box JJJ have a midsize green rectangle and another midsize orange rectangle.

Question:
Where is the midsize green rectangle in JJJ relative to the midsize green rectangle in DDD?

Reasoning:
Step 1: Target X is the midsize green rectangle in JJJ. Reference Y is the midsize green rectangle in DDD.
Step 2: Build the relevant chain: green rectangle in JJJ -> JJJ -> EEE -> DDD -> green rectangle in DDD.
Step 3: The target rectangle is inside JJJ. This is a PP relation.
Step 4: EEE is above JJJ and in front of JJJ. Since the reasoning chain requires relations from JJJ toward EEE, apply inverse reasoning carefully: JJJ is below and behind EEE.
Step 5: Apply combination reasoning: the target rectangle is inside JJJ, and JJJ is below/behind EEE, so the target rectangle is below and behind EEE.
Step 6: DDD is in front of EEE. This is a Dir relation in the opposite order, so apply inverse reasoning: EEE is behind DDD.
Step 7: EEE is below DDD. This is a Dir relation already in the useful direction.
Step 8: Apply transitivity reasoning for repeated Dir relations: target is behind EEE and EEE is behind DDD, so target is behind DDD. Target is below EEE and EEE is below DDD, so target is below DDD.
Step 9: The reference rectangle is inside DDD. This is PP. Apply inverse reasoning: DDD contains the reference rectangle.
Step 10: Apply combination reasoning: target is below/behind DDD, and DDD contains the reference object, so target is below and behind the reference object.
Step 11: Apply contradiction filtering: do not output above with below or in front with behind.

Final answer:
[
"below",
"behind"
]

Example 5:
Context:
A small green watermelon and a big orange melon are within a box called one. Box one contains a medium orange melon. Another box named two with a medium orange melon has this box. Disconnected from this box is another box named three with a small orange melon and a medium yellow apple. Box two is far from this box. A small yellow apple is covered by box three.

Question:
Where is the medium yellow apple regarding the big orange melon?

Reasoning:
Step 1: Target X is the medium yellow apple. Reference Y is the big orange melon.
Step 2: Build the relevant chain: medium yellow apple -> box three -> box two -> box one -> big orange melon.
Step 3: The target apple is inside box three. This is a PP relation.
Step 4: Box two is far from box three. This is a Dis relation, so apply symmetry reasoning: box three is far from box two.
Step 5: Box three is disconnected from box two. This explicitly supports an outside relation between the two boxes.
Step 6: Apply combination reasoning carefully: the target apple is inside box three, and box three has explicitly supported far and outside relations with box two, so the target apple inherits these relations with box two.
Step 7: Box two has box one. This is PP, meaning box two contains box one.
Step 8: Apply combination reasoning: the target apple is far/outside box two, and box two contains box one, so the target apple is far from and outside box one.
Step 9: The big orange melon is inside box one. This is PP. Apply inverse reasoning: box one contains the big orange melon.
Step 10: Apply combination reasoning: the target apple is far/outside box one, and box one contains the reference object, so the target apple is far from and outside the reference object.
Step 11: Apply contradiction filtering: do not add near because far is supported; do not add inside because outside is supported.

Final answer:
[
"far",
"outside"
]
"""


# ============================================================
# Agent 2: rule-bank prompt construction
# ============================================================

class RuleBankPromptAgent:
    """
    Builds the prompt used by the reasoning LLM.
    This agent selects the correct rule bank from the router output.
    """

    def build_messages(self, profile: TaskProfile, context: str, question: str) -> List[Dict[str, str]]:
        if profile.task_name == "directional_frame":
            user_content = f"Context: {context} Question: {question}"
            return (
                [{"role": "system", "content": FRAME_RULE_BANK}]
                + list(FRAME_FEW_SHOTS)
                + [{"role": "user", "content": user_content}]
            )

        raise ValueError("build_messages is only used for chat-style directional prompts.")

    def build_prompt(self, profile: TaskProfile, context: str, question: str) -> str:
        if profile.task_name == "compositional_multilabel":
            return f"""
You are an expert assistant with knowledge of spatial relations and rules to compose them under the assumptions that contexts are provided from a fixed orientation point of view, objects/entities are treated as extended objects, directions are quantitatively unspecified, and relations may be incomplete.

Your task is to identify the subset of entities from the context that are relevant, combine their spatial relations using valid composition rules, and infer the spatial relations between the entities asked about in the question.

The list of all possible spatial relations is:
left, right, above, below, behind, in front, near, far, outside, outside and touching, partially overlapping, inside and touching, inside, contains and touches, contains, overlapping.

Always provide the final answer only in terms of these spatial relations. Include all spatial relations that hold true when multiple labels are correct.

Use this rule checklist internally:
{COMPOSITIONAL_RULE_BANK}

Below are worked examples.

{COMPOSITIONAL_FEW_SHOT_BLOCK}

Now solve the new question using the same style as the worked examples.

Important output rules for the new question:
- Do not write <think> or </think>.
- Do not write meta-comments such as "The user wants me to solve" or "I need to analyze".
- Do not list all entities in the scene.
- Do not repeat the full context.
- Do not compare with previous examples.
- Do not write uncertainty loops such as "Wait", "Let's double check", "One more check", or "Maybe".
- Write between 6 and 10 reasoning steps.
- Each step must either identify the target/reference, build the chain, apply inverse/symmetry/transitivity/combination, or filter contradictions.
- After Step 10, stop reasoning and write the final answer.
- The final answer must be a valid JSON list using only allowed labels.

Context:
{context}

Question:
{question}

Reasoning:
Step 1:
""".strip()

        raise ValueError("build_prompt is only used for plain-text compositional prompts.")

    def to_model_prompt(
        self,
        tokenizer: AutoTokenizer,
        profile: TaskProfile,
        context: str,
        question: str,
    ) -> str:
        if profile.task_name == "directional_frame":
            messages = self.build_messages(profile, context, question)
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        return self.build_prompt(profile, context, question)


# ============================================================
# Agent 3: disambiguation / parsing / repair
# ============================================================

FRAME_ANSWER_RE = re.compile(r"(?is)\banswer\s*:\s*(front|back|left|right)\b")
FRAME_LABEL_RE = re.compile(r"\b(front|back|left|right)\b", re.IGNORECASE)


def sort_compositional_labels(labels: Sequence[str]) -> List[str]:
    s = set(labels)
    return [x for x in COMPOSITIONAL_LABELS if x in s]


class DisambiguationRepairAgent:
    """
    Parses and validates output. In the compositional setting, it can also
    create a repair prompt if the generated answer is empty or malformed.
    """

    def normalize_directional(self, x: Any) -> Optional[str]:
        if x is None:
            return None
        t = str(x).strip().lower()
        if t in DIRECTION_LABELS:
            return t
        if t == "behind":
            return "back"
        if t == "forward":
            return "front"
        return None

    def parse_directional_answer(self, text: str) -> str:
        t = (text or "").strip()

        matches = list(FRAME_ANSWER_RE.finditer(t))
        if matches:
            return matches[0].group(1).lower()

        labels = FRAME_LABEL_RE.findall(t)
        if labels:
            return labels[-1].lower()

        low = t.lower()
        if "behind" in low:
            return "back"
        if "forward" in low:
            return "front"

        return "unknown"

    def gold_directional_set(self, example: Dict[str, Any]) -> Set[str]:
        ca = example.get("candidate_answer")

        if isinstance(ca, list):
            out = {self.normalize_directional(v) for v in ca}
            return {v for v in out if v in DIRECTION_LABELS}

        if isinstance(ca, str):
            v = self.normalize_directional(ca)
            return {v} if v in DIRECTION_LABELS else set()

        return set()

    def remove_direct_opposites(self, labels: Sequence[str]) -> List[str]:
        labels = set(labels)

        opposite_pairs = [
            ("left", "right"),
            ("above", "below"),
            ("behind", "in front"),
            ("inside", "contains"),
            ("inside and touching", "contains and touches"),
        ]

        for a, b in opposite_pairs:
            if a in labels and b in labels:
                # Keep unchanged for fair evaluation; the LLM repair may extract
                # a cleaner final list if the answer is malformed.
                return sort_compositional_labels(labels)

        return sort_compositional_labels(labels)

    def parse_compositional_answer(self, raw: str) -> List[str]:
        raw = raw or ""
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE)

        matches = re.findall(r"\[.*?\]", raw, flags=re.DOTALL)

        for m in reversed(matches):
            try:
                arr = json.loads(m)
            except Exception:
                continue

            if not isinstance(arr, list):
                continue

            cleaned = []
            for x in arr:
                if isinstance(x, str):
                    x = x.strip().lower()
                    if x in COMPOSITIONAL_LABELS:
                        cleaned.append(x)

            if cleaned:
                return self.remove_direct_opposites(sort_compositional_labels(cleaned))

        return []

    def build_repair_prompt(self, raw_answer: str) -> str:
        return f"""
You are the disambiguation and answer-repair agent.

Extract the final answer from the model output.

Allowed labels:
{COMPOSITIONAL_LABELS}

Model output:
{raw_answer}

Return only a valid JSON list using allowed labels.
Do not explain.
Do not add labels that are not clearly in the final answer.

JSON:
""".strip()


# ============================================================
# Orchestration wrapper
# ============================================================

class AgenticSpatialOrchestrator:
    """
    Fixed 3-agent workflow:
    1. ContextRouterAgent selects the reasoning profile / rule bank.
    2. RuleBankPromptAgent builds the LLM prompt from the rule bank.
    3. DisambiguationRepairAgent parses and validates the answer.
    """

    def __init__(self, task: str):
        self.router_agent = ContextRouterAgent()
        self.prompt_agent = RuleBankPromptAgent()
        self.repair_agent = DisambiguationRepairAgent()
        self.profile = self.router_agent.route(task)

    def build_prompt(
        self,
        tokenizer: Optional[AutoTokenizer],
        context: str,
        question: str,
    ) -> str:
        if self.profile.task_name == "directional_frame":
            if tokenizer is None:
                raise ValueError("tokenizer is required for directional chat prompts.")
            return self.prompt_agent.to_model_prompt(
                tokenizer=tokenizer,
                profile=self.profile,
                context=context,
                question=question,
            )

        return self.prompt_agent.build_prompt(
            profile=self.profile,
            context=context,
            question=question,
        )

    def parse_answer(self, raw: str) -> Any:
        if self.profile.task_name == "directional_frame":
            return self.repair_agent.parse_directional_answer(raw)

        return self.repair_agent.parse_compositional_answer(raw)

    def build_repair_prompt(self, raw_answer: str) -> str:
        return self.repair_agent.build_repair_prompt(raw_answer)

    def metadata(self) -> Dict[str, Any]:
        return {
            "agentic_workflow": {
                "agents": [
                    "ContextRouterAgent",
                    "RuleBankPromptAgent",
                    "DisambiguationRepairAgent",
                ],
                "router_selected_task": self.profile.task_name,
                "allowed_labels": self.profile.allowed_labels,
                "answer_format": self.profile.answer_format,
                "multilabel": self.profile.multilabel,
            }
        }
