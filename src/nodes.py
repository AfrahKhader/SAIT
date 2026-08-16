"""Agent components: the shared state and every graph node.

Each node is a callable class holding its own dependencies (LLM, prompt), which
is all LangGraph needs — it accepts any callable as a node.
"""

from __future__ import annotations

import json
import operator
import re
from typing import Annotated, Sequence

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


# --------------------------------------------------------------------------
# Multimodal content helpers
# --------------------------------------------------------------------------
# A message's content can be a plain string OR a list of parts (text + images),
# e.g. [{"type": "text", "text": "..."},
#       {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}]
# These helpers read either shape safely.
def message_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ).strip()
    return ""


def message_has_image(content) -> bool:
    return isinstance(content, list) and any(
        isinstance(p, dict) and p.get("type") in ("image_url", "image")
        for p in content
    )


# --------------------------------------------------------------------------
# Shared state
# --------------------------------------------------------------------------
class AgentState(TypedDict):
    # `add_messages` appends rather than overwrites — needed for the checkpointer.
    messages: Annotated[Sequence[BaseMessage], add_messages]
    guard_verdict: str  # set by GuardrailNode, read by the routing in edges.py
    scaffold_level: int  # 0 = most open (Probe) ... 3 = Tell. Set by AssessNode.
    attempts: int        # consecutive unsuccessful attempts at the current point
    last_verdict: str    # AssessNode's verdict for THIS turn; read by MisconceptionNode

    # --- misconception detection (set by MisconceptionNode) ---
    current_question_id: str          # which bank problem is in play ("" = unknown)
    detected_misconceptions: list      # IDs fired THIS turn; overwritten each turn
    # Accumulates across the session, so a repeated misconception is visible.
    # `operator.add` concatenates instead of overwriting, like `add_messages`.
    misconception_history: Annotated[list, operator.add]


# --------------------------------------------------------------------------
# Guardrail
# --------------------------------------------------------------------------
VALID_VERDICTS = {"ON_TOPIC", "OFF_TOPIC", "INJECTION", "EMPTY"}


class GuardrailNode:
    """Classifies the latest student input; stamps `guard_verdict` onto state."""

    def __init__(self, llm, system_prompt: str, min_length: int = 3):
        self.llm = llm
        self.system_prompt = system_prompt
        self.min_length = min_length

    def __call__(self, state: AgentState) -> dict:
        content = self._latest_human_content(state)
        text = message_text(content)

        # A sketch is a legitimate attempt at the problem — let image-bearing
        # messages through to the tutor (which can actually see the drawing).
        if message_has_image(content):
            verdict = "ON_TOPIC"
            print(f"[guard] {verdict} (sketch)")
            return {"guard_verdict": verdict}

        verdict = self._trivial_check(text)            # cheap, no LLM
        if verdict is None:                            # only pay when unsure
            verdict = (
                self.llm.invoke(
                    [
                        SystemMessage(content=self.system_prompt),
                        HumanMessage(content=text),
                    ]
                )
                .content.strip()
                .upper()
            )
            if verdict not in VALID_VERDICTS:
                verdict = "ON_TOPIC"                    # fail open

        print(f"[guard] {verdict}")
        return {"guard_verdict": verdict}

    @staticmethod
    def _latest_human_content(state: AgentState):
        for m in reversed(state["messages"]):
            if isinstance(m, HumanMessage):
                return m.content
        return ""

    def _trivial_check(self, text: str) -> str | None:
        stripped = text.strip()
        if not stripped or len(stripped) < self.min_length:
            return "EMPTY"
        if not re.search(r"[a-zA-Z]", stripped):
            return "EMPTY"
        if " " not in stripped and not re.search(r"[aeiouAEIOU]", stripped):
            return "EMPTY"
        return None


# --- Scaffolding: assessment node + support-level ladder ---
LEVEL_GUIDANCE = {
    0: "SUPPORT LEVEL 0 (least support): Ask ONE broad, open, conceptual question "
       "that makes the student engage with the underlying idea. Give no hints.",
    1: "SUPPORT LEVEL 1: Narrow the student's attention to the relevant idea with a "
       "focused question. Still do not give the answer.",
    2: "SUPPORT LEVEL 2: Break the problem into a smaller sub-question, or give a "
       "partial scaffold (e.g. state one fact, then ask them to take the next step).",
    3: "SUPPORT LEVEL 3 (most support): The student is stuck. Reveal ONLY the single "
       "next step, then immediately ask a question that moves them forward. Never "
       "dump the whole solution.",
}

ASSESS_VERDICTS = {"CORRECT", "PARTIAL", "STUCK", "META"}


class AssessNode:
    """Reads the student's latest reply against the tutor's last question and
    updates the scaffolding level: fade on success, climb on struggle, and allow
    a Tell after repeated failure."""

    def __init__(self, llm):
        self.llm = llm

    def __call__(self, state: AgentState) -> dict:
        messages = state["messages"]
        level = state.get("scaffold_level", 0)
        attempts = state.get("attempts", 0)

        student_content = self._last_human(messages)
        tutor_question = self._last_tutor_question(messages)

        if not tutor_question or message_has_image(student_content):
            print(f"[assess] SKIP -> level {level}, attempts {attempts}")
            return {"scaffold_level": level, "attempts": attempts, "last_verdict": ""}

        verdict = self._classify(tutor_question, message_text(student_content))
        level, attempts = self._update(level, attempts, verdict)
        print(f"[assess] {verdict} -> level {level}, attempts {attempts}")
        return {"scaffold_level": level, "attempts": attempts, "last_verdict": verdict}

    @staticmethod
    def _last_human(messages):
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                return m.content
        return ""

    @staticmethod
    def _last_tutor_question(messages):
        for m in reversed(messages):
            if isinstance(m, AIMessage) and not m.tool_calls and m.content:
                return m.content
        return ""

    def _classify(self, question: str, answer: str) -> str:
        prompt = (
            f'A tutor asked: "{question}"\n'
            f'The student replied: "{answer}"\n\n'
            "Classify the student's reply as EXACTLY one word:\n"
            "CORRECT - correct and complete for what was asked\n"
            "PARTIAL - on the right track but incomplete or slightly off\n"
            "STUCK   - wrong, confused, or says they don't know\n"
            "META    - not an attempt (a question back, or an instruction like "
            "'simpler please')\n"
            "Reply with one word only."
        )
        verdict = self.llm.invoke([HumanMessage(content=prompt)]).content.strip().upper()
        return verdict if verdict in ASSESS_VERDICTS else "PARTIAL"

    @staticmethod
    def _update(level: int, attempts: int, verdict: str):
        if verdict == "CORRECT":
            level, attempts = max(0, level - 1), 0
        elif verdict == "STUCK":
            level, attempts = min(3, level + 1), attempts + 1
        elif verdict == "PARTIAL":
            attempts += 1
        if attempts >= 3:
            level, attempts = 3, 0
        return level, attempts

# --------------------------------------------------------------------------
# Misconception detection
# --------------------------------------------------------------------------
# Fixed-list matching: the detector is shown ONLY the catalogued misconceptions
# for the question in play, and may return a subset of those IDs or nothing.
# Two guards against the classic failure of list-matching — a model handed a
# numbered list will find *something* to match:
#   1. "no misconception" is an explicit, legitimate answer (an empty list).
#   2. every match must carry a verbatim quote from the student's own words.
#      A match whose quote is absent from the reply is dropped as confabulated.

DETECT_PROMPT = """You are grading one step of a student's algorithms work.

QUESTION:
{question}

REFERENCE SOLUTION:
{solution}

THE STUDENT WROTE:
"{student}"

Below is the complete list of known misconceptions for THIS question. You may
only choose from this list. Do not invent new IDs, and do not choose one that is
merely related — the student's own words must show the error.

{candidates}

Flag a misconception ONLY if the student is currently COMMITTING to that mistaken
belief. Do NOT flag it if they are rejecting it, questioning it, or correcting a
mistake they previously made (e.g. "I first thought X, but actually Y" — here X is
not a misconception, it is being discarded). If the student's statement is
consistent with the REFERENCE SOLUTION above, it is not a misconception. Merely
mentioning a concept, term, or structure correctly is never a misconception.

Most student replies contain NO misconception from this list: they may be
correct, partly correct, or simply a question back to the tutor. Returning an
empty list is the expected answer in those cases and is never penalised.

Reply with JSON only, no prose:
{{"matched": [{{"id": "<an ID from the list above>",
               "evidence": "<a short VERBATIM quote where the student COMMITS to the error>",
               "confidence": <0.0-1.0>}}]}}

If nothing in the list applies, reply exactly: {{"matched": []}}"""


# Open-ended mode: no staff list exists for this question, so the agent must
# infer a misconception from the reference solution. This has no fixed list to
# constrain it, so it is the most false-positive-prone path — the guards (commit
# not mention/reject, consistency with the solution, verbatim evidence, a higher
# confidence floor) do the heavy lifting here.
DETECT_OPEN_PROMPT = """You are grading one step of a student's algorithms work.

QUESTION:
{question}

REFERENCE SOLUTION:
{solution}

THE STUDENT WROTE:
"{student}"

Does the student's reply reveal a genuine conceptual misconception — a mistaken
belief about how the problem or the underlying idea works? Judge ONLY against the
reference solution above.

Flag a misconception ONLY if the student is currently COMMITTING to a mistaken
belief. Do NOT flag: a reply consistent with the reference solution; an incomplete
but correct step; a question back to the tutor; or a belief the student is
rejecting or correcting ("I first thought X, but actually Y"). If in doubt, do not
flag — a missed misconception is far cheaper than a false accusation.

If you do flag one, describe it in ONE short phrase naming the specific mistaken
belief (e.g. "treats n repeated O(1) operations as amortized O(n)"), not a vague
"is confused".

Reply with JSON only, no prose:
{{"misconception": "<short phrase, or null if none>",
  "evidence": "<a short VERBATIM quote where the student COMMITS to the error, or ''>",
  "confidence": <0.0-1.0>}}"""


class MisconceptionNode:
    """Matches the student's latest reply against the fixed misconception list
    for the problem in play, and records which (if any) fired.

    Skips cheaply — costing no LLM call — whenever detection cannot apply: no
    catalog, a sketch, no student attempt yet, or a question with no list.
    """

    def __init__(self, llm, catalog, min_confidence: float = 0.6,
                 open_min_confidence: float = 0.7):
        self.llm = llm
        self.catalog = catalog
        self.min_confidence = min_confidence
        # Open-ended detection has no fixed list to anchor it, so it demands a
        # higher confidence floor than fixed-list matching.
        self.open_min_confidence = open_min_confidence

    def __call__(self, state: AgentState) -> dict:
        unchanged = {
            "current_question_id": state.get("current_question_id", ""),
            "detected_misconceptions": [],
        }

        if self.catalog is None:
            return unchanged

        student_content = self._last_human(state["messages"])
        student = message_text(student_content)

        # A sketch needs the tutor's vision, not text matching.
        if message_has_image(student_content) or not student.strip():
            print("[misconception] SKIP (no text attempt)")
            return unchanged

        # Don't diagnose errors in a turn AssessNode already judged correct, or
        # in a non-attempt (a question back). This removes the main false-positive
        # source: a correct answer that merely *mentions* a concept lexically
        # overlapping a catalogued misconception. PARTIAL/STUCK still run — that
        # is where misconceptions actually live.
        verdict = state.get("last_verdict", "")
        if verdict in ("CORRECT", "META"):
            print(f"[misconception] SKIP (verdict {verdict})")
            return {
                "current_question_id": state.get("current_question_id", ""),
                "detected_misconceptions": [],
            }

        question_id = state.get("current_question_id") or self._identify(student, state)
        if not question_id:
            print("[misconception] SKIP (no question identified)")
            return unchanged

        item = self.catalog.question(question_id)
        if not item:
            print(f"[misconception] SKIP ({question_id}: not in bank)")
            return {"current_question_id": question_id, "detected_misconceptions": []}

        # Hybrid: fixed-list where a staff catalogue exists, open-ended otherwise.
        if self.catalog.has_catalog(question_id):
            candidates = self.catalog.for_question(question_id)
            matched = self._detect(item, student, candidates)
            mode = "list"
        else:
            matched = self._detect_open(item, student)
            mode = "open"

        if matched:
            shown = ", ".join(m if len(m) < 60 else m[:57] + "…" for m in matched)
            print(f"[misconception] {question_id} ({mode}) -> {shown}")
        else:
            print(f"[misconception] {question_id} ({mode}) -> none")

        return {
            "current_question_id": question_id,
            "detected_misconceptions": matched,
            "misconception_history": matched,
        }

    # -- internals --------------------------------------------------------

    @staticmethod
    def _last_human(messages):
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                return m.content
        return ""

    def _identify(self, student: str, state: AgentState) -> str:
        """Work out which bank problem the student is on, once per session.

        The result is cached in `current_question_id`, so this costs one LLM
        call per problem rather than one per turn. Note the trade-off: if the
        student moves to a different problem mid-session the cached ID goes
        stale — clear the field when you switch problems.
        """
        menu = self.catalog.identification_menu()
        if not menu:
            return ""

        # Give the model the opening exchange, where the problem is usually
        # stated, rather than only the latest reply ("yes, n log n").
        opening = " ".join(
            message_text(m.content)
            for m in list(state["messages"])[:4]
        )[:1500]

        prompt = (
            "Which of these problems is the student working on?\n\n"
            f"{menu}\n\n"
            f"Conversation so far: \"{opening}\"\n"
            f"Student's latest message: \"{student}\"\n\n"
            "Reply with the question ID alone (e.g. Q5-d). If the conversation "
            "does not clearly match one of them, reply exactly: NONE"
        )
        try:
            answer = self.llm.invoke([HumanMessage(content=prompt)]).content.strip()
        except Exception as e:
            print(f"[misconception] identify failed: {e}")
            return ""

        answer = answer.split()[0].strip(".,:\"'") if answer else "NONE"
        if answer in self.catalog.usable_question_ids():
            print(f"[misconception] identified question {answer}")
            return answer
        return ""

    def _detect(self, item: dict, student: str, candidates) -> list:
        """Run the fixed-list match. Fails closed: any error means no detection,
        so a parsing problem can never fabricate an accusation."""
        valid = {m.misconception_id for m in candidates}
        prompt = DETECT_PROMPT.format(
            question=item["question"],
            solution=item.get("solution") or "(not available)",
            student=student,
            candidates="\n".join(m.as_option() for m in candidates),
        )

        try:
            raw = self.llm.bind(
                response_format={"type": "json_object"}
            ).invoke([HumanMessage(content=prompt)]).content
            payload = json.loads(raw)
        except Exception as e:
            print(f"[misconception] detection failed: {e}")
            return []

        student_lower = student.lower()
        matched = []
        for hit in payload.get("matched", []) or []:
            if not isinstance(hit, dict):
                continue
            mid = hit.get("id")
            if mid not in valid or mid in matched:
                continue  # hallucinated or duplicate ID
            if float(hit.get("confidence", 0) or 0) < self.min_confidence:
                continue
            # The evidence must really be in the student's words. This is the
            # single most effective filter against an eager match.
            evidence = (hit.get("evidence") or "").strip().strip('"').lower()
            if not evidence or evidence not in student_lower:
                print(f"[misconception] dropped {mid} (evidence not in reply)")
                continue
            matched.append(mid)
        return matched

    def _detect_open(self, item: dict, student: str) -> list:
        """Infer a misconception when no staff list exists. Returns a one-element
        list holding a short freeform description, or [] for none.

        Same guards as fixed-list mode: needs a real solution to ground on, a
        verbatim evidence quote, and a (higher) confidence floor. Fails closed."""
        if not self.catalog._has_usable_solution(item):
            return []   # nothing solid to ground an inference on

        prompt = DETECT_OPEN_PROMPT.format(
            question=item["question"],
            solution=item["solution"],
            student=student,
        )
        try:
            raw = self.llm.bind(
                response_format={"type": "json_object"}
            ).invoke([HumanMessage(content=prompt)]).content
            payload = json.loads(raw)
        except Exception as e:
            print(f"[misconception] open detection failed: {e}")
            return []

        desc = (payload.get("misconception") or "").strip()
        if not desc or desc.lower() in ("null", "none"):
            return []
        if float(payload.get("confidence", 0) or 0) < self.open_min_confidence:
            return []
        evidence = (payload.get("evidence") or "").strip().strip('"').lower()
        if not evidence or evidence not in student.lower():
            print("[misconception] dropped open hit (evidence not in reply)")
            return []
        return [desc]


# --------------------------------------------------------------------------
# Tutor
# --------------------------------------------------------------------------
# Detection is only useful if it changes what the tutor says. This keeps the
# response Socratic: surface the contradiction, never announce the diagnosis.
MISCONCEPTION_GUIDANCE = (
    "\n\nLIKELY MISCONCEPTION — the student's last reply suggests they hold "
    "this specific belief:\n{details}\n"
    "Do NOT name it, quote this text, or tell them they are wrong outright. "
    "Ask ONE question, or give ONE small concrete counter-example, that makes "
    "them test that belief themselves and see the contradiction. This detection "
    "may be mistaken, so phrase it so a student who does NOT hold the "
    "misconception can still answer naturally."
)

REPEAT_GUIDANCE = (
    "\nThis is not the first time this misconception has appeared this session. "
    "The previous indirect approach did not land, so be more concrete: give a "
    "specific small case where their belief visibly fails, then ask what it implies."
)


class SocraticNode:
    """The tutor: a tool-bound LLM with the Socratic prompt prepended."""

    def __init__(self, llm, system_prompt: str, catalog=None):
        self.llm = llm  # already bound with tools
        self.system_prompt = system_prompt
        self.catalog = catalog  # optional; only used to describe detected IDs

    def __call__(self, state: AgentState) -> dict:
        level = state.get("scaffold_level", 0)
        system = (
            self.system_prompt
            + "\n\nADAPTIVE SCAFFOLDING — follow this on THIS turn:\n"
            + LEVEL_GUIDANCE.get(level, LEVEL_GUIDANCE[0])
            + self._misconception_block(state)
        )
        messages = [SystemMessage(content=system)] + list(state["messages"])
        return {"messages": [self.llm.invoke(messages)]}

    def _misconception_block(self, state: AgentState) -> str:
        """Extra prompt section describing whatever the detector just found."""
        detected = state.get("detected_misconceptions") or []
        if not detected or self.catalog is None:
            return ""

        details = "\n".join(f"- {self.catalog.describe(mid)}" for mid in detected)
        block = MISCONCEPTION_GUIDANCE.format(details=details)

        # Seen before this session? Then the gentle approach already failed once.
        history = state.get("misconception_history") or []
        if any(history.count(mid) > 1 for mid in detected):
            block += REPEAT_GUIDANCE
        return block


# --------------------------------------------------------------------------
# Tool executor
# --------------------------------------------------------------------------
class RetrieverNode:
    """Executes whatever tool calls the tutor requested."""

    def __init__(self, tools):
        self.tools_dict = {t.name: t for t in tools}

    def __call__(self, state: AgentState) -> dict:
        tool_calls = state["messages"][-1].tool_calls
        results = []
        for t in tool_calls:
            query = t["args"].get("query", "No query provided")
            print(f"Calling Tool: {t['name']} with query: {query}")

            if t["name"] not in self.tools_dict:
                print(f"\nTool: {t['name']} does not exist.")
                result = (
                    "Incorrect Tool Name, Please Retry and Select tool from "
                    "List of Available tools."
                )
            else:
                result = self.tools_dict[t["name"]].invoke(t["args"].get("query", ""))
                print(f"Result length: {len(str(result))}")

            results.append(
                ToolMessage(tool_call_id=t["id"], name=t["name"], content=str(result))
            )

        print("Tools Execution Complete. Back to the model!")
        return {"messages": results}


# --------------------------------------------------------------------------
# Redirect (blocked input)
# --------------------------------------------------------------------------
class RedirectNode:
    """Returns a canned redirect when the guardrail blocks input. No LLM call."""

    def __init__(self, redirects: dict):
        self.redirects = redirects

    def __call__(self, state: AgentState) -> dict:
        verdict = state["guard_verdict"]
        message = self.redirects.get(verdict, self.redirects["OFF_TOPIC"])
        return {"messages": [AIMessage(content=message)]}