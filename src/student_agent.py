"""Simulated student agents for testing the tutor end-to-end.

A student agent is NOT a graph node. It sits OUTSIDE the compiled tutor graph and
plays the human role, generating replies turn by turn so a whole conversation can
run unattended. Its value for THIS system is ground truth: give a student a known
misconception (a catalogue ID) and you know exactly what the detector *should*
fire — so the injected belief becomes the label for scoring detection.

Two profile kinds:
  * a student HOLDING a specific misconception  -> tests detection recall
  * a "clean" student holding none              -> tests detection false-positives

Usage is in run_student_eval.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


@dataclass
class StudentProfile:
    """Who the simulated student is for one session."""

    question_id: str                       # which bank problem they work on
    misconception_id: str | None = None    # a catalogue ID they hold, or None (clean)
    misconception_text: str = ""           # human-readable belief (filled from catalog)
    knowledge: str = "average"             # weak | average | strong
    stubbornness: int = 2                  # tutor pushes needed before they reconsider
    style: str = "concise"                 # concise | chatty
    label: str = ""                        # short name for the scorecard

    def is_clean(self) -> bool:
        return self.misconception_id is None


SOLVED_TAG = "[SOLVED]"

_PERSONA = """You are role-playing a university student in an algorithms course
(MIT 6.006). You are working through ONE problem with a Socratic tutor who asks
you questions and gives hints but will not hand over the answer.

THE PROBLEM:
{question}

Your knowledge level: {knowledge}. {knowledge_note}

{belief_block}

How to behave:
- Reply the way a real student would: attempt the reasoning, think out loud,
  ask a question when unsure. Keep it {style} — usually 1-3 sentences.
- Stay in character. NEVER say you are an AI, never mention "role-play", and
  never use the word "misconception".
- Do the thinking yourself; don't just echo the tutor's words back.
- When you genuinely feel you now understand and could finish the problem,
  end your message with the tag {solved_tag} on its own.
- Otherwise keep working. Do not give up prematurely."""

_BELIEF_HELD = """You are firmly convinced of this specific idea and believe it is CORRECT:
"{belief}"

This idea is actually WRONG, but you do not know that. Your job is to reason AS IF it
were true, the way a real student who held it would:
- Bring it into your reasoning early, in your own words, and ACT on it when you work
  through the problem, so your conclusions follow from it.
- Do NOT independently reach the correct answer. If you feel yourself starting to reason
  toward the right answer, STOP and re-assert your belief instead. Resisting that pull is
  the most important part of your role.
- Give the belief up ONLY if the tutor confronts it with a CONCRETE counter-example or
  directly shows why it fails — not merely a leading question or gentle hint — and only
  after they have pushed on it at least {stubbornness} time(s).
- Stay a believable student: engage genuinely, but always through the lens of this belief.
  Don't blurt it out mechanically on turn one and don't lecture — let the wrong conclusion
  emerge from applying it."""

_BELIEF_CLEAN = """You do not hold any deep misconception about this problem. You
reason soundly, though you may still be unsure or take small missteps a normal
student would. Don't manufacture a wrong belief you don't have."""

_KNOWLEDGE_NOTE = {
    "weak": "You often feel unsure and need to build up the idea step by step.",
    "average": "You know the basics but have to work to connect them.",
    "strong": "You pick things up quickly once pointed in the right direction.",
}


class StudentAgent:
    """An LLM role-playing a student, from a StudentProfile.

    Maintains its own view of the dialogue (from the student's side the tutor is
    the 'user'), so each `reply()` is a natural continuation in character.
    """

    def __init__(self, llm, profile: StudentProfile, question_text: str):
        self.llm = llm
        self.profile = profile
        self.solved = False

        if profile.is_clean():
            belief_block = _BELIEF_CLEAN
        else:
            belief_block = _BELIEF_HELD.format(
                belief=profile.misconception_text or "(unspecified)",
                stubbornness=profile.stubbornness,
            )

        self.system = _PERSONA.format(
            question=question_text,
            knowledge=profile.knowledge,
            knowledge_note=_KNOWLEDGE_NOTE.get(profile.knowledge, ""),
            belief_block=belief_block,
            style=profile.style,
            solved_tag=SOLVED_TAG,
        )
        # From the student's POV: tutor turns are 'user', student turns 'assistant'.
        self._turns: list = []

    def reply(self, tutor_message: str) -> str:
        """Produce the next student utterance given the tutor's latest message."""
        messages = [SystemMessage(content=self.system)]
        for role, text in self._turns:
            messages.append(
                HumanMessage(content=text) if role == "tutor"
                else AIMessage(content=text)
            )
        messages.append(HumanMessage(content=tutor_message))

        raw = self.llm.invoke(messages).content.strip()

        if SOLVED_TAG in raw:
            self.solved = True
            raw = raw.replace(SOLVED_TAG, "").strip() or "I think I've got it now, thanks."

        self._turns.append(("tutor", tutor_message))
        self._turns.append(("student", raw))
        return raw
