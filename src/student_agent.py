
from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


@dataclass
class StudentProfile:
    """Who the simulated student is for one session."""
    # which bank problem they work on
    question_id: str  
    # a catalogue ID they hold, or None (clean)                     
    misconception_id: str | None = None   
     # human-readable belief (filled from catalog) 
    misconception_text: str = "" 
    # weak, average or strong         
    knowledge: str = "average" 
    # tutor pushes needed before they reconsider            
    stubbornness: int = 2 
    # concise, chatty                 
    style: str = "concise"                 
    label: str = ""               

    def is_clean(self) -> bool:
        return self.misconception_id is None


SOLVED_TAG = "[SOLVED]"

_PERSONA = """You are role-playing a university student in an algorithms course
(MIT 6.006), working through one problem with a Socratic tutor who asks questions
and gives hints but will not hand over the answer.

THE PROBLEM:
{question}

Your knowledge level: {knowledge}. {knowledge_note}

{belief_block}

How to behave:
- Reply as a real student would: attempt the reasoning, think aloud, and ask a
  question when genuinely unsure. Keep it {style} — usually 1-3 sentences.
- Commit to a position. Take a clear stance and reason from it rather than hedging
  or listing every possibility; a real student commits first and is corrected later.
- Stay in character. Never say you are an AI, never mention "role-play", and never
  use the word "misconception".
- Do the reasoning yourself; do not simply restate or agree with the tutor's words.
- Only end a message with the tag {solved_tag} (on its own line) once you can state
  the full solution in your own words — not merely because the tutor sounds satisfied
  or you feel generally reassured.
- Otherwise, keep working. Do not give up, and do not conclude prematurely."""

_BELIEF_HELD = """You are firmly convinced of the following idea and believe it is CORRECT:
"{belief}"

This idea is in fact incorrect, but you are unaware of that, and nothing about it seems
wrong to you. Your task is to reason as though it were true, in the way a student who
genuinely held it would:
- Apply it early: within your first substantive reply, use this belief to work on the
  problem, so that your conclusion clearly follows from it. Do not leave it implicit.
- Do not independently arrive at the correct answer. If you find yourself reasoning
  toward the right answer, pause and apply your belief again instead — resisting that
  tendency is the most important part of your role.
- Maintain the belief under mild pressure. A leading question, a hint, or the tutor
  restating the goal should not change your mind. Abandon it only when the tutor
  provides a concrete counter-example, or works through a specific case in which the
  belief clearly fails, and only after at least {stubbornness} such distinct challenges.
- When you do give it up, respond as a real student would: acknowledge the specific case
  that changed your mind, then reason forward correctly from that point.
- Remain believable: engage with each question honestly, but always through the lens of
  this belief. Do not state it mechanically, and do not lecture — allow the incorrect
  conclusion to emerge naturally from applying it."""

_BELIEF_CLEAN = """You do not hold any significant misconception about this problem. You
reason soundly, but, like a typical student, you may begin with an incomplete answer,
make a minor error, or need a small prompt before reaching the full solution; you rarely
produce the complete correct answer on your very first reply. Do not invent an incorrect
belief you do not hold, and do not fail deliberately — simply work through the problem as
a capable but imperfect student would."""

_KNOWLEDGE_NOTE = {
    "weak": "Your grasp of the fundamentals is limited: you sometimes misremember "
            "definitions, need ideas broken into small steps, and make errors you do "
            "not notice until they are pointed out.",
    "average": "You understand the basic concepts but must work to connect them, and you "
               "occasionally overlook details.",
    "strong": "You reason quickly and accurately once pointed in the right direction, and "
              "rarely need more than a small hint.",
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
