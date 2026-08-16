"""Misconception infrastructure: per-question error lists + the full question bank.

This is data plumbing (not an agent component), so the catalog and the question
bank it indexes live together here — the same split as `retrieval.py`.

Hybrid strategy. Every question in the bank is usable by the tutor. For a
question that HAS a staff-authored catalogue, detection is fixed-list: the
detector may only return IDs from that question's own list (citable, exactly
scorable). For a question with NO catalogue, detection is open-ended: the agent
infers a misconception from the reference solution and the student's reply. The
question bank (all 65) is the source for both modes; the catalogue (14 so far)
just upgrades the questions it covers to the stricter fixed-list mode.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from config import Config


@dataclass
class Misconception:
    """One atomic, named error a student can make on one question."""

    misconception_id: str
    question_id: str
    label: str
    description: str
    origin: str  # "staff" = from the official solutions; "generated" = inferred
    topic: str | None = None

    def as_option(self) -> str:
        """One line for the numbered candidate list shown to the detector."""
        return f"{self.misconception_id}: {self.label} — {self.description}"


class MisconceptionCatalog:
    """Owns the misconception catalog and the question bank it refers to."""

    def __init__(self, config: Config):
        self.config = config
        self._by_question: dict[str, list[Misconception]] = {}
        self._questions: dict[str, dict] = {}

    def build(self) -> "MisconceptionCatalog":
        """Load both JSON files. Returns self for chaining, like KnowledgeBase."""
        cfg = self.config.misconceptions

        catalog_path = Path(cfg.catalog_path)
        questions_path = Path(cfg.questions_path)
        for p in (catalog_path, questions_path):
            if not p.exists():
                raise FileNotFoundError(f"Misconception data not found: {p.resolve()}")

        with catalog_path.open("r", encoding="utf-8") as f:
            catalog = json.load(f)
        with questions_path.open("r", encoding="utf-8") as f:
            bank = json.load(f)

        for entry in catalog["misconceptions"]:
            m = Misconception(
                misconception_id=entry["misconception_id"],
                question_id=entry["question_id"],
                label=entry["label"],
                description=entry["description"],
                origin=entry.get("origin", "generated"),
                topic=entry.get("topic"),
            )
            self._by_question.setdefault(m.question_id, []).append(m)

        for topic in bank["topics"]:
            for q in topic["questions"]:
                q = dict(q)
                q["topic"] = topic["topic"]
                self._questions[q["question_id"]] = q

        staff = sum(
            1 for ms in self._by_question.values() for m in ms if m.origin == "staff"
        )
        total = sum(len(ms) for ms in self._by_question.values())
        print(
            f"Loaded misconception catalog: {total} misconceptions "
            f"({staff} staff) over {len(self._by_question)} of "
            f"{len(self._questions)} questions"
        )
        return self

    # -- lookups ----------------------------------------------------------

    def for_question(self, question_id: str) -> list[Misconception]:
        """The fixed candidate list for one question ([] if none catalogued)."""
        return self._by_question.get(question_id, [])

    def has_catalog(self, question_id: str) -> bool:
        """True -> fixed-list detection; False -> open-ended detection."""
        return bool(self._by_question.get(question_id))

    def question(self, question_id: str) -> dict | None:
        """The bank item: question text, reference solution, topic, source."""
        return self._questions.get(question_id)

    @staticmethod
    def _has_usable_solution(q: dict) -> bool:
        """A solution we can actually ground on (open-ended mode needs this).

        Filters out the handful of bank entries whose solution is a placeholder
        or a bare answer with no reasoning (e.g. 'NOT PROVIDED', 'A.').
        """
        sol = (q.get("solution") or "").strip()
        return len(sol) >= 25 and "NOT PROVIDED" not in sol.upper()

    def all_question_ids(self) -> list[str]:
        """Every question in the bank — the detector may fire on any of these."""
        return sorted(self._questions)

    def usable_question_ids(self) -> list[str]:
        """Questions with a solution solid enough to tutor / detect against."""
        return sorted(
            qid for qid, q in self._questions.items() if self._has_usable_solution(q)
        )

    def describe(self, misconception_id: str) -> str:
        """Human-readable text for one ID, for injecting into the tutor prompt."""
        for ms in self._by_question.values():
            for m in ms:
                if m.misconception_id == misconception_id:
                    return f"{m.label} — {m.description}"
        return misconception_id

    def catalogued_question_ids(self) -> list[str]:
        """Questions the detector can actually fire on (i.e. have a fixed list)."""
        return sorted(self._by_question)

    def topics_with_catalog(self) -> dict[str, list[str]]:
        """{topic: [question_id, ...]} for questions that have a catalogue."""
        grouped: dict[str, list[str]] = {}
        for qid in self.catalogued_question_ids():
            q = self._questions.get(qid)
            topic = (q or {}).get("topic") or "Other"
            grouped.setdefault(topic, []).append(qid)
        return grouped

    def topics_all(self) -> dict[str, list[str]]:
        """{topic: [question_id, ...]} over ALL questions with a usable solution.

        Drives the opening topic buttons. Catalogued questions come first within
        each topic, so a topic that has staff misconceptions opens on one of them
        (fixed-list mode) while still exposing every topic in the bank.
        """
        catalogued = set(self.catalogued_question_ids())
        grouped: dict[str, list[str]] = {}
        for qid in self.usable_question_ids():
            topic = (self._questions.get(qid) or {}).get("topic") or "Other"
            grouped.setdefault(topic, []).append(qid)
        # within each topic, catalogued questions first
        for topic, qids in grouped.items():
            qids.sort(key=lambda q: (q not in catalogued, q))
        return grouped

    def starter_for_topic(self, topic: str) -> str | None:
        """Representative question to open a topic: catalogued if any, else usable."""
        qids = self.topics_all().get(topic) or []
        return qids[0] if qids else None

    def identification_menu(self, only_catalogued: bool = False) -> str:
        """Compact question list used to work out which problem is in play.

        Defaults to ALL usable questions, so the tutor can identify any problem in
        the bank — not only the catalogued ones. Pass only_catalogued=True for the
        narrower, higher-precision menu.
        """
        qids = (
            self.catalogued_question_ids() if only_catalogued
            else self.usable_question_ids()
        )
        lines = []
        for qid in qids:
            q = self._questions.get(qid)
            if not q:
                continue
            stem = " ".join(q["question"].split())[:160]
            lines.append(f"{qid} [{q['topic']}]: {stem}")
        return "\n".join(lines)