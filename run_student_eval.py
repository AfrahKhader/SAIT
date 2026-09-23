"""Run simulated students against the tutor and score the misconception detector.

    python run_student_eval.py

For each student profile we drive the compiled tutor graph turn by turn, then
compare what the detector fired against the misconception the student was told to
hold. Because the injected belief is known, catalogued-question sessions give
exact detection labels:

  * held misconception caught      -> recall
  * something else fired           -> spurious (precision hit)
  * clean student triggered anything -> false positive

Open-ended questions are reported too, but their match is fuzzy (no stable ID),
so treat catalogued sessions as the trustworthy numbers.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(ROOT / "data" / ".env", override=True)

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from config import Config
from graph import build_agent
from misconceptions import MisconceptionCatalog
from nodes import message_text
from retrieval import KnowledgeBase
from student_agent import StudentAgent, StudentProfile


# --------------------------------------------------------------------------
# Capture console output (turn-by-turn transcript + node [tag] logs) per session
# --------------------------------------------------------------------------
class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


@contextmanager
def tee_stdout(buffer):
    """Mirror everything printed inside the block into `buffer` too."""
    old = sys.stdout
    sys.stdout = _Tee(old, buffer)
    try:
        yield
    finally:
        sys.stdout = old


# --------------------------------------------------------------------------
# One session
# --------------------------------------------------------------------------
@dataclass
class SessionResult:
    profile: StudentProfile
    fired: list          # cumulative misconception_history at the end
    turns: int
    solved: bool
    scaffold_path: list  # scaffold_level after each turn
    transcript: list     # [(speaker, text)]
    expressed: bool | None = None  # did the student actually commit to the belief?


def run_session(agent, catalog, student_llm, profile, max_turns=8, verbose=True,
                log_buffer=None):
    """Drive one full student<->tutor conversation and collect what fired.

    Node logs ([guard]/[assess]/[misconception]/tool prints) happen during the
    tutor's invoke, so we tee ONLY those calls into `log_buffer`. The STUDENT/
    TUTOR turns are printed outside that tee, so they reach the console (for a
    live interleaved view) but never pollute the logs buffer — keeping the two
    output files cleanly separated.
    """
    item = catalog.question(profile.question_id)
    if not item:
        raise ValueError(f"{profile.question_id} not in bank")

    thread = {"configurable": {"thread_id": f"student-{uuid.uuid4().hex[:8]}"}}
    student = StudentAgent(student_llm, profile, item["question"])

    opening = (
        f"I'd like to work on this problem:\n\n{item['question']}\n\n"
        "Can you help me get started?"
    )
    transcript = [("student", opening)]
    scaffold_path = []

    if verbose:
        print(f"\nSTUDENT: {opening}")
    tutor_msg = _tutor_turn(agent, thread, opening, pin=profile.question_id,
                            log_buffer=log_buffer)
    transcript.append(("tutor", tutor_msg))
    if verbose:
        print(f"TUTOR: {tutor_msg}")

    for _ in range(max_turns):
        reply = student.reply(tutor_msg)
        transcript.append(("student", reply))
        if verbose:
            print(f"\nSTUDENT: {reply}")
        tutor_msg = _tutor_turn(agent, thread, reply, log_buffer=log_buffer)
        transcript.append(("tutor", tutor_msg))
        if verbose:
            print(f"TUTOR: {tutor_msg}")

        scaffold_path.append(agent.get_state(thread).values.get("scaffold_level", 0))
        if student.solved:
            break

    state = agent.get_state(thread).values
    result = SessionResult(
        profile=profile,
        fired=list(state.get("misconception_history", []) or []),
        turns=len([t for t in transcript if t[0] == "student"]),
        solved=student.solved,
        scaffold_path=scaffold_path,
        transcript=transcript,
    )
    if verbose:
        _print_session(result, catalog)
    return result


def _tutor_turn(agent, thread, student_text, pin=None, log_buffer=None):
    payload = {"messages": [HumanMessage(content=student_text)]}
    if pin:
        payload["current_question_id"] = pin
    if log_buffer is not None:
        with tee_stdout(log_buffer):          # capture node [tag] logs here only
            out = agent.invoke(payload, config=thread)
    else:
        out = agent.invoke(payload, config=thread)
    return message_text(out["messages"][-1].content) or "(no reply)"


def student_expressed(judge_llm, profile, transcript) -> bool:
    """Did the student actually COMMIT to the belief they were told to hold?

    This is the guard against a false conclusion: if the student agent reasoned
    correctly and never expressed the misconception (common when the student LLM
    is too capable), a non-detection is NOT a detector miss — the session simply
    didn't test recall, so we exclude it rather than count it against the detector.
    """
    student_turns = "\n".join(t for s, t in transcript if s == "student")
    prompt = (
        "A student was asked to role-play holding this specific MISTAKEN belief:\n"
        f'"{profile.misconception_text}"\n\n'
        "Here is everything the student said:\n"
        f"{student_turns}\n\n"
        "Did the student actually COMMIT to that mistaken belief at any point — i.e. "
        "state it or reason as if it were true — rather than reasoning correctly "
        "throughout? Answer with one word: YES or NO."
    )
    try:
        ans = judge_llm.invoke([HumanMessage(content=prompt)]).content.strip().upper()
    except Exception as e:
        print(f"[validity] judge failed: {e}")
        return True   # fail open: don't silently drop a session on a judge error
    return ans.startswith("YES")


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def score(results, catalog):
    """Aggregate detection quality, splitting catalogued (exact) from open.

    Recall is computed ONLY over sessions where the student actually expressed the
    belief — sessions where they didn't are reported separately as excluded, not
    counted as detector misses.
    """
    held_cat = [r for r in results
                if not r.profile.is_clean() and catalog.has_catalog(r.profile.question_id)]
    clean = [r for r in results if r.profile.is_clean()]
    held_open = [r for r in results
                 if not r.profile.is_clean() and not catalog.has_catalog(r.profile.question_id)]

    valid = [r for r in held_cat if r.expressed]
    excluded = [r for r in held_cat if not r.expressed]

    caught = sum(1 for r in valid if r.profile.misconception_id in r.fired)
    spurious = sum(
        1 for r in valid
        if any(f != r.profile.misconception_id for f in r.fired)
    )
    clean_fired = sum(1 for r in clean if r.fired)

    print("\n" + "=" * 62)
    print("DETECTION SCORECARD")
    print("=" * 62)
    if held_cat:
        print(f"Catalogued sessions   : {len(held_cat)}  "
              f"({len(valid)} expressed the belief, {len(excluded)} did not)")
        if valid:
            print(f"Recall (valid only)   : {caught}/{len(valid)} "
                  f"= {caught/len(valid):.0%}  (held misconception caught)")
            print(f"Spurious on valid     : {spurious}/{len(valid)}  "
                  f"(also fired an ID the student did NOT hold)")
        else:
            print("Recall (valid only)   : n/a — no session expressed its belief")
        if excluded:
            names = ", ".join(r.profile.label or r.profile.question_id for r in excluded)
            print(f"Excluded (never expressed): {len(excluded)}  -> {names}")
            print("   ^ student agent reasoned correctly; rerun with a weaker "
                  "--student-model or a firmer persona to actually test these.")
    if clean:
        print(f"Clean false-positives : {clean_fired}/{len(clean)}  "
              f"(clean student that triggered anything)")
    if held_open:
        fired_any = sum(1 for r in held_open if r.fired)
        print(f"Open-ended fired      : {fired_any}/{len(held_open)}  "
              f"(fuzzy — inspect transcripts, no exact label)")
    solved = sum(1 for r in results if r.solved)
    print(f"Reached resolution    : {solved}/{len(results)}")
    print("=" * 62)


# --------------------------------------------------------------------------
# Pretty printing
# --------------------------------------------------------------------------
def _print_session(r, catalog):
    p = r.profile
    tag = "CLEAN" if p.is_clean() else p.misconception_id
    print(f"\n--- {p.label or p.question_id} | holds: {tag} "
          f"| {'list' if catalog.has_catalog(p.question_id) else 'open'} mode ---")
    print(f"    fired: {r.fired or 'nothing'}")
    print(f"    scaffold: {r.scaffold_path} | turns: {r.turns} | solved: {r.solved}")
    if not p.is_clean() and catalog.has_catalog(p.question_id):
        if r.expressed is False:
            print(f"    -> EXCLUDED: student never expressed the belief (not a detector miss)")
        else:
            hit = "CAUGHT" if p.misconception_id in r.fired else "MISSED"
            print(f"    -> {hit} the held misconception")
    elif p.is_clean():
        print(f"    -> {'FALSE POSITIVE' if r.fired else 'clean, as expected'}")


# --------------------------------------------------------------------------
# Profiles + main
# --------------------------------------------------------------------------
def build_profiles(catalog):
    """A small suite: a few held catalogued misconceptions + clean controls.

    misconception_text is filled from the catalogue so the student expresses the
    exact belief the detector is scored against.
    """
    def held(qid, mid, **kw):
        return StudentProfile(
            question_id=qid, misconception_id=mid,
            misconception_text=catalog.describe(mid),
            label=kw.pop("label", f"{qid}/{mid}"), **kw,
        )

    return [
        held("Q6", "Q6.M2", knowledge="average", label="hash amortized-vs-expected"),
        held("Q2-a", "AMORT-1", knowledge="average", label="repeated O(1) = amortized O(m)"),
        held("Q4-b", "Q4-b.M1", knowledge="weak", label="heap-order vs traversal-order"),
        held("Q5-d", "Q5-d.M2", knowledge="strong", label="mis-simplifying exp bound"),
        # Clean controls — the detector should stay silent on these.
        StudentProfile("Q5-b", None, knowledge="strong", label="clean / lower bounds"),
        StudentProfile("Q7", None, knowledge="average", label="clean / augmentation"),
    ]


def _session_to_dict(r):
    return {
        "label": r.profile.label,
        "question_id": r.profile.question_id,
        "held_misconception": r.profile.misconception_id,
        "expressed_belief": r.expressed,
        "fired": r.fired,
        "turns": r.turns,
        "solved": r.solved,
        "scaffold_path": r.scaffold_path,
        "transcript": [{"speaker": s, "text": t} for s, t in r.transcript],
    }


def _format_conversation(r) -> str:
    """Clean transcript for one session — no logs."""
    head = (f"Session: {r.profile.label or r.profile.question_id}   "
            f"(question {r.profile.question_id}, "
            f"held: {r.profile.misconception_id or 'none / clean'})")
    lines = [head, "-" * len(head), ""]
    for speaker, text in r.transcript:
        lines.append(f"{'STUDENT' if speaker == 'student' else 'TUTOR'}: {text}")
        lines.append("")
    return "\n".join(lines)


def _session_header(r) -> str:
    return (f"Session: {r.profile.label or r.profile.question_id}   "
            f"(question {r.profile.question_id}, "
            f"held: {r.profile.misconception_id or 'none / clean'})")


def main():
    ap = argparse.ArgumentParser(description="Run simulated students against the tutor.")
    ap.add_argument("-n", type=int, default=None,
                    help="run only the first N profiles (cheap smoke run)")
    ap.add_argument("--outdir", default="eval_runs",
                    help="where to write the run folder (default: eval_runs/)")
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--student-model", default="gpt-4o-mini",
                    help="model for the simulated student (default: gpt-4o-mini, "
                         "deliberately weaker than the tutor to role-play error faithfully)")
    ap.add_argument("--student-temp", type=float, default=0.4,
                    help="student temperature (lower = less likely to drift to correct)")
    args = ap.parse_args()

    config = Config.load()
    kb = KnowledgeBase(config).build()
    agent = build_agent(config, kb)
    catalog = MisconceptionCatalog(config).build()

    # A separate, weaker model for the student decorrelates it from the tutor and
    # makes it role-play confusion instead of reasoning its way to the right answer.
    student_llm = ChatOpenAI(model=args.student_model, temperature=args.student_temp)
    # The validity judge should be capable and neutral — use the tutor-grade model.
    judge_llm = ChatOpenAI(model=config.llm.tutor_model, temperature=0)

    profiles = build_profiles(catalog)
    if args.n:
        profiles = profiles[:args.n]

    run_dir = Path(args.outdir) / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Running {len(profiles)} student sessions -> {run_dir}")
    print(f"tutor={config.llm.tutor_model} | student={args.student_model} "
          f"@ T={args.student_temp}")

    results = []
    conversations = []   # clean transcripts
    logs = []            # node [tag] logs only
    for p in profiles:
        log_buffer = io.StringIO()   # captures ONLY the node logs (see _tutor_turn)
        r = run_session(agent, catalog, student_llm, p,
                        max_turns=args.max_turns, log_buffer=log_buffer)
        # Validity: for a held-belief student, did they actually express it?
        if not p.is_clean():
            r.expressed = student_expressed(judge_llm, p, r.transcript)
            print(f"    [validity] expressed belief: {r.expressed}")
        results.append(r)
        conversations.append(_format_conversation(r))
        logs.append(f"{_session_header(r)}\n{'-' * 60}\n{log_buffer.getvalue().strip()}\n")

    sep = "\n\n" + "=" * 70 + "\n\n"

    # FILE 1 — the conversations only.
    (run_dir / "conversation.txt").write_text(sep.join(conversations), encoding="utf-8")
    # FILE 2 — the logs only.
    (run_dir / "logs.txt").write_text(sep.join(logs), encoding="utf-8")

    # Structured results + scorecard as before.
    (run_dir / "results.json").write_text(
        json.dumps([_session_to_dict(r) for r in results], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    card = io.StringIO()
    with tee_stdout(card):
        score(results, catalog)
    (run_dir / "scorecard.txt").write_text(card.getvalue(), encoding="utf-8")

    print(f"\nSaved in {run_dir}:")
    print("  conversation.txt  — the dialogue only")
    print("  logs.txt          — the [guard]/[assess]/[misconception] logs only")
    print("  results.json      — structured per-session data")
    print("  scorecard.txt     — the aggregate detection numbers")


if __name__ == "__main__":
    main()
