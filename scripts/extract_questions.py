#!/usr/bin/env python3
"""
Extract algorithms-course questions from PDFs into one topic-grouped JSON file.
(OpenAI / GPT version, using the Responses API.)

Works on mixed formats (quizzes, exams, problem sets, problem collections). Each
PDF is sent as an input_file; OpenAI feeds the model both the extracted text and
the page images, so it can describe figures (graphs, trees, boards). Output is
forced into a fixed schema via Structured Outputs (strict json_schema), then
grouped by topic.

Usage:
    pip install openai pypdf
    export OPENAI_API_KEY=sk-...

    python extract_questions.py --input ./pdfs --output questions.json --course "MIT 6.006"
    python extract_questions.py --input quiz1.pdf                    # single file
    python extract_questions.py --input ./pdfs --dry-run            # no API key needed; shows plan
"""
import argparse
import base64
import glob
import io
import json
import os
import sys
import time
from collections import OrderedDict

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
# gpt-5.6-terra: balances intelligence and cost, has vision (needed for figures).
# Alternatives: "gpt-5.6-sol" (flagship, messy scans) or "gpt-5.6-luna" (cheapest).
MODEL = os.environ.get("EXTRACTOR_MODEL", "gpt-5.6-terra")
REASONING_EFFORT = os.environ.get("EXTRACTOR_EFFORT", "medium")  # none/low/medium/high/xhigh/max
MAX_OUTPUT_TOKENS = 16000
PDF_DETAIL = "high"       # page-image detail for figures; "low" is cheaper
PAGES_PER_CHUNK = 40      # keep each call focused; also avoids giant single requests
CHUNK_OVERLAP = 1         # avoid losing a problem split across a boundary; dups removed by id
MAX_INLINE_MB = 50        # OpenAI request limit is 50 MB per file; above this use the Files API
MAX_RETRIES = 3

# Canonical topics — EDIT to match your syllabus. Extraction is biased toward these.
CANONICAL_TOPICS = [
    "Asymptotic Notation & Growth", "Recurrences & the Master Theorem",
    "Running-Time Analysis (worst-case / expected / amortized)",
    "Hashing & Hash Tables", "Binary Heaps & Priority Queues",
    "Binary Search Trees", "Balanced BSTs / AVL Trees", "Tree/BST Augmentation",
    "Data-Structure Design", "Comparison Sorting (merge / heap / quick)",
    "Linear-Time Sorting (counting / radix / bucket)", "Sorting Lower Bounds",
    "Order Statistics & Selection", "Graph Representations",
    "Breadth-First Search (BFS)", "Depth-First Search (DFS)",
    "Topological Sort & DAGs", "Connected Components",
    "Bipartiteness & Graph Coloring", "Shortest Paths (unweighted / BFS)",
    "Shortest Paths (weighted: Dijkstra / Bellman-Ford)", "Minimum Spanning Trees",
    "State-Space / Implicit Graph Search", "Divide & Conquer", "Greedy Algorithms",
    "Dynamic Programming", "Complexity (P, NP, Reductions)", "Randomized Algorithms",
]

SYSTEM_PROMPT = """\
You extract structured practice questions from undergraduate ALGORITHMS course \
documents (quizzes, exams, problem sets, problem collections) for an AI tutor. You \
are given the document as text plus rendered page images, so you can SEE every figure.

Rules:

GRANULARITY: one entry per smallest self-contained question. Split multi-part \
problems into parts (Problem 5-1 -> PS5-1a, PS5-1b; Problem 3 (a)(b)(c) -> Q1-3a, \
Q1-3b, Q1-3c). Skip administrative items with no algorithmic content (e.g. "write \
your name").

IDS: build a stable question_id from the document's own numbering (quizzes/exams \
-> "Q<n>-...", problem sets -> "PS<n>-..."). If none, use "P1","P2",...

QUESTION: faithful, concise restatement of what the student must do, including \
essential given data (arrays, matrices, required time bounds, key constraints). \
Condense long flavor text; keep the actual task.

SOLUTION: faithfully summarize the provided solution — final answer, algorithm/key \
idea, essential reasoning, complexity. Include code if the code IS the answer. If no \
solution is present, set solution to "NOT PROVIDED IN SOURCE". Do not invent one.

IMAGE_DESCRIPTION: if the question or solution has a figure, describe precisely what \
you see, grounded in the text (label vertices/edges, tree parent-child structure, \
board layout, axes). If there is no figure, set it to null. Never fabricate a figure.

COMMON_MISTAKES: set ONLY when the document explicitly gives instructor-written \
common mistakes/pitfalls for THAT question (often labeled "Common Mistakes"); \
summarize them. Otherwise null. NEVER invent. Grading rubrics are NOT common mistakes.

TOPIC: short label; strongly prefer one of these canonical topics when it fits:
{taxonomy}
If nothing fits, write a concise new topic name.

Be accurate over comprehensive."""

# Strict Structured-Outputs schema. Strict mode requires: every property listed in
# "required", additionalProperties:false, and nullable fields typed as ["string","null"].
SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "source_type": {"type": "string",
                        "enum": ["quiz", "exam", "problem_set",
                                 "problem_collection", "worksheet", "other"]},
        "document_title": {"type": "string"},
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "question_id": {"type": "string"},
                    "topic": {"type": "string"},
                    "question": {"type": "string"},
                    "solution": {"type": "string"},
                    "image_description": {"type": ["string", "null"]},
                    "common_mistakes": {"type": ["string", "null"]},
                },
                "required": ["question_id", "topic", "question", "solution",
                             "image_description", "common_mistakes"],
            },
        },
    },
    "required": ["source_type", "document_title", "questions"],
}


# ---------------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------------
def iter_chunks(path):
    """Yield (label, base64_pdf). Small PDFs -> one chunk; large PDFs -> overlapping
    page ranges so no problem is lost at a boundary."""
    from pypdf import PdfReader, PdfWriter

    def encode(b):
        if len(b) > MAX_INLINE_MB * 1024 * 1024:
            raise ValueError(f"chunk is {len(b)/1e6:.1f} MB (> {MAX_INLINE_MB} MB "
                             "limit); upload it via the Files API and pass file_id instead.")
        return base64.standard_b64encode(b).decode("utf-8")

    reader = PdfReader(path)
    n = len(reader.pages)
    if n <= PAGES_PER_CHUNK:
        with open(path, "rb") as f:
            yield ("all", encode(f.read()))
        return

    step = PAGES_PER_CHUNK - CHUNK_OVERLAP
    start = 0
    while start < n:
        end = min(start + PAGES_PER_CHUNK, n)
        writer = PdfWriter()
        for p in range(start, end):
            writer.add_page(reader.pages[p])
        buf = io.BytesIO()
        writer.write(buf)
        yield (f"pages_{start+1}-{end}", encode(buf.getvalue()))
        if end == n:
            break
        start += step


def page_count(path):
    from pypdf import PdfReader
    return len(PdfReader(path).pages)


# ---------------------------------------------------------------------------
# OpenAI call
# ---------------------------------------------------------------------------
def extract_chunk(client, filename, pdf_b64, chunk_note, usage):
    system = SYSTEM_PROMPT.format(taxonomy="\n".join(f"- {t}" for t in CANONICAL_TOPICS))
    user_text = "Extract all questions from this document following the rules."
    if chunk_note:
        user_text += f" {chunk_note}"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.responses.create(
                model=MODEL,
                reasoning={"effort": REASONING_EFFORT},
                max_output_tokens=MAX_OUTPUT_TOKENS,
                input=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": [
                        {"type": "input_file",
                         "filename": filename,
                         "file_data": f"data:application/pdf;base64,{pdf_b64}",
                         "detail": PDF_DETAIL},
                        {"type": "input_text", "text": user_text},
                    ]},
                ],
                text={"format": {
                    "type": "json_schema",
                    "name": "question_extraction",
                    "schema": SCHEMA,
                    "strict": True,
                }},
            )
            if getattr(resp, "usage", None):
                usage["in"] += resp.usage.input_tokens
                usage["out"] += resp.usage.output_tokens

            if resp.status == "incomplete":
                reason = getattr(resp.incomplete_details, "reason", "unknown")
                raise RuntimeError(f"incomplete response ({reason}); raise MAX_OUTPUT_TOKENS "
                                   "or lower PAGES_PER_CHUNK")

            # Strict Structured Outputs guarantees schema-valid JSON in output_text.
            return json.loads(resp.output_text)

        except Exception as e:  # noqa: BLE001
            if attempt == MAX_RETRIES:
                raise
            time.sleep(2.0 * 2 ** (attempt - 1))


def clean(q):
    """Empty/'none' strings in optional fields -> None."""
    for k in ("image_description", "common_mistakes"):
        v = q.get(k)
        if isinstance(v, str) and v.strip().lower() in ("", "none", "n/a", "null"):
            q[k] = None
    return q


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def extract_pdf(client, path, usage):
    fname = os.path.basename(path)
    chunks = list(iter_chunks(path))
    print(f"  {fname}: {page_count(path)} page(s), {len(chunks)} chunk(s)")
    seen, out = set(), []
    for label, b64 in chunks:
        note = "" if label == "all" else (
            f"(Page slice {label}; only emit problems you can read fully.)")
        data = extract_chunk(client, fname, b64, note, usage)
        for q in data.get("questions", []):
            if not q.get("question_id") or q["question_id"] in seen:
                continue
            seen.add(q["question_id"])
            q["source"] = fname
            q["source_type"] = data.get("source_type")
            out.append(clean(q))
    print(f"    -> {len(out)} question(s)")
    return out


def group_by_topic(questions, course):
    topics = OrderedDict()
    for q in questions:
        topics.setdefault(q["topic"], []).append(q)
    return {
        "course": course,
        "topics": [
            {"topic": t, "questions": [
                {"question_id": q["question_id"], "source": q.get("source"),
                 "source_type": q.get("source_type"), "question": q["question"],
                 "solution": q["solution"], "image_description": q.get("image_description"),
                 "common_mistakes": q.get("common_mistakes")}
                for q in qs]}
            for t, qs in topics.items()
        ],
    }


def main():
    ap = argparse.ArgumentParser(description="Extract algorithms questions from PDFs (OpenAI).")
    ap.add_argument("--input", required=True, help="a PDF file or a folder of PDFs")
    ap.add_argument("--output", default="questions.json")
    ap.add_argument("--course", default="Algorithms Course")
    ap.add_argument("--dry-run", action="store_true",
                    help="list files/pages/chunks and exit (no API key needed)")
    args = ap.parse_args()

    paths = (sorted(glob.glob(os.path.join(args.input, "*.pdf")))
             if os.path.isdir(args.input) else [args.input])
    if not paths:
        sys.exit(f"No PDFs found at {args.input}")

    if args.dry_run:
        print(f"Model would be: {MODEL} (effort={REASONING_EFFORT})\nFiles to process:")
        for p in paths:
            chunks = [lab for lab, _ in iter_chunks(p)]
            print(f"  {os.path.basename(p)}: {page_count(p)} pages -> chunks {chunks}")
        print("Dry run OK. Remove --dry-run and set OPENAI_API_KEY to extract.")
        return

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("ERROR: set OPENAI_API_KEY (or use --dry-run).")

    from openai import OpenAI
    client = OpenAI()

    usage = {"in": 0, "out": 0}
    all_q = []
    print(f"Model: {MODEL} (effort={REASONING_EFFORT})\nExtracting {len(paths)} file(s):")
    for p in paths:
        try:
            all_q += extract_pdf(client, p, usage)
        except Exception as e:  # noqa: BLE001
            print(f"  !! {os.path.basename(p)} failed: {e}", file=sys.stderr)

    if not all_q:
        sys.exit("No questions extracted.")

    result = group_by_topic(all_q, args.course)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    n = sum(len(t["questions"]) for t in result["topics"])
    price = {  # USD per million tokens (input, output)
        "gpt-5.6-sol": (5, 30), "gpt-5.6": (5, 30),
        "gpt-5.6-terra": (2.5, 15), "gpt-5.6-luna": (1, 6),
    }
    pin, pout = price.get(MODEL, (0, 0))
    cost = usage["in"]/1e6*pin + usage["out"]/1e6*pout
    print(f"\nDone. {n} questions across {len(result['topics'])} topics -> {args.output}")
    print(f"Tokens: {usage['in']:,} in / {usage['out']:,} out (~${cost:.2f})")


if __name__ == "__main__":
    main()
