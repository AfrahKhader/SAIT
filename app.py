"""Streamlit front end for the Socratic RAG tutor — persistent side whiteboard.

Run from the SAIT project root:

    streamlit run app.py

Layout: chat on the left, a whiteboard fixed on the right that PERSISTS across
turns, so the student can iteratively add to / fix their solution. Tools: draw,
white-pen erase, and select-then-delete for removing individual strokes without
wiping the board.

Visual system (see THEME_CSS below):
  - Newsreader (serif) is used ONLY for the tutor's replies — its words are the
    lecture-note content, everything else is app chrome.
  - Inter for chrome, JetBrains Mono for question ids / model names.
  - Marker blue #1E6BB8 as the single accent; the board is the one raised
    surface on a cool off-white page.
"""

import base64
import io
import os
import random
import sys
import uuid
from pathlib import Path

# --- make the src/ modules importable (same bootstrap as main.py) ---
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402
from PIL import Image  # noqa: E402
from streamlit_drawable_canvas import st_canvas  # noqa: E402

from config import Config  # noqa: E402
from graph import build_agent  # noqa: E402
from retrieval import KnowledgeBase  # noqa: E402

try:  # optional — the app runs fine without the misconception catalog
    from misconceptions import MisconceptionCatalog  # noqa: E402
except Exception:
    MisconceptionCatalog = None

load_dotenv()

st.set_page_config(
    page_title="Socratic Algorithms Tutor",
    page_icon="✎",
    layout="wide",
    initial_sidebar_state="expanded",
)


TOPIC_ALLOWLIST = [
    ("Binary Search Trees",       ("binary search tree",)),
    ("Graph Representations",     ("graph representation",)),
    ("Breadth-First Search",      ("breadth-first", "breadth first")),
    ("Depth-First Search",        ("depth-first", "depth first")),
    ("Dijkstra's Algorithm",      ("dijkstra",)),
]

CHAT_H = 640 
BOARD_H = 600 

BOARD_W = 560


THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Newsreader:opsz,wght@6..72,400;6..72,500&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --ink:        #14181F;
  --ink-soft:   #5B6472;
  --paper:      #F6F6F3;
  --surface:    #FFFFFF;
  --marker:     #1E6BB8;
  --wash:       #EAF2FA;
  --erase:      #B23A2F;
  --rule:       #E3E3DD;
  --radius:     12px;
}

/* ---------- base ---------- */
.stApp { background: var(--paper); color: var(--ink); }
.stApp, .stApp button, .stApp input, .stApp textarea, .stApp select, .stApp label {
  font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
}
.block-container { padding-top: 2rem; padding-bottom: 2.5rem; max-width: 1680px; }
code, kbd, pre { font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, monospace !important; }

/* ---------- masthead ---------- */
.masthead { border-bottom: 1px solid var(--rule); padding-bottom: .85rem; margin-bottom: 1.35rem; }
.masthead .eyebrow {
  font-family: 'JetBrains Mono', monospace; font-size: .66rem; letter-spacing: .16em;
  text-transform: uppercase; color: var(--marker); font-weight: 500;
}
.masthead h1 {
  font-family: 'Newsreader', Georgia, serif; font-weight: 500; font-size: 1.9rem;
  letter-spacing: -.012em; margin: .15rem 0 0; color: var(--ink); line-height: 1.15;
}
.masthead p { margin: .3rem 0 0; color: var(--ink-soft); font-size: .88rem; }

/* ---------- panels (scoped by invisible anchor spans) ---------- */
.panel-anchor { display: none; }
div[data-testid="stElementContainer"]:has(> div > .panel-anchor) { display: none; height: 0; }

[data-testid="stVerticalBlockBorderWrapper"]:has(.panel-chat),
[data-testid="stVerticalBlockBorderWrapper"]:has(.panel-composer),
[data-testid="stVerticalBlockBorderWrapper"]:has(.panel-board) {
  background: var(--surface);
  border: 1px solid var(--rule);
  border-radius: var(--radius);
  box-shadow: 0 1px 2px rgba(20,24,31,.04);
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.panel-board) {
  box-shadow: 0 2px 10px rgba(20,24,31,.07);   /* the board sits slightly proud */
  overflow-x: auto;   /* if BOARD_W outgrows the column, scroll instead of clip */
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.panel-composer) { margin-top: .9rem; }

/* ---------- chat: type does the role-signalling, not avatars ---------- */
[data-testid="stChatMessage"] { background: transparent; padding: .3rem 0 .95rem; }
[data-testid="stChatMessage"] [data-testid^="stChatMessageAvatar"] { display: none; }

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
  border-left: 2px solid var(--marker); padding-left: 1rem; margin: .2rem 0 1.1rem;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) p,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) li {
  font-family: 'Newsreader', Georgia, serif !important;
  font-size: 1.045rem; line-height: 1.62; color: var(--ink);
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  background: var(--wash); border-radius: 10px; padding: .7rem .95rem;
  margin: .2rem 0 1.1rem 14%;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) p {
  font-size: .93rem; line-height: 1.5;
}

/* ---------- empty state ---------- */
.welcome h2 {
  font-family: 'Newsreader', Georgia, serif; font-weight: 500; font-size: 1.45rem;
  margin: .2rem 0 .5rem; letter-spacing: -.01em;
}
.welcome p { color: var(--ink-soft); font-size: .95rem; line-height: 1.6; margin: 0 0 .3rem; }
.section-label {
  font-family: 'JetBrains Mono', monospace; font-size: .65rem; letter-spacing: .14em;
  text-transform: uppercase; color: var(--ink-soft); margin: 1.3rem 0 .55rem;
}

/* ---------- board tray ---------- */
.tray-label {
  font-family: 'JetBrains Mono', monospace; font-size: .65rem; letter-spacing: .14em;
  text-transform: uppercase; color: var(--ink-soft); margin-bottom: .45rem;
}
[data-testid="stCustomComponentV1"] { border-radius: 8px; overflow: hidden; }
.st-key-clear_board button:hover { color: var(--erase); border-color: var(--erase); }

/* ---------- attachment chip ---------- */
.chip {
  display: inline-flex; align-items: center; gap: .4rem;
  background: var(--wash); color: var(--marker);
  border-radius: 999px; padding: .25rem .7rem;
  font-size: .78rem; font-weight: 500;
}

/* ---------- controls ---------- */
.stButton button, .stFormSubmitButton button { border-radius: 8px; font-weight: 500; }
.stTextArea textarea { border-radius: 8px; font-size: .93rem; }
[data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--rule); }

/* ---------- layout ---------- */
div[data-testid="stHorizontalBlock"] { align-items: flex-start; }

/* ---------- quality floor ---------- */
:focus-visible { outline: 2px solid var(--marker); outline-offset: 2px; }
@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }

</style>

"""

st.markdown(THEME_CSS, unsafe_allow_html=True)

def anchor(name: str) -> None:
    """Drop an invisible marker so CSS can scope styles to a specific panel."""
    st.markdown(f'<span class="panel-anchor {name}"></span>', unsafe_allow_html=True)


def choice_control(label, options, key):
    """Segmented control where available (Streamlit >= 1.40), radio otherwise."""
    if hasattr(st, "segmented_control"):
        picked = st.segmented_control(
            label, options, default=options[0], key=key, label_visibility="collapsed"
        )
        return picked or options[0]
    return st.radio(label, options, horizontal=True, label_visibility="collapsed", key=key)


@st.cache_resource(show_spinner="Loading lecture notes and preparing the tutor…")
def load_agent():
    config = Config.load()
    knowledge_base = KnowledgeBase(config).build()
    agent = build_agent(config, knowledge_base)

    # A UI-side catalog for the opening topic buttons. Separate from the one the
    # graph builds internally — this one just needs the topic/question lookups.
    catalog = None
    misc_cfg = getattr(config, "misconceptions", None)
    if MisconceptionCatalog is not None and misc_cfg is not None and misc_cfg.enabled:
        try:
            catalog = MisconceptionCatalog(config).build()
        except Exception as e:  # never let a catalog problem block the tutor
            print(f"[app] misconception catalog unavailable: {e}")

    return config, agent, catalog


if not os.getenv("OPENAI_API_KEY"):
    st.error(
        "No `OPENAI_API_KEY` found. Add it to a `.env` file in the project root "
        "(see `.env.example`), then reload."
    )
    st.stop()

config, agent, catalog = load_agent()


st.session_state.setdefault("thread_id", str(uuid.uuid4()))
st.session_state.setdefault("messages", [])       # [{role, text, image?}]
st.session_state.setdefault("board_version", 0)   # bump ONLY on explicit "Clear board"


def canvas_to_png(canvas_result):
    """Flatten the canvas onto white; return PNG bytes, or None if blank."""
    if canvas_result is None or canvas_result.image_data is None:
        return None
    rgba = Image.fromarray(canvas_result.image_data.astype("uint8"), "RGBA")
    white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    img = Image.alpha_composite(white, rgba).convert("RGB")
    if img.convert("L").getextrema() == (255, 255):   # entirely white => nothing drawn
        return None
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def png_to_data_url(png_bytes: bytes) -> str:
    return f"data:image/png;base64,{base64.b64encode(png_bytes).decode()}"


def content_to_text(content) -> str:
    """Coerce an LLM message's content to a string.

    gpt-4o sometimes returns content as a list of blocks (esp. after image
    input); st.markdown() needs a plain string, so flatten it here.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        ).strip()
    return str(content)



with st.sidebar:
    st.markdown(
        '<div class="masthead" style="border:none;padding:0;margin-bottom:1rem">'
        # '<div class="eyebrow">Socratic method</div>'
        '<h1 style="font-size:1.3rem">SAIT Tutor</h1>'
        "</div>",
        unsafe_allow_html=True,
    )
    if st.button("Start a new conversation", use_container_width=True, type="primary"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.board_version += 1
        st.rerun()

    with st.expander("How this works"):
        st.markdown(
            "- The tutor asks questions and gives hints. It won't hand over the answer.\n"
            "- Anything on the whiteboard is sent with your next message.\n"
            "- The board keeps its strokes between turns, so you can keep refining."
        )

    st.divider()
    st.caption(f"Tutor model &nbsp;`{config.llm.tutor_model}`", unsafe_allow_html=True)



st.markdown(
    '<div class="masthead">'
    '<div class="eyebrow">Guided practice · your lecture notes</div>'
    "<h1>Socratic Algorithms Tutor</h1>"
    "<p>Start by choosing a topic from the following options: </p>"
    "</div>",
    unsafe_allow_html=True,
)

chat_col, board_col = st.columns([5, 4], gap="large")


with board_col:
    with st.container(border=True):
        anchor("panel-board")
        st.markdown('<div class="tray-label">Whiteboard</div>',
                    unsafe_allow_html=True)

        tool_col, pen_col, clear_col = st.columns([5, 2, 2], vertical_alignment="center")
        with tool_col:
            tool = choice_control("Tool", ["Draw", "Erase", "Select"], key="board_tool")
        with pen_col:
            if hasattr(st, "popover"):
                with st.popover("Pen", use_container_width=True):
                    st.slider("Pen size", 1, 30, 3, key="pen_size")
                pen = st.session_state.get("pen_size", 3)
            else:
                pen = st.slider("Pen size", 1, 30, 3, key="pen_size")
        with clear_col:
            clear = st.button(
                "Clear", key="clear_board", use_container_width=True,
                help="Erase everything on the board",
            )

        # Map the chosen tool to canvas parameters.
        tool_map = {
            "Draw": ("freedraw", "#14181F", pen),
            "Erase": ("freedraw", "#FFFFFF", max(pen, 15)),   # white pen paints over
            "Select": ("transform", "#14181F", pen),          # click stroke, press Delete
        }
        drawing_mode, stroke_color, stroke_w = tool_map[tool]

        if clear:
            st.session_state.board_version += 1
            st.rerun()

        canvas_result = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=stroke_w,
            stroke_color=stroke_color,
            background_color="#FFFFFF",
            height=BOARD_H,
            width=BOARD_W,
            drawing_mode=drawing_mode,
            display_toolbar=True,          # undo / redo / delete buttons under the canvas
            key=f"board_{st.session_state.board_version}",
        )
        # No `initial_drawing` re-injection: feeding the saved drawing back on every
        # rerun made the newest stroke blink. With a stable key the canvas keeps its
        # own strokes, and we avoid forcing a rerun on send (see the handler below).

        # Computed once here so the composer can tell the student what's about to
        # be attached; reused verbatim by the submit handler further down.
        board_png = canvas_to_png(canvas_result)

        with st.expander("Tool guide"):
            st.markdown(
                "**Draw** — sketch freehand.\n\n"
                "**Erase** — a white pen that paints over your strokes.\n\n"
                "**Select** — click a single stroke, then press *Delete* to remove "
                "just that one.\n\n"
                "Undo and redo live in the small toolbar under the board."
            )

# ---- Chat ----
with chat_col:
    # Fixed-height scrolling transcript: the conversation scrolls INSIDE this box
    # rather than growing the page.
    history = st.container(height=CHAT_H, border=True)
    with history:
        anchor("panel-chat")

        student_has_spoken = any(m["role"] == "user" for m in st.session_state.messages)

        # if not st.session_state.messages:
        #     st.markdown(
        #         '<div class="welcome">'
        #         "<h2>Start with a topic</h2>"
        #     #     "<p>I teach by asking questions — I won't hand over the answer. "
        #     #     "Describe a problem in your own words, tell me where your reasoning "
        #     #     "broke down, or sketch your working on the board and send it for "
        #     #     "feedback. The board keeps its strokes between turns, so you can "
        #     #     "keep refining.</p>"
        #         "</div>",
        #         unsafe_allow_html=True,
        #     )

        # --- Opening topic buttons ---------------------------------------
        # Shown only before the student's first turn. A pick opens a catalogued
        # problem AND pins its id (handled below), so the detector fires from the
        # first attempt and skips its identification call. The grid is built from
        # the catalog, so it widens as you fill in the remaining questions.
        if catalog is not None and not student_has_spoken:
            topics = catalog.topics_all()
            names = list(topics)

            # Keep only the allowlisted topics, in the order listed above.
            offered, claimed, missing = [], set(), []
            for label, keywords in TOPIC_ALLOWLIST:
                members = [
                    n for n in names
                    if n not in claimed and any(k in n.lower() for k in keywords)
                ]
                if members:
                    claimed.update(members)
                    offered.append((label, members))
                else:
                    missing.append(label)
            if missing:
                # Terminal only — the student shouldn't see the app's plumbing.
                print(f"[app] not in the misconception catalog, so not offered: "
                      f"{', '.join(missing)}")

            usable = set(catalog.usable_question_ids())

            st.markdown('<div class="section-label"></div>',
                        unsafe_allow_html=True)
            cols = st.columns(2)
            for i, (label, members) in enumerate(offered):
                # Open any one of the topic's problems, so clicking the same
                # button twice doesn't always serve the same question.
                pool = [q for t in members for q in topics[t] if q in usable]
                if not pool:
                    pool = [topics[members[0]][0]]
                if cols[i % 2].button(
                    label,
                    key=f"group-{label}",
                    use_container_width=True,
                    help="Covers: " + ", ".join(members),
                ):
                    st.session_state["_pending_qid"] = random.choice(pool)

            # Finding #8 — offer, don't mandate: one tap lets the tutor choose.
            if offered and st.button("Pick one for me", use_container_width=True):
                # Drawn from the offered topics only — otherwise this quietly
                # hands back the whole catalog the allowlist just filtered out.
                anywhere = [q for _, members in offered for t in members
                            for q in topics[t] if q in usable]
                st.session_state["_pending_qid"] = random.choice(
                    anywhere or list(usable)
                )

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if msg.get("text"):
                    st.markdown(msg["text"])
                if msg.get("image"):
                    st.image(msg["image"], caption="your sketch", width=280)

    # Composer stays below the scrolling box, always visible.
    with st.container(border=True):
        anchor("panel-composer")
        with st.form("chat_form", clear_on_submit=True, border=False):
            text = st.text_area(
                "Your message",
                height=90,
                placeholder="Ask a question or explain your thinking…",
                label_visibility="collapsed",
            )
            chip_col, send_col = st.columns([3, 1], vertical_alignment="center")
            with chip_col:
                if board_png is not None:
                    st.markdown(
                        '<span class="chip">✎ Your sketch goes with this message</span>',
                        unsafe_allow_html=True,
                    )
            with send_col:
                submitted = st.form_submit_button(
                    "Send", type="primary", use_container_width=True
                )

        # The handlers below run after both columns, so their spinners and
        # warnings would otherwise appear at the very bottom of the page, far
        # from the Send button. Reserve a slot for them here instead.
        status_slot = st.empty()



pending_qid = st.session_state.pop("_pending_qid", None)
if pending_qid and catalog is not None:
    item = catalog.question(pending_qid)
    if item:
        topic = item.get("topic", "this problem")
        display = f"Let's work on **{topic}** (`{pending_qid}`)."
        # The bubble stays short; the agent gets the full problem to probe on.
        graph_content = (
            f"I'd like to work on this problem:\n\n{item['question']}\n\n"
            "Please start by helping me think about how to approach it."
        )
        st.session_state.messages.append(
            {"role": "user", "text": display, "image": None}
        )
        with status_slot, st.spinner("Thinking…"):
            session = {"configurable": {"thread_id": st.session_state.thread_id}}
            try:
                # Passing `current_question_id` lets the detector skip its own
                # identification LLM call — the pick already tells us the problem.
                result = agent.invoke(
                    {
                        "messages": [HumanMessage(content=graph_content)],
                        "current_question_id": pending_qid,
                    },
                    config=session,
                )
                reply = content_to_text(result["messages"][-1].content)
                if not reply:
                    reply = "*(The tutor returned an empty response.)*"
            except Exception as e:
                import traceback
                traceback.print_exc()
                reply = f"Something went wrong: `{e}`"
        st.session_state.messages.append({"role": "assistant", "text": reply})
        st.rerun()   # safe: board empty on the opening move; hides the buttons


if submitted:
    png = board_png

    if not text.strip() and png is None:
        status_slot.warning("Type a message or draw something first.")
    else:
        if png is not None:
            parts = []
            if text.strip():
                parts.append({"type": "text", "text": text})
            parts.append({"type": "image_url", "image_url": {"url": png_to_data_url(png)}})
            message_content = parts
        else:
            message_content = text

        st.session_state.messages.append({"role": "user", "text": text, "image": png})
        # Render this turn INLINE into the scrolling history container. We avoid
        # st.rerun() on purpose: a rerun would re-render the canvas and wipe the
        # student's drawing. The turn is in session_state, so on the next natural
        # rerun the history loop above will render it normally (no duplication).
        with history:
            with st.chat_message("user"):
                if text.strip():
                    st.markdown(text)
                if png is not None:
                    st.image(png, caption="your sketch", width=280)

        with status_slot, st.spinner("Thinking…"):
            session = {"configurable": {"thread_id": st.session_state.thread_id}}
            try:
                result = agent.invoke(
                    {"messages": [HumanMessage(content=message_content)]},
                    config=session,
                )
                reply = content_to_text(result["messages"][-1].content)
                if not reply:
                    reply = "*(The tutor returned an empty response.)*"
            except Exception as e:
                import traceback
                traceback.print_exc()   # full traceback to the terminal
                reply = f"Something went wrong: `{e}`"

        st.session_state.messages.append({"role": "assistant", "text": reply})
        with history:
            with st.chat_message("assistant"):
                st.markdown(reply)