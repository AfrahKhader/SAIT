"""Entry point for the Socratic RAG tutor.

Run from the SAIT project root:

    python main.py

This lives at the project root; the importable modules live in src/. The line
below puts src/ on the import path (anchored to this file's location, so it
works regardless of your current working directory).
"""

import sys
from pathlib import Path

# --- make the src/ modules importable ---
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from dotenv import load_dotenv  # noqa: E402  (import after path setup, intentional)
from langchain_core.messages import HumanMessage  # noqa: E402

from config import Config  # noqa: E402
from graph import build_agent  # noqa: E402
from retrieval import KnowledgeBase  # noqa: E402


def main() -> None:
    load_dotenv()  # picks up OPENAI_API_KEY from .env

    config = Config.load()  # defaults to SAIT/config/*.yaml
    knowledge_base = KnowledgeBase(config).build()
    agent = build_agent(config, knowledge_base)
    session = {"configurable": {"thread_id": config.session.thread_id}}

    agent.get_graph().print_ascii()
    print("\n=== SOCRATIC RAG TUTOR ===")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ("exit", "quit"):
            break

        result = agent.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=session,
        )
        print("\n=== TUTOR ===")
        print(result["messages"][-1].content)


if __name__ == "__main__":
    main()