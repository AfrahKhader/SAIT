"""Agent components: edges

These decide which node the graph moves to next, based on the current state.
graph.py wires them into `add_conditional_edges`.
"""

from nodes import AgentState


def route_after_guard(state: AgentState) -> str:
    """After the guardrail: clean input -> tutor, everything else -> redirect."""
    # return "socratic" if state["guard_verdict"] == "ON_TOPIC" else "redirect"
    return "assess" if state["guard_verdict"] == "ON_TOPIC" else "redirect"


def should_continue(state: AgentState) -> bool:
    """After the tutor: True if it asked to call a tool, else the turn ends."""
    last = state["messages"][-1]
    return hasattr(last, "tool_calls") and len(last.tool_calls) > 0