"""Agent components: the graph — builds nodes, wires edges, compiles the agent.
    guardrail --ON_TOPIC--> assess --> misconception --> socratic <--> retriever_agent --> END
    --blocked---> redirect --> END
"""

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from config import Config
from edges import route_after_guard, should_continue
from misconceptions import MisconceptionCatalog
from nodes import (
    AgentState,
    AssessNode,
    GuardrailNode,
    MisconceptionNode,
    RedirectNode,
    RetrieverNode,
    SocraticNode,
)
from retrieval import make_retriever_tool


def build_agent(config: Config, knowledge_base):
    """Construct every component from config and return the compiled graph."""

    # --- tools ---
    tools = [make_retriever_tool(knowledge_base.as_retriever())]

    # --- LLMs (from config) ---
    tutor_llm = ChatOpenAI(
        model=config.llm.tutor_model,
        temperature=config.llm.tutor_temperature,
    ).bind_tools(tools)

    guard_llm = ChatOpenAI(
        model=config.llm.guard_model,
        temperature=config.llm.guard_temperature,
    )

    # --- misconception catalog (None = detection disabled, node self-skips) ---
    catalog = (
        MisconceptionCatalog(config).build()
        if config.misconceptions.enabled
        else None
    )

    # --- nodes ---
    guardrail = GuardrailNode(
        guard_llm, config.prompts.guard_system_prompt, config.guardrail.min_length
    )
    socratic = SocraticNode(tutor_llm, config.prompts.socratic_system_prompt, catalog)
    retriever_agent = RetrieverNode(tools)
    redirect = RedirectNode(config.prompts.redirects)
    assess = AssessNode(guard_llm)
    misconception = MisconceptionNode(guard_llm, catalog)

    # --- graph (nodes + edges) ---
    graph = StateGraph(AgentState)
    graph.add_node("assess", assess)
    graph.add_node("misconception", misconception)
    graph.add_node("guardrail", guardrail)
    graph.add_node("redirect", redirect)
    graph.add_node("socratic", socratic)
    graph.add_node("retriever_agent", retriever_agent)

    graph.set_entry_point("guardrail")
    graph.add_conditional_edges(
        "guardrail",
        route_after_guard,
        {"assess": "assess", "redirect": "redirect"},
    )
    graph.add_edge("redirect", END)
    # Detection sits on the student-turn path only. The socratic <-> retriever
    # loop returns straight to socratic, so a tool call does not re-run the
    # detector on the same reply.
    graph.add_edge("assess", "misconception")
    graph.add_edge("misconception", "socratic")
    graph.add_conditional_edges(
        "socratic",
        should_continue,
        {True: "retriever_agent", False: END},
    )
    graph.add_edge("retriever_agent", "socratic")

    return graph.compile(checkpointer=MemorySaver())