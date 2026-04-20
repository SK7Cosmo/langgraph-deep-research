"""
Core Research Workflow

The goal of this module is to gather the context requested by the research brief.

This module implements the core research phase of the research workflow, where we:
1. Conduct web searches using tavily tool
2. Analyze results after every search to check for comprehensiveness
3. Format search results into a well structured summarized output
"""

# Imports
from langchain_groq import ChatGroq # Langchain LLM model integration
from typing_extensions import Literal # To define constants
from langgraph.graph import StateGraph, START, END # Langgraph Entities to build and design control flow
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage # Different Message types

# Custom Imports
from src.utils.util_functions import summarization_llm
from src.utils.util_functions import extract_tool_content
from src.utils.util_tools import tavily_search_tool, think_tool
from src.state_definitions import ResearcherState, ResearcherOutputState
from src.prompts import research_agent_prompt, clean_research_findings_prompt

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configuration
# LLM to handle the core research
research_agent_llm = ChatGroq(
    model="openai/gpt-oss-120b", # Better for agents
    temperature=0, # Factual response
    max_tokens= 3000)

# Set up tools and model binding
tools = [tavily_search_tool, think_tool]
tools_by_name = {tool.name: tool for tool in tools}

research_agent_llm_with_tools = research_agent_llm.bind_tools(tools)


# Defining Nodes
def call_research_agent_llm(state: ResearcherState):
    """
    Analyze current state and decide on next actions.

    The model analyzes the current conversation state and decides whether to:
    1. Call search tools to gather more information
    2. Provide a final answer based on gathered information

    Returns updated state with the model's response.
    """
    return {
        "researcher_messages": [
            research_agent_llm_with_tools.invoke(
                [SystemMessage(content=research_agent_prompt)] + state["researcher_messages"]
            )
        ]
    }


def execute_tools(state: ResearcherState):
    """
    Executes all tool calls from the latest LLM responses.
    Returns updated state with tool execution results.
    """
    tool_calls = state["researcher_messages"][-1].tool_calls

    # Execute all tool calls and append the tool results
    observations = []
    for tool_call in tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observations.append(tool.invoke(tool_call["args"]))

    # Format the tool message outputs
    tool_outputs = [
        ToolMessage(
            content=observation,
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        ) for observation, tool_call in zip(observations, tool_calls)
    ]

    return {"researcher_messages": tool_outputs}


def clean_research_findings(state: ResearcherState) -> dict:
    """
    Extracts tool messages from the researcher messages and prepare
    """
    # Final summary - AI version
    ai_summary = ''
    for msg in reversed(state.get("researcher_messages", [])):
        if isinstance(msg, AIMessage):
            ai_summary = msg
            break

    # Extracted search tool results
    raw_data = extract_tool_content(state.get("researcher_messages", []))

    if not raw_data.strip():
        return {"cleaned_research_findings": "No research data available."}

    # Invoke the summarization_llm with research cleaning instructions and search tool results
    messages = [
        SystemMessage(content=clean_research_findings_prompt),
        HumanMessage(content=f"Research data:\n{raw_data}")]
    response = summarization_llm.invoke(messages)

    content = (response.content or "").strip()

    if not content:
        content = "No meaningful findings extracted."

    return {"cleaned_research_findings": content,
            "ai_summary": ai_summary}


def continue_or_terminate(state: ResearcherState) -> Literal["execute_tools", "clean_research_findings"]:
    """
    This is a routing function
    Determines whether the agent should continue the research loop or provide
    a final answer based on the LLM made tool calls.

    Returns either of these:
        "execute_tools": Continue to tool execution
        "clean_research_findings": Stop and clean the research findings
    """
    messages = state["researcher_messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, continue to tool execution. Otherwise, we have a final answer
    if last_message.tool_calls:
        return "execute_tools"
    return "clean_research_findings"


# Build the research workflow
research_agent_builder = StateGraph(ResearcherState, output_schema=ResearcherOutputState)

# Add nodes to the graph
research_agent_builder.add_node("call_research_agent_llm", call_research_agent_llm)
research_agent_builder.add_node("execute_tools", execute_tools)
research_agent_builder.add_node("clean_research_findings", clean_research_findings)

# Add edges to connect nodes
research_agent_builder.add_edge(START, "call_research_agent_llm")
research_agent_builder.add_conditional_edges(
    "call_research_agent_llm",
    continue_or_terminate, # routing function
    {
        "execute_tools": "execute_tools", # Continue research loop
        "clean_research_findings": "clean_research_findings", # Provide final answer
    },
)
research_agent_builder.add_edge("execute_tools", "call_research_agent_llm") # Loop back for more research
research_agent_builder.add_edge("clean_research_findings", END)

# Compile the agent
research_agent = research_agent_builder.compile()