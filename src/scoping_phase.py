"""
User Clarification and Research Brief Generation

This module implements the scoping phase of the research workflow, where we:
1. Assess if the user's request needs clarification
2. Generate a detailed research brief from the conversation
"""

# Imports
from typing_extensions import Literal 	# To define constants
from langchain_groq import ChatGroq		# Langchain LLM model integration

# Langgraph Entities to build and design control flow
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langchain_core.messages import HumanMessage, AIMessage, get_buffer_string # Different Message types

# Custom Imports
from src.state_definitions import BriefingAgentInputState, BriefingAgentOutputState
from src.prompts import clarify_with_user_instructions, transform_messages_into_research_topic_prompt
from src.utils.util_functions import parse_clarification_response

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Setting up LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile", # better reasoning model
    temperature=0) # Factual response

# Defining Nodes
def clarify_with_user(state: BriefingAgentInputState) -> Command[Literal["write_research_brief", "__end__"]]:
	"""
    Determine if the user's request contains sufficient information to proceed with research.
    """

	# Invoke the model with clarification instructions and user conversations
	# get_buffer_string => list of messages to a string
	response = llm.invoke([HumanMessage(
		content=clarify_with_user_instructions.format(messages=get_buffer_string(messages=state.get("messages", []))))])

	# To match expected structured output
	parsed_response = parse_clarification_response(response.content)

	# Route based on clarification need
	# If clarification is needed, ask ; else generate research brief
	if parsed_response['need_clarification']:
		return Command(
			goto=END,
			update={"messages": [AIMessage(content=parsed_response['question'])]}
			# Follow up question is added to AI Messge history
		)
	else:
		return Command(
			goto="write_research_brief",
			update={"messages": [AIMessage(content=parsed_response['verification'])]}
			# Verification acknowledgement is added to AI Message history
		)


def write_research_brief(state: BriefingAgentInputState):
	"""
    Transform the conversation history into a comprehensive research brief.
    """

	# Generate research brief from conversation history)
	research_brief = llm.invoke([HumanMessage(content=transform_messages_into_research_topic_prompt.format(
		messages=get_buffer_string(messages=state.get("messages", []))))])

	# Update state with generated research brief and pass it to the supervisor
	return {
		"research_brief": research_brief.content,
		"supervisor_messages": [HumanMessage(content=f"{research_brief.content}.")]
	}

# Graph Construction
scope_agent_builder = StateGraph(BriefingAgentOutputState, input_schema=BriefingAgentInputState)

# Add workflow nodes
scope_agent_builder.add_node("clarify_with_user", clarify_with_user)
scope_agent_builder.add_node("write_research_brief", write_research_brief)

# Add workflow edges
scope_agent_builder.add_edge(START, "clarify_with_user") # END if further clarification needed, else goto write_research_brief
scope_agent_builder.add_edge("write_research_brief", END)

# Compile the workflow
scope_agent = scope_agent_builder.compile() 	# No need for checkpointer while running through 'langgraph dev'
