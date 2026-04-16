"""
User Clarification and Research Brief Generation

This module implements the scoping phase of the research workflow, where we:
1. Assess if the user's request needs clarification
2. Generate a detailed research brief from the conversation
"""

# Imports
import os

from typing_extensions import Literal 	# To define constants
from langchain.chat_models import init_chat_model 	# Langchain LLM model integration

# Langgraph Entities to build and design control flow
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langchain_core.messages import HumanMessage, AIMessage, get_buffer_string # Different Message types

# Custom Imports
from src.state_schema_definitions import AgentInputState, AgentMasterState
from src.state_schema_definitions import ClarifyWithUserSchema, ResearchQuestionSchema
from src.prompts import clarify_with_user_instructions, transform_messages_into_research_topic_prompt


# Setting up LLM
llm = init_chat_model(
	"openai:gpt-4.1",
	temperature=0,  # Factual response
	max_tokens=1000,  # OpenRouter free token constraints
	openai_api_key=os.getenv("OPENROUTER_API_KEY"),
	openai_api_base="https://openrouter.ai/api/v1"
)


# Defining Nodes
def clarify_with_user(state: AgentMasterState) -> Command[Literal["write_research_brief", "__end__"]]:
	"""
	Determine if the user's request contains sufficient information to proceed with research.
	"""
	# Set up structured output from the LLM
	structured_output_model = llm.with_structured_output(ClarifyWithUserSchema)

	# Invoke the model with clarification instructions and user conversations
	# get_buffer_string => list of messages to a string
	response = structured_output_model.invoke([HumanMessage(content=clarify_with_user_instructions.format(messages=get_buffer_string(
		messages=state.get("messages", []))))])

	# Route based on clarification need
	# If clarification is needed, ask ; else generate research brief
	if response.need_clarification:
		return Command(
			goto=END,
			update={"messages": [AIMessage(content=response.question)]}  # Follow up question is added to AI Messge history
		)
	else:
		return Command(
			goto="write_research_brief",
			update={"messages": [AIMessage(content=response.verification)]}  # Verification acknowledgement is added to AI Message history
		)


# Defining Nodes
def write_research_brief(state: AgentMasterState):
	"""
	Transform the conversation history into a comprehensive research brief.
	"""
	# Set up structured output from the LLM
	structured_output_model = llm.with_structured_output(ResearchQuestionSchema)

	# Generate research brief from conversation history)

	response = structured_output_model.invoke([HumanMessage(content=transform_messages_into_research_topic_prompt.format(
		messages=get_buffer_string(messages=state.get("messages", []))))])

	# Update state with generated research brief and pass it to the supervisor
	return {
		"research_brief": response.research_brief,
		"supervisor_messages": [HumanMessage(content=f"{response.research_brief}.")]
	}


# Defining Workflow
# Build the scoping workflow
deep_researcher_builder = StateGraph(AgentMasterState, input_schema=AgentInputState)

# Add workflow nodes
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)
deep_researcher_builder.add_node("write_research_brief", write_research_brief)

# Add workflow edges
deep_researcher_builder.add_edge(START, "clarify_with_user") # END if further clarification needed, else goto write_research_brief
deep_researcher_builder.add_edge("write_research_brief", END)

# Compile the workflow
app = deep_researcher_builder.compile() 	# No need for checkpointer while running through 'langgraph dev'
