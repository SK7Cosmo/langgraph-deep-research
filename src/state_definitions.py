
"""
This module defines the state objects and structured schemas used for
the research agent workflow
"""

# Imports
import operator

from pydantic import BaseModel, Field  # For runtime validation on state and output schemas
from langgraph.graph import MessagesState  # Default Graph State with messages field
from langchain_core.messages import BaseMessage  # Default class to represent a message
from langgraph.graph.message import add_messages  # Chat History Management to append messages
from typing_extensions import Optional, Annotated, Sequence, List, TypedDict # To support older python versions


class BriefingAgentInputState(MessagesState):
	"""Input state with only messages from user input."""
	pass


class BriefingAgentOutputState(MessagesState):
	"""
	Output state with the genenrated research brief
	"""
	# Research brief generated from user conversation history
	research_brief: str


class ResearcherState(TypedDict):
	"""
	State for the research agent containing message history and research metadata.
	"""
	researcher_messages: Annotated[Sequence[BaseMessage], add_messages]

class ResearcherOutputState(TypedDict):
	"""
	Output state for the research agent containing final research results.
	"""
	ai_summary: str
	cleaned_research_findings: str
