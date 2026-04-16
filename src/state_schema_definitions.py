
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


class AgentInputState(MessagesState):
	"""Input state with only messages from user input."""
	pass


class AgentMasterState(MessagesState):
	"""
	Main state for the full multi-agent research system.
	Extends MessagesState with additional fields for research coordination.
	"""

	# Research brief generated from user conversation history
	research_brief: Optional[str]
	# Messages exchanged with the supervisor agent for coordination
	supervisor_messages: Annotated[Sequence[BaseMessage], add_messages]
	# Raw unprocessed research notes collected during the research phase
	raw_notes: Annotated[list[str], operator.add] = []
	# Processed and structured notes ready for report generation
	notes: Annotated[list[str], operator.add] = []
	# Final formatted research report
	final_report: str


class ResearcherState(TypedDict):
	"""
	State for the research agent containing message history and research metadata.
	"""
	researcher_messages: Annotated[Sequence[BaseMessage], add_messages]
	tool_call_iterations: int
	research_topic: str
	compressed_research: str
	raw_notes: Annotated[List[str], operator.add]


class ResearcherOutputState(TypedDict):
	"""
	Output state for the research agent containing final research results.
	"""
	compressed_research: str
	raw_notes: Annotated[List[str], operator.add]
	researcher_messages: Annotated[Sequence[BaseMessage], add_messages]


class ClarifyWithUserSchema(BaseModel):
	"""Schema for user clarification decision and questions."""

	need_clarification: bool = Field(
		description="Whether the user needs to be asked a clarifying question.",
	)
	question: str = Field(
		description="A question to ask the user to clarify the report scope",
	)
	verification: str = Field(
		description="Verify message that we will start research after the user has provided the necessary information.",
	)


class ResearchQuestionSchema(BaseModel):
	"""Schema for structured research brief generation."""

	research_brief: str = Field(
		description="A research question that will be used to guide the research.",
	)


class SummarySchema(BaseModel):
	"""Schema for webpage content summarization."""
	summary: str = Field(description="Concise summary of the webpage content")