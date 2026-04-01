from IPython.display import Markdown, display
from langchain_core.messages import HumanMessage, AIMessage


def format_markdown_messages(messages):
	"""
	Function to format the conversations suitable for Jupyter Notebook,
	"""

	md = ''
	for msg in messages:
		if isinstance(msg, HumanMessage):
			md += f"### 🧑 Human\n{msg.content}\n\n"
		elif isinstance(msg, AIMessage):
			md += f"### 🤖 AI\n{msg.content}\n\n"

	display(Markdown(md))