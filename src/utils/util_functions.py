# Imports
import time
import tiktoken # Intel chunking

from typing import Any
from tavily import TavilyClient # Web search engine
from langchain_groq import ChatGroq	# Langchain LLM model integration
from typing_extensions import List # To support older python versions
from typing_extensions import Literal 	# To define constants
from IPython.display import Markdown, display
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Custom Imports
from src.prompts import summarize_webpage_prompt, combine_summaries_prompt

# Setting up the tavily client to conduct web search
tavily_client = TavilyClient()

# LLM to handle summarization
summarization_llm = ChatGroq(
    model="llama-3.1-8b-instant", # Better for summarization [Fast and Cheap]
    temperature=0, # Factual response
    max_tokens=2000)

def format_markdown_messages(messages):
	"""
	Function to format the conversations suitable for Jupyter Notebook,
	"""

	md = ''
	for msg in messages:
		if not msg.content:
			continue
		if isinstance(msg, HumanMessage):
			md += f"### 🧑 Human\n{msg.content}\n\n"
		elif isinstance(msg, AIMessage):
			md += f"### 🤖 AI\n{msg.content}\n\n"
		elif isinstance(msg, ToolMessage):
			md += f"### ⛏️ Tool\n{msg.content}\n\n"
		else:
			md += "Invalid"

	display(Markdown(md))


def parse_clarification_response(text: str):
	"""
	Parse response from llm model to map with expected structured output
	"""
	lines = text.lower().split("\n")

	result = {
		"need_clarification": False,
		"question": "",
		"verification": ""
	}

	for line in lines:
		if "need_clarification" in line:
			result["need_clarification"] = "true" in line

		elif line.startswith("question"):
			result["question"] = line.split(":", 1)[-1].strip()

		elif line.startswith("verification"):
			result["verification"] = line.split(":", 1)[-1].strip()

	return result


def chunk_text_by_tokens(
    text: str,
    chunk_token_limit: int,
    overlap_tokens: int,
    model: str = "llama-3.1-8b-instant", # Cheap and faster for chunking
):
	"""
	Splits text into token-aware chunks.
	Doing this to deal with massive input [scraped web content]

	Args:
		text: input text [webpage content]
		model: model name (for correct tokenizer)
		chunk_token_limit: max tokens per chunk
		overlap_tokens: overlap between chunks

	Returns:
		List of text chunks
	"""

	enc = tiktoken.get_encoding("cl100k_base")

	tokens = enc.encode(text)
	chunks = []

	start = 0
	total_tokens = len(tokens)

	while start < total_tokens:
		end = start + chunk_token_limit
		chunk_tokens = tokens[start:end]

		chunk_text = enc.decode(chunk_tokens)
		chunks.append(chunk_text)

		start = end - overlap_tokens # To ensure context continuity across chunks

	return chunks


def select_chunks(chunks):
	"""
    Filter chunks by position
    """
	if len(chunks) <= 3:
		return chunks

	return [
		chunks[0],  # intro - first chunk
		chunks[len(chunks) // 2],  # core - middle chunk
		chunks[-1]  # conclusion - last chunk
	]


def tavily_search(search_queries, max_results=3,
                  topic: Literal["general", "news", "finance"] = "general",
                  include_raw_content=True) -> List[dict]:
	"""
	Perform search using Tavily API for the given query.

	Args:
		search_queries: List of search queries to execute
		max_results: Maximum number of results per query - defaults to 3
		topic: Topic filter for search results - defaults to general
		include_raw_content: Whether to include raw webpage content

	Returns:
		List of search result dictionaries
	"""

	# Web Sseach for each query
	search_docs = []
	for query in search_queries:
		result = tavily_client.search(query=query,
									  max_results=max_results,
									  include_raw_content=include_raw_content,
									  topic=topic)
		search_docs.append(result)

	return search_docs


def summarize_webpage_content(webpage_content: str) -> str | list[Any]:
	"""
    Summarize webpage content using the LLM model.

    Args:
        webpage_content: Raw webpage content to summarize

    Returns:
        Formatted summary with key excerpts
    """
	try:
		full_chunks = chunk_text_by_tokens(webpage_content,
		                                   model="llama-3.1-8b-instant",
		                                   chunk_token_limit=3000,
		                                   overlap_tokens=500)

		# Select ROI chunks - to accomodate token limit - Assuming these chunks are good enough to be summarized
		roi_chunks = select_chunks(full_chunks)
		# Generate summary for each chunk and create a consolidated summary
		chunk_summaries = []
		chunk_ctr = 1
		for chunk in roi_chunks:
			summary = summarization_llm.invoke(
				[HumanMessage(content=summarize_webpage_prompt.format(webpage_content=chunk))])
			chunk_summaries.append(summary.content)
			chunk_ctr += 1

		final_summary = summarization_llm.invoke(
			[HumanMessage(content=combine_summaries_prompt.format(summaries=(" ".join(chunk_summaries))))])

		return final_summary.content

	except Exception as e:
		# Content might not be suitable to be summarized - return raw webpage content
		print("Web page summarization error: ", e)
		return webpage_content[:1000] + "..." if len(webpage_content) > 1000 else webpage_content


def deduplicate_search_results(search_results: List[dict]) -> dict:
	"""
    Deduplicate search results by URL to avoid processing duplicate content.

    Args:
        search_results: List of search result dictionaries

    Returns:
        Dictionary mapping URLs to unique results
    """
	unique_results = {}

	for response in search_results:
		for result in response['results']:
			url = result['url']
			if url not in unique_results:
				unique_results[url] = result

	return unique_results


def process_search_results(unique_results: dict) -> dict:
	"""
    Consolidate summarized content from each of the search results.

    Args:
        unique_results: Dictionary of unique search results

    Returns:
        Dictionary of processed results with summaries
    """
	summarized_results = {}
	result_ctr = 1

	for url, result in unique_results.items():
		# Pause execution for 15 seconds before processing next search result to deal with token limit per minute for the LLMs
		time.sleep(15)

		if not result.get("raw_content"):
			result_ctr += 1
			continue
		else:
			# Summarize raw content for better processing
			content = summarize_webpage_content(result['raw_content'])

		summarized_results[url] = {
			'title': result['title'],
			'content': content
		}
		result_ctr += 1

	return summarized_results


def format_search_output(summarized_results: dict) -> str:
	"""
    Format summarized search results into a well-structured string output.

    Args:
        summarized_results: Dictionary of processed search results

    Returns:
        Formatted string of search results with clear source separation
    """
	if not summarized_results:
		return "No valid search results found. Please try different search queries or use a different search API."

	formatted_output = "Search results: \n"

	for i, (url, result) in enumerate(summarized_results.items(), 1):
		formatted_output += f"\n--- SOURCE {i}: {result['title']} ---\n"
		formatted_output += f"\nURL: {url}\n"
		formatted_output += f"\nSUMMARY:\n{result['content']}\n"
		formatted_output += "-" * 80 + "\n"

	return formatted_output


def extract_tool_content(messages):
	"""
	Extract search tool results from the researcher state
	"""
	contents = []
	for msg in messages:
		if isinstance(msg, ToolMessage) and msg.content:
			contents.append(msg.content)
	return "\n\n".join(contents)
