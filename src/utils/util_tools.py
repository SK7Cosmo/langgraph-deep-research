# Imports
from typing_extensions import Literal # To define constants
from typing_extensions import Annotated # To support older python versions
from langchain_core.tools import tool, InjectedToolArg # Tool oriented imports

# Custom Imports
from src.utils.util_functions import tavily_search, deduplicate_search_results, process_search_results, format_search_output

@tool(parse_docstring=True)
def tavily_search_tool(
    query: str,
    max_results: Annotated[int, InjectedToolArg] = 3,
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general") -> str:
    """
    Fetch results from Tavily search API and apply content summarization logic.
    InjectedToolArg: To ensure that these pasrams are hidden to the LLM to avoid hallucinations

    Args:
        query: A single search query to execute
        max_results: Maximum number of results to return
        topic: Topic to filter results by ('general', 'news', 'finance')

    Returns:
        Formatted string of search results with summaries
    """
    print("\n\n\nNew Search Query being processed...")
    print(query)

    # Execute search for the query - asynchronous
    search_results = tavily_search([query],
                                    max_results=max_results,
                                    topic=topic,
                                    include_raw_content=True)


    # Deduplicate results by URL to avoid processing duplicate content
    unique_results = deduplicate_search_results(search_results)

    # Prepare summarization of each of the search results
    summarized_results = process_search_results(unique_results)

    # Format output for consumption
    return format_search_output(summarized_results)


@tool(parse_docstring=True)
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making.

    Use this tool after each search to analyze results and plan next steps systematically.
    This creates a deliberate pause in the research workflow for quality decision-making.

    Reflection should address:
    1. Analysis of current findings - What concrete information have I gathered?
    2. Gap assessment - What crucial information is still missing?
    3. Quality evaluation - Do I have sufficient evidence/examples for a good answer?
    4. Strategic decision - Should I continue searching or provide my answer?

    Args:
        reflection: Your detailed reflection on research progress, findings, gaps, and next steps

    Returns:
        Confirmation that reflection was recorded for decision-making
    """
    return f"Reflection recorded: {reflection}"