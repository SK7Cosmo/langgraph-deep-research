"""Prompt templates for the deep research system.

This module contains all prompt templates used across the research workflow components,
1. clarify_with_user_instructions: scoping phase
2. transform_messages_into_research_topic_prompt: scoping phase
3. research_agent_prompt: researching phase
4. summarize_webpage_prompt: researching phase
5. combine_summaries_prompt: researching phase
6. clean_research_findings_prompt: researching phase
"""

clarify_with_user_instructions = """
Check if the user request is clear enough to start research.

Messages:
{messages}

Decide:
- Do you need clarification?
- Or can you proceed?

Rules:
- Ask only if necessary
- Do not make assumptions on preferences
- Do NOT repeat what user already said
- Be concise

Output format:

need_clarification: true/false
question: <your question or empty>
verification: <short message if no clarification>

If asking:
need_clarification: true
question: <your question>
verification:

If not asking:
need_clarification: false
question:
verification:

For the verification message when no clarification is needed:
- Acknowledge that you have sufficient information to proceed
- Briefly summarize the key aspects of what you understand from their request
- Confirm that you will now begin the research process
"""


transform_messages_into_research_topic_prompt = """
Convert the user conversation into ONE clear research question.

Messages:
{messages}

Rules:
- Include all user-provided details
- Do NOT assume missing preferences
- When research quality requires considering additional dimensions that the user hasn't specified, acknowledge them as open considerations rather than assumed preferences.
- Keep it specific and actionable
- Phrase the request from the perspective of the user.

IMPORTANT:
- The question should naturally guide research toward reliable sources such as:
  official websites, government publications, academic research, and verified reports

Output:
One clear research question only.
"""


research_agent_prompt = """
You are a research agent.

Goal:
Find information using tools.

Tools:
- tavily_search_tool → search web
- think_tool → reflect

STRICT RULES:
- Prefer official, authoritative, or content-rich sources
- Avoid aggregator/listing pages
- Avoid pages with only names, ratings, or short descriptions
- Only select sources that provide meaningful insights, explanations, or data
- Do not call tavily_search_tool more than 2 times

Process:
1. Search broadly
2. Evaluate source quality BEFORE using it
3. Ignore low-value sources
4. After each search → use think_tool
5. Identify missing info
6. Search again if needed
7. Stop when answer is clear OR Stop after calling tavily_search_tool 2 times, whichever is earlier
"""

summarize_webpage_prompt = """
Summarize the text below.

Rules:
- Use 5–8 bullet points
- Each bullet ≤ 15 words
- Keep key facts, numbers, names, dates
- No repetition
- Do NOT add new information

Text:
{webpage_content}

Output:
- point 1
- point 2
"""


combine_summaries_prompt = """
You are given multiple summaries from different parts of the same webpage.

Your task:
Combine them into ONE structured summary WITHOUT losing any information.

Rules:
- Include ALL points from all summaries
- Do NOT omit any detail
- Merge duplicate or similar points
- Preserve facts, numbers, names, and dates
- Do NOT add new information

Output:
- Use bullet points
- One bullet per unique point
- No limit on number of bullets
- Keep each bullet clear and concise

Summaries:
{summaries}
"""


clean_research_findings_prompt = """
Organize the research findings clearly.

Tasks:
- Keep ALL important facts
- Remove duplicates
- Ignore internal thoughts (think_tool)
- IGNORE low-quality or listing-style sources
- Prefer sources with real insights or explanations

CRITICAL CITATION RULES:
- EVERY finding MUST end with a citation in the format [n]
- Do NOT write any finding without a citation
- The citation number [n] MUST correspond to the source in the Sources section
- If multiple sources support a fact, use multiple citations like [1][2]
- NEVER invent citations
- NEVER leave a finding without a source reference
- Each source should start in a new line

Format:

Queries:
- ...

Findings:
- Fact 1 [1]
- Fact 2 [2]
- Fact supported by multiple sources [1][3]

Sources:
[1] Title - URL
[2] Title - URL
[3] Title - URL

Rules:
- Do NOT invent information
- Do NOT drop important facts
- Keep it structured and clean
- Ensure ALL findings are traceable to Sources via [n]
"""