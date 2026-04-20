# Deep Research Agent

Two pieces of deep research agent developed [independent pieces for easy POC]

- **Scoping Phase** → Understands user intent and generates a research brief  
- **Researching Phase** → Gathers, filters, and synthesizes information and generates consolidated summary for the reearch brief  

Designed for **high-quality, citation-backed insights** while optimizing for **token efficiency and source reliability**.

---

## 🧠 Architecture Overview

### 🔹 1. Scoping Phase (Research Brief Generation)

- Clarifies user intent (if needed)
- Generates a structured **research brief**
- Ensures alignment before deep research begins

#### Flow
<img width="1441" height="799" alt="scoping_phase" src="https://github.com/user-attachments/assets/55494073-3cb4-4d02-80b1-b7ea88dde0c2" />

---

### 🔹 2. Researching Phase (Deep Research Execution)

- Executes iterative search + reasoning loop
- Filters low-quality sources
- Produces:
  - ✅ Consolidated summary  
  - ✅ Report-style findings with citations  

#### Flow
<img width="1218" height="771" alt="researching_phase" src="https://github.com/user-attachments/assets/36f32ec8-6628-4fcb-8d1b-94c71deb1c01" />

---


---

## ⚙️ Models & Tools

| Component | Tool / Model | Purpose |
|----------|-------------|--------|
| 🌐 Web Search | Tavily | Retrieve search results |
| 🧠 Reasoning | llama-3.3-70b-versatile | Generate research brief |
| ✂️ Summarization | llama-3.1-8b-instant | Chunking + summarization |
| 🎯 Orchestration | openai/gpt-oss-120b | Agent control & decision making |

---

## 🔍 Key Features

### ✅ High-Quality Source Filtering
- Skips sources without `raw_content`
- Avoids low-signal aggregator/listing pages
- Focuses on **insightful, content-rich sources**

---

## ✅ Token Optimization Strategy
- Chunk-based summarization
- Avoids full-page ingestion by smart chunk selection strategy

---

### ✅ Rate Limit Handling
- Introduces **15-second delay** between searches  
- Prevents token-per-minute overflow issues  

---


---


## ⚙️ Configuration

### 🔹 1. Create Environment

Create a separate environment and install dependencies:

```bash
python -m venv langchain_env
source langchain_env/bin/activate   # Mac/Linux
langchain_env\Scripts\activate      # Windows

pip install -r requirements.txt
```

---

### 🔹 2. Setup Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_key
TAVILY_API_KEY=your_key
LANGSMITH_TRACING=True
LANGSMITH_API_KEY=your_key
LANGSMITH_PROJECT=deep_research_agent
```

---

## 🔹 3. Setup Jupyter Kernel (Important)

Run these commands **after activating the environment**:

```bash
pip install ipykernel
python -m ipykernel install --user --name=langchain_env --display-name "Python (langchain_env)"
```

Then select **Python (langchain_env)** as the kernel inside Jupyter Notebook.

---

## 🔹 4. LangGraph Deployment

* Ensure `langgraph.json` is present in the project
* Start local deployment:

```bash
langgraph dev
```


