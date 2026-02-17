# %% [markdown]
# ## 1. Imports & Set up

# %%
import pandas as pd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os
import asyncio
import json
import certifi

from pymongo import MongoClient, UpdateOne
from pymongo.server_api import ServerApi

from datetime import datetime, timedelta
from urllib.request import urlopen

from difflib import SequenceMatcher
import requests
import logging

import streamlit as st

# %%
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.types import interrupt, Command

from typing import Dict, List, Any, Optional, Literal, Union
from bson import ObjectId
from collections import defaultdict

from openai import OpenAI, AsyncOpenAI
from langgraph.graph import START, END, StateGraph
from pydantic import BaseModel, Field, field_validator

from IPython.display import Image, display

from tavily import TavilyClient, AsyncTavilyClient

# %%
load_dotenv(".env", override=True)

OpenAI_api_key = os.getenv("OpenAI_api_key")
llm_model = "gpt-5-nano"
embed = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OpenAI_api_key)
llm_client = OpenAI(api_key=OpenAI_api_key)

financialmodellingprep_api_key = os.getenv("financialmodellingprep_api_key")

tavily_api_key = os.getenv("tavily_api_key")

# use logging to debug and track flow instead of print statements since we have async code and print statements can get jumbled up in the output, making it hard to follow the flow. 
# With logging we can have timestamps and log levels to better understand the sequence of events and identify where things might be going wrong.
logger = logging.getLogger(__name__)

# %% [markdown]
# ## 2. Input Handling & Global State Set up

# %%
class InputTicker_State(BaseModel):
    news_answer: str = Field("Summarised final news to give to the end user")
    bullish_act: Any = Field("What to do to profit from this information")
    bearish_act: Any = Field("What to do to prevent bad events from affecting my stocks")

    ticker: Optional[str] = Field(description="The stock ticker symbol, e.g., AAPL for Apple Inc.")
    user_query: Optional[str]
    name: str = Field(description="The name of the company associated with the ticker symbol.")
    industry: str = Field(description="The industry of the company associated with the ticker symbol.")
    firm_description: Optional[str] = Field(description="A brief description of the company.")

    exchange: Optional[str] = Field(description="The stock exchange where the ticker is listed, e.g., NASDAQ, NYSE.")
    one_weekb4_today: str = Field(description = "yyyy-mm-dd date that is 1 week b4 today to get latest articles")
    
    is_valid_input: Literal["valid", "invalid"] = Field(description="Indicates whether the input ticker is valid or not.")
    error_message: Optional[str] = ""

    internet_search_query: List[str] = Field(description="The search queries generated using llm based on the user's input.")
    tavily_article_list: Optional[List[Dict]] = Field(description="stores articles from tavily extraction")
    useful_articles_list: Optional[List[Dict]] = Field(description="stores articles as it flows thru the graph")

# %%
def llm_extract(state: InputTicker_State) -> dict:
    """
    Behaviour:
    - If Streamlit already provided state.user_query AND no error_message: do not interrupt
    - Else: interrupt to request input (retry flow)
    """

    # Decide whether to ask user again
    needs_user_input = (not state.user_query) or state.error_message != ""

    if needs_user_input:
        if state.error_message:
            prompt = f"{state.error_message}\n\nPlease retry :"
        else:
            prompt = "Enter your stock/news query or guidance query :"

        user_text = interrupt({"prompt": prompt})
    else:
        user_text = state.user_query  # first run comes from Streamlit

    response = llm_client.chat.completions.create(
        model=llm_model,
        # If your client supports it, keep this (recommended):
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract ONE US-listed stock ticker if explicitly present, and a concise web/news search query.\n"
                    "Rules:\n"
                    "- Output exactly one ticker.\n"
                    "- If multiple tickers appear, choose the most central one.\n"
                    "- Keep the query short and specific.\n"
                    "- Return ONLY a JSON object with fields: ticker, query.\n"
                )
            },
            {"role": "user", "content": str(user_text)},
        ],
    )

    raw = (response.choices[0].message.content or "").strip()
    print(raw)
    answer = json.loads(raw)

    return {
        "ticker": (answer.get("ticker") or "").strip(),
        "user_query": (answer.get("query") or "").strip(),
        "error_message": "",
    }

# %%
def validate_input(state: InputTicker_State) -> InputTicker_State:

    logger.info(f"Validating input ticker: {state.ticker} and user query")
    response = llm_client.chat.completions.create(
        model=llm_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Validate if this is a valid stock and news related search request. Be strict. If it is not valid return an error message saying invalid and"
                    "detailing why its not valid else return 1 word valid"
                )
            },
            {
                "role": "user",
                "content": f"Ticker: {state.ticker}\n Query: {state.user_query}\n"
            }
        ]
    )

    answer = response.choices[0].message.content.strip()

    if answer == "valid":
        return {"is_valid_input": "valid", 
                "error_message": ""
                }
    else:
        return {"is_valid_input": "invalid", 
                "error_message": answer
                }

# %%
def get_jsonparsed_data(url):
    response = urlopen(url, cafile=certifi.where())
    data = response.read().decode("utf-8")
    return json.loads(data)

def handle_ticker_input(state: InputTicker_State) -> Dict[str, Any]:
    search_ticker_endpoint = (
        "https://financialmodelingprep.com/stable/profile?symbol=" + 
        state.ticker + 
        "&apikey=" + financialmodellingprep_api_key + 
        "&limit=1"
    )
    
    try:
        result = get_jsonparsed_data(search_ticker_endpoint)
        logger.info(f"Data for Ticker {result[0]['symbol']}: {result[0]['companyName']} has been extracted")
        
        # Return a dictionary with the fields to update
        return {
            "ticker": result[0]['symbol'],
            "name": result[0]['companyName'],
            "exchange": result[0].get('exchange', None),
            "industry": result[0].get('industry', None),
            "firm_description": result[0].get('description', None)
        }
    
    except IndexError as ie: 
        print("Please key in a valid ticker")
        raise ValueError(f"Invalid ticker: {state.ticker}")
    except Exception as e:
        print(f"Error handling ticker input for {state.ticker}: {e}")
        raise e

# %% [markdown]
# ## 3. News Search Section

# %%
class ArticleLLMFields(BaseModel):
    """Analysis of a news article for relevance and key information extraction"""
    keep: bool = Field(description="Whether to keep this article for further analysis")
    keep_score: float = Field(description="Score indicating relevance/quality of the article (0-1 or 0-10 scale)",ge=0, le=1)
    keep_reason: str = Field(description="Explanation for why the article should be kept or discarded")
    
    primary_topic: str = Field(description="The main topic or theme of the article")
    topics: List[str] = Field(description="List of all relevant topics covered in the article")
    
    stance: Literal["bullish", "bearish", "neutral", "unclear"] = Field(description="Market stance conveyed by the article: bullish (positive), bearish (negative), neutral, or unclear")
    time_horizon: Literal["short", "medium", "long", "unclear"] = Field(description="Time horizon of the impact: short (<1 year), medium (1-3 years), long (>3 years), or unclear")
        
    text_for_rag: str = Field(description="Information-dense text preserving all numbers, dates, and entities for RAG system")
    summary: str = Field(description="Concise summary of the article in 2-6 sentences", min_length=20)

class ArticleFields(BaseModel):
    news_url: str
    title: str
    score: float
    published_date: str
    content: str
    llm_output: ArticleLLMFields

# %%
# OLD NOT USED ANYMORE BUT KEEP FOR REFERENCE
async def search_query(query: str, tavily_client):
    """Run a single Tavily search query asynchronously
    tavily_client.search() is a synchronous blocking function (it's not actually async, even though you wrote async def)
    """
    result = await tavily_client.search(
        query=query,
        max_results=2,
        topic="news",
        search_depth="advanced",
        days=7,
        include_raw_content=True,
        exclude_domains=["linkedin.com", "youtube.com", "wikipedia.com", "facebook.com"]
    )

    return result.get("results", [])

# %%
def deduplicate_aggressive(articles: list[dict], min_score: float = 0.3) -> list[dict]:
    """
    More aggressive deduplication using multiple criteria
    """
    unique_articles = []
    seen_urls = set()
    seen_titles = set()
    
    for article in articles:
        # exclude articles with bad score < 0.3
        if article.get('score', min_score) < min_score:
            continue

        url = article.get('url', '')
        title = article.get('title', '').lower().strip()
        title = title[:title.rfind("-")].strip()
        
        # Skip if URL or exact title already seen
        if url in seen_urls or title in seen_titles:
            continue
        
        # Check for very similar titles (fuzzy matching)
        is_similar = False
        for seen_title in seen_titles:
            similarity = SequenceMatcher(None, title, seen_title).ratio()
            if similarity > 0.75:  # 75% similar
                is_similar = True
                break
        
        if not is_similar:
            unique_articles.append(article)
            seen_urls.add(url)
            seen_titles.add(title)
    
    logger.info(f"After removing duplicates, unique articles count: {len(unique_articles)}")
    return unique_articles

# %%
def generate_internet_search_query(state: InputTicker_State) -> Dict[str, List[str]]:
    """
    Generate 2 concise web/news search queries and store them in state.internet_search_query (List[str]).
    """

    response = llm_client.chat.completions.create(
        model=llm_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You create web/news search queries.\n"
                    "Return ONLY valid JSON with this exact schema:\n"
                    "{\n"
                    '  "internet_search_query": ["<query1>", "<query2>"]\n'
                    "}\n"
                    "Rules:\n"
                    "You create web/news search queries for INVESTOR GUIDANCE monitoring.\n"
                    "Return ONLY valid JSON with this exact schema:\n"
                    "{\n"
                    '  "internet_search_query": ["<query1>", "<query2>"]\n'
                    "}\n"
                    "Rules:\n"
                    "- Exactly 2 queries.\n"
                    "- Each query <= 12 words.\n"
                    "- Query1 MUST target guidance/earnings changes (e.g., guidance raised/cut, outlook, forecast, EPS/revenue).\n"
                    "- Query2 MUST target context/drivers (industry trend, demand, pricing, regulation, competitors).\n"
                    "- Always include the anchor (ticker and/or company) in Query1.\n"
                    "- Query2 may include anchor OR just industry/macro terms depending on user intent.\n"
                    "- Use widely used finance terms: earnings, guidance, outlook, forecast, raises, cuts, revises.\n"
                    "- No bullet points, no extra text, JSON only."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"User original query: {state.user_query}\n"
                    f"Validated ticker: {state.ticker}\n"
                    f"Company name: {state.name}\n"
                    f"Industry: {state.industry}\n"
                ),
            },
        ],
    )

    raw = (response.choices[0].message.content or "").strip()

    # Parse JSON robustly
    try:
        data = json.loads(raw)
        queries = data.get("internet_search_query", [])

    except json.JSONDecodeError:
        # Fallback: split lines / bullets if model didn't follow instructions
        lines = [ln.strip("-• \t") for ln in raw.splitlines() if ln.strip()]
        queries = lines[:2]

    # Final sanitisation
    queries = [q.strip() for q in queries if isinstance(q, str) and q.strip()]

    # Ensure exactly 2 queries (fallback if needed)
    if len(queries) < 2:
        queries = queries + [
            f"{state.ticker} latest news",
            f"{state.industry} latest news",
        ]
        queries = queries[:2]

    elif len(queries) > 2:
        queries = queries[:2]

    return {"internet_search_query": queries}

# %%
async def run_parallel_searches(state: InputTicker_State) -> Dict[str, Any]:
    # Initialize async client
    tavily_client = AsyncTavilyClient(api_key=tavily_api_key)

    """Run both queries in parallel"""
    query1 = state.internet_search_query[0]
    query2 = state.internet_search_query[1]
    
    logger.info(f"Etracting news from online sites using tavily")
    # Run both queries concurrently
    results = await asyncio.gather(
        *(tavily_client.search(
            query=q,
            max_results=1,
            topic="news",
            search_depth="advanced",
            days=7,
            include_raw_content=True,
            exclude_domains=["linkedin.com", "youtube.com", "wikipedia.com", "facebook.com"]
            )
        for q in [query1, query2]
        ) 
    )
    
    articles_query1 = results[0].get("results", [])
    articles_query2 = results[1].get("results", [])
    
    # Combine results if needed
    all_articles = deduplicate_aggressive(articles_query1 + articles_query2)
    # print(all_articles)
    logger.info(f"Extracted news from online sites using tavily: {len(all_articles)} articles")
    logger.info(f"Processing {len(all_articles)} articles...")
    
    return {"tavily_article_list" : all_articles}

# %%
SYSTEM_PROMPT = f"""You are a financial-news traige assistant.
    Use ONLY the provided article content and metadata. Do not add external facts.
    If the content is paywalled, boilerplate, or not substantively about the subject, set keep=false.

    Output MUST be valid JSON matching the schema.
"""

def build_user_prompt(article: Dict[str, Any], subject_hint: Optional[str] = None) -> str:
    """
    subject_hint: optional (e.g., ticker name) to help the LLM judge relevance when extending project to 1 stock.
    """
    url = article.get("url", "")
    title = article.get("title", "")
    published_date = article.get("published_date", "")
    score = article.get("score", None)
    content = (article.get("content") or "").strip()

    # Keep prompts stable: truncate very long pages
    if len(content) > 14000:
        content = content[:11000] + "\n...\n" + content[-2500:]

    hint_line = f"Subject hint: {subject_hint}\n" if subject_hint else ""

    return f"""{hint_line} Metadata:
        - url: {url}
        - title: {title}
        - published_date: {published_date}
        - score: {score}

        Task:
        Read the article content and produce JSON fields as per the schema:
        - Decide keep or not, keep_score and keep_reason
        keep_reason must be of type string
        - Identify primary_topic, topics
        - Determine stance used only these values - ("bullish", "bearish", "neutral", "unclear")
        - Determine time_horizon of the impact based on claims, use only these values - ("short", "medium", "long") where short is <1Year, medium is 1-3 years and long is >3 years
        For stance & time_horizon, if unsure simply return unclear
        - Write text_for_rag (information-dense; preserve numbers/dates/entities)
        - Write summary (2–6 sentences)
        Note that text_for_rag and summary fields must be of type string

        Article content:
        \"\"\"{content}\"\"\"
        """

# %%
async def analyse_one_article_llm(
    llm_client,
    article: Dict[str, Any],
    llm_model: str,
    subject_hint: Optional[str] = None,
    max_retries: int = 1
) -> Dict[str, Any]:
    """
    Returns a new dict:
      {
        ...original article fields...,
        "llm": <ArticleLLMFields as dict>
      }
    Raises if the LLM response can't be validated after retries (handled by caller).
    """

    for attempt in range(max_retries + 1):
        try:
            prompt = build_user_prompt(article, subject_hint=subject_hint)

            resp = llm_client.chat.completions.create(
                model=llm_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )

            raw = (resp.choices[0].message.content or "").strip()

            # Validate structured output
            parsed = ArticleLLMFields.model_validate_json(raw)

            article_fields = ArticleFields(
                news_url=article.get("url", "No url"), 
                title=article.get("title", "No title"),
                score=article.get("score", 0.0),
                published_date=article.get("published_date", "No date"),
                content=article.get("content", "No content"),
                llm_output=parsed
            )
            
            return article_fields.model_dump()
        
        # handle failure logic - by retrying twice
        except Exception as e:
            if attempt < max_retries:
                print(f"Attempt {attempt + 1} failed for {article.get('title', 'Unknown title')} due to {e}")
                await asyncio.sleep(1.5)
            else:
                print(f"Final attempt failed for {article.get('title', 'Unknown title')}")
                return {}


# %%
async def process_all_articles_node(state: InputTicker_State) -> Dict[str, Any]:
    """
    LangGraph node that processes all articles with LLM analysis.
    
    Expected state keys:
        - articles: List[Dict[str, Any]] - processed articles from Tavily + LLM output
        - client: OpenAI client
        - model: str - model name
        - subject_hint: Optional[str] - hint for analysis
    
    Returns updated state with:
        - processed_articles: List[ArticleFields] - successfully processed articles
    """
    articles = state.tavily_article_list
    
    # Process all articles concurrently with retry logic
    tasks = [
        analyse_one_article_llm(
            llm_client= llm_client,
            article= article,
            llm_model= llm_model
        )
        for article in articles
    ]
    
    results = await asyncio.gather(*tasks)
    
    # Separate successful and failed articles
    processed_articles = []
    failed_articles = []
    
    for i, result in enumerate(results):
        if result != {}:
            processed_articles.append(result)
        else:
            failed_articles.append(articles[i])
    
    print(f"Successfully processed: {len(processed_articles)}/{len(articles)}")
    print(f"Failed: {len(failed_articles)}/{len(articles)}")
    
    return {
        "useful_articles_list": processed_articles,
    }

# %%
def create_finalReport_node(state: InputTicker_State) -> InputTicker_State:
    logger.info(f"Extracted trading report from LLM processed articles")
    articles = state.useful_articles_list

    kept = [a for a in articles if a.get("llm_output").get("keep") is True]
    kept.sort(key=lambda x: x.get("llm_output").get("keep_score", 0.0), reverse=True)

    top_k = kept[:2]

    evidence_lines = []
    for index, article in enumerate(top_k, start=1):
        llm_article = article.get("llm_output") or {}

        primary_topic = llm_article.get("primary_topic", "")
        stance =  llm_article.get("stance", "unclear")
        horizon = llm_article.get("time_horizon", "unclear")  # your schema uses time_horizon
        summary = llm_article.get("summary", "")

        evidence_lines.append(
            f"[{index}] topic={primary_topic} | stance={stance}/{horizon}\n"
            f"summary: {summary}"
        )

    evidence = "\n\n".join(evidence_lines)

    response = llm_client.chat.completions.create(
        model=llm_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "I am a finance news synthesiser. I only use the provided article evidence for facts. "
                    "I provide general risk-management and idea-generation guidance for stock trading for RETAIL TRADERS! "
                    "I keep it concise and structured. "
                    "Return ONLY valid JSON matching the schema."
                ),
            },
            {
                "role": "user",
                "content": f"""Query: {state.user_query} Do let me know how I should use this information in stock trading.

                    Article evidence (each item is a separate article; treat as potentially overlapping):
                    {evidence}

                    Task:
                    1) YOU MUST produce JSON fields as per the schema to answer the query in 2–4 sentences, synthesising the main themes across the evidence.
                    2) Give actionables to protect against stock shocks: 3–4 bullet points.
                    3) Give potential opportunities: 3–4 bullet points.

                    Constraints:
                    - Do not invent facts not supported by the evidence.
                    - If evidence is mixed/unclear, say so plainly.
                    - Bullets must be concise and practical.

                    Output Schema (JSON only):
                    - news_answer: string (covers the latest important news)
                    - bullish_act: string (actionables to capitalise on these trends)
                    - bearish_act: string (pitfalls / bearish news to be aware of)
                    """,
            },
        ],
    )

    raw = (response.choices[0].message.content or "").strip()

    # Validate structured output (raw is a JSON string, so parse it)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: keep pipeline alive even if the model breaks format
        return {
            "news_answer": "I do not have enough high-confidence articles to answer this query.",
            "bearish_act": "I do not have enough high-confidence articles to answer this query.",
            "bullish_act": "I do not have enough high-confidence articles to answer this query.",
        }

    return {
        "news_answer": parsed.get("news_answer", ""),
        "bullish_act": parsed.get("bullish_act", ""),
        "bearish_act": parsed.get("bearish_act", ""),
    }

# %% [markdown]
# ## 4. Graph Initialisation & Post workflow Handling

# %%
workflow = StateGraph(InputTicker_State)
workflow.add_node("llm_extract", llm_extract)
workflow.add_node("validate_input", validate_input)
workflow.add_node("user_ticker_input", handle_ticker_input)
workflow.add_node("generate_internet_search_query", generate_internet_search_query)
workflow.add_node("Internet_Search_Tool", run_parallel_searches)
workflow.add_node("llm_output_article", process_all_articles_node)
workflow.add_node("llm_generate_ArticleReport", create_finalReport_node)

# Add edges
workflow.add_edge(START, "llm_extract")
workflow.add_edge("llm_extract", "validate_input")
# use path_map for conditional edges instead of ifelse lambda since the mermaid graph visualisation doesnt know what other values can be taken in 
# also cleaner since the graph itself will throw an error if we forget to account for a possible value of is_valid_input
# this is bcos it needs the all possible paths to be defined at graph compile time to visualise them, rather than waiting for runtime like the ifelse lambda function would allow
workflow.add_conditional_edges(
    "validate_input",
    lambda s: s.is_valid_input, 
    path_map={
        "invalid": "llm_extract",
        "valid": "user_ticker_input",
    },
)
workflow.add_edge("user_ticker_input", "generate_internet_search_query")
workflow.add_edge("generate_internet_search_query", "Internet_Search_Tool")
workflow.add_edge("Internet_Search_Tool", "llm_output_article")
workflow.add_edge("llm_output_article", "llm_generate_ArticleReport")
workflow.add_edge("llm_generate_ArticleReport", END)

# Compile the graph
graph = workflow.compile(checkpointer=MemorySaver())

# Create config
config = {"configurable": {"thread_id": "1"}}

# inititalise start state = 
OneWeek_b4 = datetime.now() - timedelta(days = 7)
OneWeek_b4_today = str(OneWeek_b4.year) + "-" + str(OneWeek_b4.month) + "-" + str(OneWeek_b4.day)
initial_state = InputTicker_State(
    ticker= "",
    user_query= "",
    industry= "",
    firm_description= "",
    error_message= "",
    name= "",
    is_valid_input= "valid",
    exchange=None,
    one_weekb4_today=OneWeek_b4_today,
    internet_search_query=[],
    tavily_article_list=None,
    useful_articles_list=None
)

#%%
MAX_ATTEMPTS = 3

st.set_page_config(page_title="Stock News Workflow", layout="wide")
st.title("Stock News Workflow")

# ---- session state ----
st.session_state.setdefault("thread_id", "streamlit-thread-1")
st.session_state.setdefault("attempts", 0)
st.session_state.setdefault("pending_prompt", None)
st.session_state.setdefault("final_state", None)
st.session_state.setdefault("progress_lines", [])

config = {"configurable": {"thread_id": st.session_state.thread_id}}

st.subheader("Progress")
progress_box = st.empty()

def push_progress(line: str):
    st.session_state.progress_lines.append(line)
    st.session_state.progress_lines = st.session_state.progress_lines[-200:]
    progress_box.info("\n".join(st.session_state.progress_lines))

def run_async(coro):
    return asyncio.run(coro)

# ---- async runner with events ----
async def run_with_progress_async(payload):
    """
    payload: initial state OR Command(resume=...)
    Runs the graph (async) and appends progress when watched nodes start.
    Returns ("interrupt", intr) or ("done", values).
    """
    push_progress("🚀 Started \n")

    async for ev in graph.astream_events(payload, config=config, version="v2"):
        ev_type = ev.get("event") or ""
        meta = ev.get("metadata") or {}
        node = meta.get("langgraph_node") or meta.get("node")

        if ev_type == "on_chain_start" and node and node != "__start__":
            push_progress(f"▶️ {node} started \n")

    snap = await graph.aget_state(config)
    if snap.next:
        intr = snap.tasks[0].interrupts[0].value
        push_progress("⏸️ Waiting for input…")
        return ("interrupt", intr)

    push_progress("✅ Finished \n")
    return ("done", snap.values)

# ---- done screen ----
if st.session_state.final_state is not None:
    st.success("Done.")
    st.json(st.session_state.final_state)

    if st.button("Reset"):
        st.session_state.thread_id = st.session_state.thread_id + "-reset"
        st.session_state.attempts = 0
        st.session_state.pending_prompt = None
        st.session_state.final_state = None
        st.session_state.progress_lines = []
        st.rerun()

    st.stop()

# ---- attempt 1 ----
if st.session_state.attempts == 0 and st.session_state.pending_prompt is None:
    user_text = st.chat_input("Enter your stock/news query (e.g., 'NVDA latest earnings')")
    if user_text:
        st.session_state.attempts = 1
        st.session_state.progress_lines = []
        push_progress("👤 Attempt 1/3: received input")

        s = initial_state.model_copy(deep=True, update={"user_query": user_text, "error_message": ""})

        status, out = run_async(run_with_progress_async(s))
        if status == "interrupt":
            st.session_state.pending_prompt = out.get("prompt", "Please retry:")
        else:
            st.session_state.final_state = out

        st.rerun()

# ---- retry attempts ----
if st.session_state.pending_prompt is not None:
    st.caption(f"Attempt {st.session_state.attempts}/{MAX_ATTEMPTS}")
    retry_text = st.chat_input(st.session_state.pending_prompt)

    if retry_text:
        if st.session_state.attempts >= MAX_ATTEMPTS:
            snap = run_async(graph.aget_state(config))
            st.session_state.final_state = snap.values
            st.session_state.pending_prompt = None
            push_progress("🛑 Retry limit reached. Showing current state.")
            st.rerun()

        st.session_state.attempts += 1
        push_progress(f"👤 Attempt {st.session_state.attempts}/3: retry input")

        status, out = run_async(run_with_progress_async(Command(resume=retry_text)))
        if status == "interrupt":
            st.session_state.pending_prompt = out.get("prompt", "Please retry:")
        else:
            st.session_state.final_state = out
            st.session_state.pending_prompt = None

        st.rerun()



