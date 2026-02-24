# %% [markdown]
# ## 1. Imports & Set up
## .venv/Scripts/activate
## 

# %% Imports from other files
from models import InputTicker_State, ArticleLLMFields, ArticleFields
from UserInputHandling import llm_extract, validate_input, handle_ticker_input, get_jsonparsed_data
from NewsExtractnProcess import search_query, deduplicate_aggressive, generate_internet_search_query, build_user_prompt, run_parallel_searches, \
    analyse_one_article_llm, process_all_articles_node, create_finalReport_node

# %%
import pandas as pd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os
import asyncio
import json
import certifi
import json 

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

# %% [markdown]
# ## 5. Streamlit UI: Async Runner and Graph Progress Tracking

#%% 
MAX_ATTEMPTS = 3

st.set_page_config(page_title="Stock News Workflow", layout="wide")
st.title("Stock News Workflow")

# ---- session state ----
st.session_state.setdefault("attempts", 0)
st.session_state.setdefault("pending_prompt", None)
st.session_state.setdefault("final_state", None)
st.session_state.setdefault("progress_lines", [])

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
#%%
if st.session_state.final_state is not None:
    st.success("Done.")
    result = st.session_state.final_state

    ## Section 1 : Summary + Actions
    st.subheader("News Summary & Actionables")
    st.markdown(result.get("news_answer", "No report generated."))
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("### 🐂 Bullish actions")
        st.markdown(result.get("bullish_act", "No bullish actions generated."))

    with col2:
        st.markdown("### 🐻 Bearish actions")
        st.markdown(result.get("bearish_act", "No bearish actions generated."))

    st.divider()

    ## Section 2: Query & Company Info
    st.subheader("🏢 Query & company context")

    left, right = st.columns([1, 1], gap="large")
    with left:
        st.markdown("**Company**")
        st.code(result.get("ticker", "—") + " - " + result.get("name", "—"), language=None)

        st.markdown("**Industry**")
        st.write(result.get("industry", "—"))

    with right:
        st.markdown("**User query**")
        st.write(result.get("user_query", "—"))

        st.markdown("**Extracted internet search queries**")
        st.markdown("\n".join([f"- 🔎 `{q}`" for q in result.get("internet_search_query", [])]))
        
    st.markdown("**Firm description**")
    st.write(result.get("firm_description", "—"))

    st.divider()

    ## Section 3: News Article 
    st.subheader("📰 Relevant news articles")
    news_article = result.get("useful_articles_list", []) or []

    for idx, art in enumerate(news_article, start=1):
        url = art.get("news_url", "") or ""
        title = art.get("title", "") or ""
        score = art.get("score", None)
        published_date = art.get("published_date", "")

        llm_output = art.get("llm_output", {}) or {}
        primary_topic = llm_output.get("primary_topic", "") or ""
        keep_reasons = llm_output.get("keep_reasons", "")
        topics = llm_output.get("topics", [])
        text_for_rag = llm_output.get("text_for_rag", "")

        # Preview row (non-expanded)
        # Your requirement: preview shows news_url, title, published date, primary topic
        expander_label = f"{idx}. {title or '(untitled)'}"

        with st.expander(expander_label, expanded=False):
            # Preview (top)
            pcol1, pcol2 = st.columns([3, 2], gap="large")
            with pcol1:
                st.markdown("**News URL**")
                if url:
                    st.markdown(f"[{url}]({url})")
                else:
                    st.write("—")

                st.markdown("**Title**")
                st.write(title or "—")

            with pcol2:
                st.markdown("**Published date**")
                st.write(published_date or "—")

                st.markdown("**Primary topic**")
                st.write(primary_topic or "—")

            st.divider()

            # Expanded details
            dcol1, dcol2 = st.columns([1, 1], gap="large")
            with dcol1:
                st.markdown("**Score**")
                st.write(score if score is not None else "—")

            with dcol2:
                st.markdown("**Topics**")
                st.write("\n".join([f"- 🧩 {t}" for t in topics]))

            st.markdown("**Keep reasons**")
            st.write("\n".join([f"- 📝 {r}" for r in keep_reasons]))

            st.divider()
            st.code(text_for_rag, language=None)

    st.divider()
    st.json(st.session_state.final_state)

    if st.button("Reset"):
        st.session_state.attempts = 0
        st.session_state.pending_prompt = None
        st.session_state.final_state = None
        st.session_state.progress_lines = []
        st.rerun()

        st.stop()

#%%
# ---- attempt 1 ----
if st.session_state.attempts == 0 and st.session_state.pending_prompt is None:
    user_text = st.chat_input("Enter your US stock/news query (e.g., 'NVDA latest earnings')")
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

#%%
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
        push_progress(f"\n 👤 Attempt {st.session_state.attempts}/3: retry input")

        s = initial_state.model_copy(deep=True, update={"user_query": retry_text, "error_message": ""})

        status, out = run_async(run_with_progress_async(s))
        if status == "interrupt":
            st.session_state.pending_prompt = out.get("prompt", "Please retry:")
        else:
            st.session_state.final_state = out
            st.session_state.pending_prompt = None

        st.rerun()



