# %% [markdown]
# ## 2. Input Handling & Global State Set up

# %% Imports 
from models import InputTicker_State
# %%
import pandas as pd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os
import asyncio
import json
import certifi
from urllib.request import urlopen
import logging
import streamlit as st

# %%
from langchain_openai import OpenAIEmbeddings
from langgraph.types import interrupt
from typing import Dict, Any
from openai import OpenAI

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
# %%
def llm_extract(state: InputTicker_State) -> dict:
    """
    Behaviour:
    - If Streamlit already provided state.user_query AND no error_message: do not interrupt
    - Else: interrupt to request input (retry flow)
    """

    # Decide whether to ask user again
    print(f"LLM Extract received state: user_query='{state.user_query}', error_message='{state.error_message}'")
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
