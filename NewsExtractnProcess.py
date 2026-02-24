# %% [markdown]
# ## 3. News Search Section

# %% Imports
from models import InputTicker_State, ArticleLLMFields, ArticleFields
import pandas as pd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os
import asyncio
import json

from difflib import SequenceMatcher
import logging

# %%
from langchain_openai import OpenAIEmbeddings
from typing import Dict, List, Any, Optional
from openai import OpenAI
from tavily import AsyncTavilyClient

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
            max_results=3,
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

def build_user_prompt(ticker, industry, article: Dict[str, Any], subject_hint: Optional[str] = None) -> str:
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
        - Decide keep or not, keep_score and keep_reason. If the article does not contain any related information to the ticker {ticker} or industry {industry}, do not keep it. 
        If the article is mostly paywalled content, boilerplate, or not substantively about the subject, do not keep it.
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
    max_retries: int = 1,
    ticker = "",
    industry = "",
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
            prompt = build_user_prompt(ticker, industry, article, subject_hint=subject_hint)

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
            llm_model= llm_model,
            ticker = state.ticker,
            industry = state.industry,
        )
        for article in articles
    ]
    
    results = await asyncio.gather(*tasks)
    
    # Separate successful and failed articles
    processed_articles = []
    failed_articles = []
    
    for i, result in enumerate(results):
        if result != {} and result.get("llm_output", {}).get("keep") is True:
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

    top_k = kept[:max(4, len(kept))]  # take top 4 or all if less than 4

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
