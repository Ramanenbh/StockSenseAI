# %% Imports
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

# %% Main Graph State
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

# %% LLM Output Schema for Article Analysis
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
