import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from src.graph.state import AgentState
from src.tools.api import get_company_news, get_insider_trades
from src.utils.llm import call_llm
from src.utils.progress import progress


class NewsAnalysisResult(BaseModel):
    """Results from news analysis."""

    relevant_news: List[Dict] = []
    summary: str = ""
    sentiment: str = "neutral"
    confidence: float = 0.0
    key_events: List[str] = []
    potential_impact: str = ""


class NewsAnalyzer:
    """Class for retrieving and analyzing company news."""

    def __init__(
        self,
        model_name: str = "gpt-4o",
        model_provider: str = "openai",
    ):
        """Initialize the NewsAnalyzer.

        Args:
            model_name: The name of the LLM model to use for analysis
            model_provider: The provider of the LLM model
        """
        self.model_name = model_name
        self.model_provider = model_provider

    def fetch_news(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        max_news_per_ticker: int = 50,
    ) -> Dict[str, List[Dict]]:
        """Fetch news for a list of company tickers.

        Args:
            tickers: List of company ticker symbols
            start_date: Start date in format YYYY-MM-DD
            end_date: End date in format YYYY-MM-DD
            max_news_per_ticker: Maximum number of news items to fetch per ticker

        Returns:
            Dictionary mapping tickers to lists of news items
        """
        progress.update_status("news_agent", None, "Fetching news data...")
        all_news = {}

        for ticker in tickers:
            progress.update_status("news_agent", None, f"Fetching news for {ticker}")
            try:
                news_items = get_company_news(
                    ticker=ticker,
                    end_date=end_date,
                    start_date=start_date,
                    limit=max_news_per_ticker,
                    get_all_news=False,
                )
                # Convert to dictionaries for easier manipulation
                news_dicts = [news.model_dump() for news in news_items]
                all_news[ticker] = news_dicts
                progress.update_status("news_agent", None, f"Found {len(news_dicts)} news items for {ticker}")
            except Exception as e:
                progress.update_status("news_agent", None, f"Error fetching news for {ticker}: {str(e)}")
                all_news[ticker] = []

        return all_news

    def analyze_news(
        self,
        ticker: str,
        news_items: List[Dict],
    ) -> NewsAnalysisResult:
        """Analyze news items to identify market-relevant information.

        Args:
            ticker: Company ticker symbol
            news_items: List of news items to analyze

        Returns:
            NewsAnalysisResult with relevant news and summary
        """
        if not news_items:
            return NewsAnalysisResult(
                relevant_news=[],
                summary=f"No news found for {ticker} in the specified date range.",
                sentiment="neutral",
                confidence=0.0,
                key_events=[],
                potential_impact="No news to analyze.",
            )

        progress.update_status("news_agent", None, f"Analyzing {len(news_items)} news items for {ticker}")

        # Create a prompt for the LLM
        template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a financial analyst specializing in identifying market-relevant news. 
                    Your task is to analyze news articles about a company and identify those that could potentially 
                    impact its stock price. Focus on events like:
                    
                    1. Earnings reports and financial performance
                    2. Management changes or restructuring
                    3. Product launches or discontinuations
                    4. Mergers, acquisitions, or divestitures
                    5. Regulatory actions or legal issues
                    6. Industry trends affecting the company
                    7. Competitor moves that could impact the company
                    8. Macroeconomic factors specifically affecting the company
                    
                    For each relevant news item, extract:
                    - The key information
                    - Why it matters to investors
                    - Potential impact on stock price (positive, negative, or neutral)
                    
                    Then provide an overall summary of the key events and their potential impact.
                    """,
                ),
                (
                    "human",
                    """Please analyze the following news articles for {ticker} to identify those that could impact its stock price:
                    
                    {news_items}
                    
                    Return your analysis in the following JSON format:
                    {{
                      "relevant_news": [
                        {{
                          "title": "Title of the article",
                          "date": "YYYY-MM-DD",
                          "key_info": "Brief summary of key information",
                          "relevance": "Why this matters to investors",
                          "potential_impact": "positive|negative|neutral|mixed"
                        }}
                      ],
                      "summary": "Overall summary of key events",
                      "key_events": ["List of 3-5 key events"],
                      "potential_impact": "Analysis of potential impact on stock price"
                    }}
                    
                    Only include truly relevant news that could impact the stock price. Exclude routine updates or general market news unrelated to this specific company.
                    """,
                ),
            ]
        )

        # Prepare news items for the prompt
        limited_news = news_items[:50]  # Limit to prevent context overflow
        news_contexts = []

        for i, news in enumerate(limited_news):
            news_context = f"News #{i+1}\n" f"Title: {news['title']}\n" f"Date: {news['date']}\n" f"Source: {news['source']}\n" f"Author: {news['author'] if news.get('author') else 'N/A'}\n" f"URL: {news['url']}\n\n"
            news_contexts.append(news_context)

        # Create prompt with the news contexts
        prompt = template.invoke({"ticker": ticker, "news_items": "\n".join(news_contexts)})

        # Define default result in case of error
        def create_default_result():
            return NewsAnalysisResult(relevant_news=[], summary=f"Failed to analyze news for {ticker}.", sentiment="neutral", confidence=0.0, key_events=[], potential_impact="Analysis failed.")

        # Call the LLM
        try:
            result = call_llm(
                prompt=prompt,
                model_name=self.model_name,
                model_provider=self.model_provider,
                pydantic_model=NewsAnalysisResult,
                agent_name="news_agent",
                default_factory=create_default_result,
            )
            return result
        except Exception as e:
            progress.update_status("news_agent", None, f"Error analyzing news for {ticker}: {str(e)}")
            return create_default_result()

    def batch_analyze(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        max_news_per_ticker: int = 50,
    ) -> Dict[str, NewsAnalysisResult]:
        """Fetch and analyze news for multiple tickers.

        Args:
            tickers: List of company ticker symbols
            start_date: Start date in format YYYY-MM-DD
            end_date: End date in format YYYY-MM-DD
            max_news_per_ticker: Maximum news items per ticker

        Returns:
            Dictionary mapping tickers to analysis results
        """
        # Fetch all news first
        all_news = self.fetch_news(tickers, start_date, end_date, max_news_per_ticker)

        # Analyze each ticker's news
        results = {}
        for ticker, news_items in all_news.items():
            results[ticker] = self.analyze_news(ticker, news_items)
            progress.update_status("news_agent", ticker, "Done")

        return results

    def get_news_for_period(
        self,
        tickers: List[str],
        end_date: str,
        days_lookback: int = 30,
        max_news_per_ticker: int = 50,
    ) -> Dict[str, NewsAnalysisResult]:
        """Convenience method to get and analyze news for a specified lookback period.

        Args:
            tickers: List of company ticker symbols
            end_date: End date in format YYYY-MM-DD
            days_lookback: Number of days to look back
            max_news_per_ticker: Maximum news items per ticker

        Returns:
            Dictionary mapping tickers to analysis results
        """
        # Calculate start date based on lookback period
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
        start_date_obj = end_date_obj - timedelta(days=days_lookback)
        start_date = start_date_obj.strftime("%Y-%m-%d")

        return self.batch_analyze(tickers, start_date, end_date, max_news_per_ticker)

    def visualize_news_sentiment(self, results: Dict[str, NewsAnalysisResult]) -> None:
        """Visualize news sentiment across tickers.

        Args:
            results: Dictionary mapping tickers to analysis results
        """
        if not results:
            print("No results to visualize.")
            return

        # Extract sentiment scores (convert to numerical values)
        sentiment_map = {"bullish": 1.0, "bearish": -1.0, "neutral": 0.0, "mixed": 0.5}

        tickers = []
        sentiment_scores = []
        confidence_scores = []

        for ticker, result in results.items():
            tickers.append(ticker)
            sentiment_scores.append(sentiment_map.get(result.sentiment, 0.0))
            confidence_scores.append(result.confidence / 100.0)  # Normalize to 0-1

        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Plot sentiment scores
        bars = ax1.bar(tickers, sentiment_scores, color=["green" if s > 0.5 else "red" if s < -0.5 else "orange" if s != 0 else "gray" for s in sentiment_scores])
        ax1.set_title("News Sentiment by Ticker")
        ax1.set_ylabel("Sentiment (-1: Bearish, 0: Neutral, 1: Bullish)")
        ax1.axhline(y=0, color="gray", linestyle="-", alpha=0.3)

        # Plot confidence scores
        ax2.bar(tickers, confidence_scores, color="skyblue")
        ax2.set_title("Confidence in Sentiment Analysis")
        ax2.set_ylabel("Confidence (0-1)")
        ax2.set_ylim(0, 1)

        plt.tight_layout()
        plt.show()


def news_agent(state: AgentState):
    """Main function for the news agent."""
    # Extract parameters from state
    tickers = state["data"]["tickers"]
    start_date = state["data"]["start_date"]
    end_date = state["data"]["end_date"]
    model_name = state["metadata"]["model_name"]
    model_provider = state["metadata"]["model_provider"]

    if os.getenv("USE_NEWS_INSIGHTS", "false").lower() == "true":
        # Initialize the NewsAnalyzer
        analyzer = NewsAnalyzer(model_name=model_name, model_provider=model_provider)

        # Fetch and analyze news
        results = analyzer.batch_analyze(tickers=tickers, end_date=end_date, start_date=start_date, max_news_per_ticker=50)

    else:
        results = {}
        for ticker in tickers:
            results[ticker] = NewsAnalysisResult(
                relevant_news=[],
                summary="No news found.",
                sentiment="neutral",
                confidence=0.0,
                key_events=[],
                potential_impact="No news to analyze.",
            )
    # Visualize the results
    # analyzer.visualize_news_sentiment(results)
    state["data"]["news_analysis"] = results

    # Return the analysis results
    return {"data": state["data"]}
