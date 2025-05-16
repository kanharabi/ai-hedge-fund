import json
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
from src.utils.llm import call_llm
from src.utils.progress import progress


class BackTestSummary(BaseModel):
    summary: str = "Summary of the backtest results"


class Backtester:
    def __init__(self, portfolio: Dict[str, float], benchmark: str = "^GSPC", end_date: Optional[datetime] = None):
        """
        Initialize the Backtester

        Args:
            portfolio: Dictionary of portfolio holdings with ticker symbols as keys and weights as values
            benchmark: Ticker symbol for the benchmark index (default: S&P 500)
        """
        self.portfolio = portfolio
        self.benchmark = benchmark
        self.horizons = {"1Y": 365, "2Y": 730, "5Y": 1825, "YTD": (datetime.now() - datetime(datetime.now().year, 1, 1)).days}

        self.max_duration = "5Y"

        if end_date and (datetime.now() - datetime.fromisoformat(end_date)).days > 3:
            self.horizons = {"TILL_DATE": (datetime.now() - datetime.fromisoformat(end_date)).days}
            self.start_from = datetime.fromisoformat(end_date).date()
            self.end_date = datetime.now().date()
            self.max_duration = f"{self.horizons.get('TILL_DATE', 3)}d"
            temp = {}
            if self.horizons.get("TILL_DATE", 3) > 365:
                temp.update({"1Y": 365})
            if self.horizons.get("TILL_DATE", 3) > 730:
                temp.update({"2Y": 730})
            if self.horizons.get("TILL_DATE", 3) > 1825:
                temp.update({"5Y": 1825})
            self.horizons = {**temp, **self.horizons}

    def fetch_historical_data(self, ticker: str, period: str = None) -> pd.DataFrame:
        """
        Fetch historical price data for a given ticker
        """
        if not period:
            try:
                if self.start_from and self.end_date:
                    start = self.start_from
                    end = self.end_date
                    period = None
                else:
                    start = None
                    end = None
                    period = self.max_duration
            except Exception as e:
                start = None
                end = None
                period = self.max_duration
        else:
            start = None
            end = None
            period = self.max_duration

        try:
            data = yf.download(ticker, start=start, end=end, period=period)
            return data[("Close", ticker)]
        except Exception as e:
            print(f"Error fetching data for {ticker}: {str(e)}")
            return pd.DataFrame()

    def calculate_portfolio_returns(self) -> pd.DataFrame:
        """
        Calculate portfolio returns based on holdings
        """
        portfolio_returns = pd.DataFrame()

        # Fetch data for each portfolio holding
        for ticker, weight in self.portfolio.items():
            returns = self.fetch_historical_data(ticker)
            if not returns.empty:
                portfolio_returns[ticker] = returns

        # Calculate portfolio returns
        if not portfolio_returns.empty:
            portfolio_returns = portfolio_returns.pct_change()
            portfolio_returns = (portfolio_returns * pd.Series(self.portfolio)).sum(axis=1)
            portfolio_returns = portfolio_returns.ffill().bfill()
            return portfolio_returns
        return pd.DataFrame()

    def calculate_benchmark_returns(self) -> pd.DataFrame:
        """
        Calculate benchmark index returns
        """
        benchmark_data = self.fetch_historical_data(self.benchmark)
        if not benchmark_data.empty:
            return benchmark_data.pct_change().ffill().bfill()
        return pd.DataFrame()

    def calculate_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """
        Calculate performance metrics
        """
        if returns.empty:
            return {}

        metrics = {}

        # Calculate CAGR (Compound Annual Growth Rate)
        years = (returns.index[-1] - returns.index[0]).days / 365
        cum_return = (1 + returns).prod() - 1  # Total cumulative return
        metrics["CAGR"] = (1 + cum_return) ** (1 / years) - 1

        # Calculate volatility
        metrics["Volatility"] = returns.std() * np.sqrt(252)

        # Calculate Max Drawdown
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        metrics["Max Drawdown"] = drawdown.min()

        return metrics

    def plot_performance(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series):
        """
        Generate performance charts
        """
        plt.figure(figsize=(15, 10))

        # Cumulative Returns
        plt.subplot(2, 1, 1)
        (1 + portfolio_returns).cumprod().plot(label="Portfolio")
        (1 + benchmark_returns).cumprod().plot(label="Benchmark")
        plt.title("Cumulative Returns")
        plt.legend()

        # Drawdowns
        plt.subplot(2, 1, 2)
        cum_portfolio = (1 + portfolio_returns).cumprod()
        cum_benchmark = (1 + benchmark_returns).cumprod()

        max_portfolio = cum_portfolio.cummax()
        max_benchmark = cum_benchmark.cummax()

        drawdown_portfolio = (cum_portfolio - max_portfolio) / max_portfolio
        drawdown_benchmark = (cum_benchmark - max_benchmark) / max_benchmark

        drawdown_portfolio.plot(label="Portfolio Drawdown")
        drawdown_benchmark.plot(label="Benchmark Drawdown")
        plt.title("Drawdowns")
        plt.legend()

        plt.tight_layout()
        plt.savefig("images/performance_benchmark.png")
        # plt.show()

    def run_backtest(self):
        """
        Run the complete backtest and generate results
        """
        portfolio_returns = self.calculate_portfolio_returns()
        benchmark_returns = self.calculate_benchmark_returns()

        if portfolio_returns.empty or benchmark_returns.empty:
            print("Error: Could not fetch required data")
            return

        # Calculate metrics for different horizons
        results = {}

        if self.start_from and self.end_date and self.max_duration != "5Y":
            for horizon, days in self.horizons.items():

                start_date = datetime.combine(self.start_from, datetime.min.time())
                end_date = start_date + timedelta(days=days)

                mask = (portfolio_returns.index >= start_date) & (portfolio_returns.index <= end_date)

                portfolio_period = portfolio_returns[mask]
                benchmark_period = benchmark_returns[mask]

                results[horizon] = {"Portfolio": self.calculate_metrics(portfolio_period), "Benchmark": self.calculate_metrics(benchmark_period)}

        else:
            for horizon, days in self.horizons.items():
                end_date = datetime.combine(datetime.now().date(), datetime.min.time())
                start_date = end_date - timedelta(days=days)

                mask = (portfolio_returns.index >= start_date) & (portfolio_returns.index <= end_date)

                portfolio_period = portfolio_returns[mask]
                benchmark_period = benchmark_returns[mask]

                results[horizon] = {"Portfolio": self.calculate_metrics(portfolio_period), "Benchmark": self.calculate_metrics(benchmark_period)}

        # Plot performance
        self.plot_performance(portfolio_returns, benchmark_returns)

        # Print summary
        # print("\nBacktest Summary:")
        # for horizon, metrics in results.items():
        #     print(f"\n{horizon} Performance:")
        #     print("Portfolio:")
        #     for metric, value in metrics["Portfolio"].items():
        #         print(f"  {metric}: {value:.2%}")
        #     print("Benchmark:")
        #     for metric, value in metrics["Benchmark"].items():
        #         print(f"  {metric}: {value:.2%}")

        return results

    def summarize_results(self, results: Dict[str, any], model_name: str, model_provider: str) -> str:
        """
        Summarize the backtest results
        """
        template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Given the results of a backtest of our portfolio against a benchmark index/stock, summarize the performance in a clear and concise manner.
                    Extract relevant insights from CAGR, volatility, and max drawdown for both the portfolio and the benchmark and Highlight any significant differences in performance.
                    Keep it simple and easy to understand.
                    Results:
                    {results}
                    Return JSON exactly in this format:
                    {{
                    "summary": "Summary of the backtest results",
                    }}
            
                    """,
                )
            ]
        )
        prompt = template.invoke({"results": results})
        return call_llm(
            prompt,
            model_name=model_name,
            model_provider=model_provider,
            max_retries=3,
            pydantic_model=BackTestSummary,
            default_factory=BackTestSummary,
            agent_name="backtester_agent",
        )


def backtester_agent(state: AgentState):
    """
    Backtester agent function to be used with the LLM agent
    """
    portfolio = {}
    total_portfolio_amount = 0
    benchmark = state.get("data", {}).get("benchmark", "VOO")
    start_date = state["data"]["start_date"]
    end_date = state["data"]["end_date"]
    # print(state)

    progress.update_status("backtester_agent", f"{state['data']['tickers']}", "Backtesting the portfolio with the benchmark stock/Index")
    current_portfolio = portfolio_distribution = state.get("data", {}).get("analyst_signals", {}).get("risk_management_agent", {})
    portfolio_distribution = state.get("data", {}).get("analyst_signals", {}).get("portfolio_manager_summary", {}).decisions

    for ticker, decision in portfolio_distribution.items():
        quantity = decision.quantity
        current_position = current_portfolio.get(ticker, {}).get("reasoning", {}).get("current_position", 0.0)
        if decision.action == "short":
            if current_position > 0.0:
                if current_position > decision.quantity:
                    quantity = current_position - decision.quantity
                else:
                    quantity = 0
            else:
                quantity = 0
            # quantity = current_position - decision.quantity
        elif decision.action == "buy":
            quantity = current_position + decision.quantity
        elif decision.action == "hold":
            quantity = current_position
        elif decision.action == "sell":
            quantity = 0

        total_holdings = decision.current_pricing * quantity
        portfolio[ticker] = total_holdings
        total_portfolio_amount += abs(total_holdings)

    # Normalize portfolio weights
    if total_portfolio_amount > 0:
        for ticker in portfolio:
            portfolio[ticker] /= total_portfolio_amount

        backtester = Backtester(portfolio, benchmark, end_date=end_date)
        results = backtester.run_backtest()
        state["data"]["backtest_results"] = results
        state["data"]["final_position"] = portfolio
        state["data"]["total_portfolio_amount"] = total_portfolio_amount

        # Summarize backtest results
        summary = backtester.summarize_results(results, state["metadata"]["model_name"], state["metadata"]["model_provider"])
        state["data"]["backtest_summary"] = summary.summary

        message = HumanMessage(content=f"Backtest results by comparing the portfolio with {benchmark} index/stock", name="backtester_agent")
        return {"messages": [message], "data": state["data"]}
    message = HumanMessage(content=f"Did not perform the backtesting with the benchmark as the total portfolio amount is 0", name="backtester_agent")
    return {"messages": [message], "data": state["data"]}


# Example usage:
if __name__ == "__main__":
    # Example portfolio (Apple, Microsoft, Amazon)
    portfolio = {"AAPL": 0.4, "MSFT": 0.3, "AMZN": 0.3}

    backtester = Backtester(portfolio, benchmark="VOO", end_date="2023-01-01")
    results = backtester.run_backtest()
    print("Backtest Results:")
