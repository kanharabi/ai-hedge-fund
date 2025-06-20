import yfinance as yf
from langchain_core.messages import HumanMessage
from src.graph.state import AgentState, show_agent_reasoning

def factor_exposure_agent(state: AgentState) -> AgentState:
    """Analyzes factor exposures for each ticker based on financial metrics."""
    factor_scores = {}
    tickers = state["data"]["tickers"]

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            scores = {
                "value": round(1 / info.get("trailingPE", 20), 3),  # inverse PE
                "growth": round(info.get("earningsGrowth", 0), 3),
                "momentum": round(info.get("52WeekChange", 0), 3),
                "volatility": round(info.get("beta", 1), 3),
                "quality": round(info.get("returnOnEquity", 0), 3),
                "size": round(info.get("marketCap", 0) / 1e12, 3)  # normalize to trillions
            }

            factor_scores[ticker] = scores

        except Exception as e:
            factor_scores[ticker] = {"error": str(e)}

    # Store in state
    state["data"]["factor_exposure"] = factor_scores

    # Optional: show reasoning
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(factor_scores, "Factor Exposure Agent")

    # Add message
    message = HumanMessage(
        content=f"Factor exposure analysis complete: {factor_scores}",
        name="factor_exposure_agent"
    )

    return {
        "messages": state["messages"] + [message],
        "data": state["data"]
    }
