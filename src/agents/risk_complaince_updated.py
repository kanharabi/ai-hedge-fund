import pandas as pd
import matplotlib.pyplot as plt
import openai
from openai import OpenAI, AzureOpenAI
import matplotlib
from langchain_core.messages import HumanMessage
from src.llm.models import get_model, get_model_info
import os
matplotlib.use('Agg') 

from src.graph.state import AgentState, show_agent_reasoning

class RiskComplianceAgent:
    def __init__(self, volume_threshold=0.01, price_jump_threshold=0.01,
                 leverage_limit=0.01, stop_loss_threshold=0.10, drawdown_limit=0.20):
        self.volume_threshold = volume_threshold
        self.price_jump_threshold = price_jump_threshold
        self.leverage_limit = leverage_limit
        self.stop_loss_threshold = stop_loss_threshold
        self.drawdown_limit = drawdown_limit
        self.openai_api_key = os.getenv('AZURE_API_KEY')
        # if openai_api_key:
        #     openai.api_key = openai_api_key

    def analyze(self, df):
        alerts = []

        df["PriceChange"] = df["Price"].pct_change()
        df["VolumeZScore"] = (df["Volume"] - df["Volume"].rolling(20).mean()) / df["Volume"].rolling(20).std()
        df["PeakValue"] = df["PortfolioValue"].cummax()
        df["Drawdown"] = (df["PortfolioValue"] - df["PeakValue"]) / df["PeakValue"]

        for i in range(1, len(df)):
            date = df.index[i]
            row = df.iloc[i]

            if abs(row["PriceChange"]) > self.price_jump_threshold and abs(row["VolumeZScore"]) > self.volume_threshold:
                alerts.append((date, "Insider Trading Pattern Detected"))

            if row["LeverageRatio"] > self.leverage_limit:
                alerts.append((date, "Excessive Leverage Detected"))

            if row["PortfolioValue"] < row["PeakValue"] * (1 - self.stop_loss_threshold):
                alerts.append((date, "Stop-Loss Triggered"))

            if row["Drawdown"] < -self.drawdown_limit:
                alerts.append((date, "Drawdown Limit Breached"))

        return pd.DataFrame(alerts, columns=["Date", "Alert"]).set_index("Date")

    def explain_alerts(self, alerts_df,model_name,model_provider):
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not provided.")

        explanations = []
        for date, row in alerts_df.iterrows():
            alert_type = row["Alert"]
            prompt = f"""
            You are a financial compliance assistant, that monitor for insider trading patterns, excessive leverage, or regulatory violations;. Explain the following alert in simple terms:
            Alert Type: {alert_type}
            Context: This alert was triggered during portfolio monitoring.
            """
            try:
                
                # openai.api_key = "DRVpWPg8EJUvpkFE5Fp3j0AQ9fjo1dsrqWFf9QGklJhTtdf1mc5tJQQJ99BFACHYHv6XJ3w3AAAAACOGQSKJ"
                # response = openai.ChatCompletion.create(model=model_name,messages=[{"role": "user", "content": prompt}])
                endpoint = "https://daska-mc38oyhh-eastus2.cognitiveservices.azure.com/"
                model_name = "o4-mini"
                deployment = "o4-mini"
                subscription_key = "DRVpWPg8EJUvpkFE5Fp3j0AQ9fjo1dsrqWFf9QGklJhTtdf1mc5tJQQJ99BFACHYHv6XJ3w3AAAAACOGQSKJ"
                api_version = "2024-12-01-preview"
                client = AzureOpenAI(api_version=api_version, azure_endpoint=endpoint, api_key=subscription_key,)
                # client = OpenAI(
                #             # This is the default and can be omitted
                #             # api_key=self.openai_api_key,
                #             api_key="DRVpWPg8EJUvpkFE5Fp3j0AQ9fjo1dsrqWFf9QGklJhTtdf1mc5tJQQJ99BFACHYHv6XJ3w3AAAAACOGQSKJ"
                #             # api_key = "1f5dcb15e3cb463aa3be65126e048e0c"
                #         )
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}]
                )

                # response = client.responses.create(
                #     model=model_name,
                #     instructions = prompt,
                #     messages=[{"role": "user", "content": prompt}]
                # )
                # model_info = get_model_info(model_name)
                # llm = get_model(model_name, model_provider)
                # result = llm.invoke(prompt)
                # explanation = response['choices'][0]['message']['content']
                explanation = response.choices[0].message.content
                # explanation = response.output_text
                # explanation = result.content
                print("Explanation : ", explanation)
            except Exception as e:
                explanation = f"Error generating explanation: {e}"

            explanations.append((date, alert_type, explanation))

        return pd.DataFrame(explanations, columns=["Date", "Alert", "Explanation"]).set_index("Date")

    def plot_alerts(self, df, alerts_df):
        plt.figure(figsize=(12, 6))
        plt.plot(df["PortfolioValue"], label="Portfolio Value", color="blue")
        for alert_date in alerts_df.index:
            plt.axvline(x=alert_date, color="red", linestyle="--", alpha=0.3)
        plt.title("Portfolio Value with Risk & Compliance Alerts")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()
        plt.show()

def watchdog_agent(state:AgentState) -> AgentState:
    messages = state['messages']
    risk_compliance_agent = RiskComplianceAgent()
    # ToDo - File to extract individual Stock info based on ticker name
    df = pd.read_csv(r"C:\\Users\\dimv\Downloads\\portfolio_data.csv", parse_dates=["Date"])
    df = df.head(2)
    df.set_index("Date", inplace=True)

    model_name=state["metadata"]["model_name"]
    model_provider=state["metadata"]["model_provider"]
    
    alerts_df = risk_compliance_agent.analyze(df)
    # Generate explanations (optional)
    explained_alerts_df = risk_compliance_agent.explain_alerts(alerts_df,model_name,model_provider)
    # Visualize
    # risk_compliance_agent.plot_alerts(df, alerts_df)
    state['data']['watchdog_df'] = explained_alerts_df
    # Add message
    message = HumanMessage(
        content=f"Watchdog Analysis for conducting risk surveillance to detect insider trading, excessive leverage, and regulatory breaches: {explained_alerts_df}",
        name="watchdog_agent"
    )
    return{
        "messages":messages + [message],
        "data": state['data']
    }
