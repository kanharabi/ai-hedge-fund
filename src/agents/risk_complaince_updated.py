import pandas as pd
import matplotlib.pyplot as plt
import openai
import matplotlib
matplotlib.use('Agg') 

class RiskComplianceAgent:
    def __init__(self, volume_threshold=2.5, price_jump_threshold=0.05,
                 leverage_limit=2.0, stop_loss_threshold=0.10, drawdown_limit=0.20,
                 openai_api_key=None):
        self.volume_threshold = volume_threshold
        self.price_jump_threshold = price_jump_threshold
        self.leverage_limit = leverage_limit
        self.stop_loss_threshold = stop_loss_threshold
        self.drawdown_limit = drawdown_limit
        self.openai_api_key = openai_api_key
        if openai_api_key:
            openai.api_key = openai_api_key

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

    def explain_alerts(self, alerts_df):
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not provided.")

        explanations = []
        for date, row in alerts_df.iterrows():
            alert_type = row["Alert"]
            prompt = f"""
            You are a financial compliance assistant. Explain the following alert in simple terms:
            Alert Type: {alert_type}
            Context: This alert was triggered during portfolio monitoring.
            """
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}]
                )
                explanation = response['choices'][0]['message']['content']
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