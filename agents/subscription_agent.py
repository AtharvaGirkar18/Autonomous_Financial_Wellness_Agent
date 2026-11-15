import pandas as pd
from agents.base import run_agent
from analytics.subscriptions import detect_subscriptions


def subscription_node(state: dict) -> dict:
    def _run(s):
        return {"subscriptions": detect_subscriptions(pd.DataFrame(s["transactions"]))}
    return run_agent(state, "Subscription Intelligence Agent", _run, ["detect_subscriptions"])
