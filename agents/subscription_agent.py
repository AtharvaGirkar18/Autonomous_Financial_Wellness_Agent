import pandas as pd
from agents.base import run_agent
from agents.usage_review_agent import reviewed_subscriptions
from analytics.subscriptions import detect_subscriptions


def subscription_node(state: dict) -> dict:
    def _run(s):
        detected = detect_subscriptions(pd.DataFrame(s["transactions"]))
        return {"subscriptions": reviewed_subscriptions(s["user_id"], detected)}
    return run_agent(state, "Subscription Intelligence Agent", _run, ["detect_subscriptions"])
