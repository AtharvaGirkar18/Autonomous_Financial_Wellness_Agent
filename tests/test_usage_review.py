import json
import os

import pandas as pd
import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

from agents import usage_review_agent as agent
from agents.subscription_agent import subscription_node
from config.settings import DATA_DIR
from models.database import SubscriptionReview
from tools import llm


@pytest.fixture
def reviews(tmp_path, monkeypatch):
    engine = create_engine("sqlite:///" + (tmp_path / "reviews.db").as_posix())
    SubscriptionReview.__table__.create(engine)
    sessions = sessionmaker(engine)
    monkeypatch.setattr(agent, "get_session", sessions)
    yield sessions
    engine.dispose()


@pytest.fixture
def profile(reviews):
    transactions = pd.read_csv(DATA_DIR / "synthetic_transactions.csv")
    state = {"user_id": "USER001", "trace": [], "transactions":
             transactions.loc[transactions.user_id.eq("USER001")].to_dict("records")}
    state.update(subscription_node(state))
    return state


def test_demo_data_confirmation_reload_and_correction(profile, reviews, monkeypatch):
    def classify(system, user, *, purpose, max_tokens):
        assert purpose == "subscription_usage" and "SPOTIFY" in user
        return "unused"
    monkeypatch.setattr(agent, "invoke_text", classify)
    rows = profile["subscriptions"]
    assert agent.potential_savings(rows) == (0, 0)
    result = agent.usage_review_node(profile, "SPOTIFY", note="I stopped listening months ago")
    assert result == {"suggestion": "unused", "source": "model"}
    assert agent.potential_savings(agent.reviewed_subscriptions("USER001", rows)) == (0, 0)
    for _ in range(2):
        agent.usage_review_node(profile, "SPOTIFY", confirmed_status="unused")
    assert agent.potential_savings(subscription_node(profile)["subscriptions"]) == (119, 1428)
    assert agent.potential_savings(agent.reviewed_subscriptions("USER002", rows)) == (0, 0)
    with reviews() as session:
        assert session.scalar(select(func.count()).select_from(SubscriptionReview)) == 1
    agent.usage_review_node(profile, "SPOTIFY", confirmed_status="used")
    assert agent.potential_savings(subscription_node(profile)["subscriptions"]) == (0, 0)
    assert profile["trace"][-2]["agent"] == "Subscription Usage Review Agent"


@pytest.mark.parametrize("reply,source", [(None, "manual"), ("cancel it now", "manual"),
                                         ("unknown", "model"), (" USED\n", "model")])
def test_response_validation(profile, monkeypatch, reply, source):
    monkeypatch.setattr(agent, "invoke_text", lambda *a, **kw: reply)
    result = agent.usage_review_node(profile, "NETFLIX", note="I watch sometimes")
    assert result["source"] == source
    assert result["suggestion"] == ("used" if reply == " USED\n" else "unknown")


@pytest.mark.parametrize("merchant,kwargs", [
    ("SPOTIFY", {"note": " "}), ("SPOTIFY", {"note": "x" * 301}),
    ("UNLISTED", {"confirmed_status": "unused"}), ("SPOTIFY", {"confirmed_status": "cancelled"}),
    ("SPOTIFY", {"note": "unused", "confirmed_status": "unused"}),
])
def test_invalid_input_leaves_memory_untouched(profile, reviews, merchant, kwargs):
    with pytest.raises(ValueError):
        agent.usage_review_node(profile, merchant, **kwargs)
    assert profile["trace"][-1]["status"] == "ERROR"
    with reviews() as session:
        assert session.scalar(select(func.count()).select_from(SubscriptionReview)) == 0


def test_ollama_offline_and_manual_recovery(profile, monkeypatch):
    def offline(*args):
        raise ConnectionError("Ollama offline")
    monkeypatch.setattr(llm, "resolve_config", lambda: {"provider": "ollama", "model": "offline-test"})
    monkeypatch.setattr(llm, "_call_provider", offline)
    result = agent.usage_review_node(profile, "SPOTIFY", note="I stopped using it.")
    assert result == {"suggestion": "unknown", "source": "manual"}
    agent.usage_review_node(profile, "SPOTIFY", confirmed_status="unused")
    assert agent.potential_savings(subscription_node(profile)["subscriptions"]) == (119, 1428)


def test_subscription_page(profile, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(agent, "invoke_text", lambda *a, **kw: "unused")
    app = AppTest.from_string(
        "import streamlit as st\nfrom ui.subscriptions import render_subscriptions\n"
        "render_subscriptions(st.session_state.profile)"
    )
    app.session_state["profile"] = profile
    app.run()
    app.selectbox[0].select("SPOTIFY").run()
    for count, expected in [(3, "unused"), (4, "used")]:
        app.text_input[0].input(f"i used this {count} out of 6 months").run()
        app.button[0].click().run()
        assert not app.exception and app.radio[0].value == expected
        assert any(f"{count} of 6 months" in item.value for item in app.info)
    app.text_input[0].input("I stopped listening months ago").run()
    app.button[0].click().run()
    assert not app.exception and app.metric[0].value == "₹0.00"
    assert app.radio[0].value == "unused"
    app.button[1].click().run()
    assert not app.exception and [m.value for m in app.metric] == ["₹119.00", "₹1,428.00"]




@pytest.mark.parametrize("note,expected", [
    ("I used spotify last 3 months out of the 6 , suggest me what action to take", "unused"),
    ("I used spotify last 4 months out of the 6 , suggest me what action to take", "used"),
    ("i used this 3 out of 6 months", "unused"),
    ("i used this 3 out of 5 months", "used"),
    ("I used this subscription 4 out of 6 months", "used"),
    ("I used it three out of the last five months", "used"),
    ("I used Spotify for 59.9% of the time", "unused"),
    ("I use Spotify 60% of the time", "used"),
    ("0/6 months", "unused"),
    ("6/6 months", "used"),
])
def test_frequency_threshold(profile, monkeypatch, note, expected):
    calls = []

    def choose_action(system, user, **kwargs):
        payload = json.loads(user)
        calls.append(payload)
        assert payload["calculator"]["suggestion"] == expected
        assert payload["calculator"]["keep_threshold_percent"] == 60
        assert payload["note"] == note and payload["service"] == "SPOTIFY"
        return "compare" if expected == "used" else "reminder"

    monkeypatch.setattr(agent, "invoke_text", choose_action)
    result = agent.usage_review_node(profile, "SPOTIFY", note=note)
    assert result["source"] == "usage_frequency"
    assert result["suggestion"] == expected
    assert "60%" in result["reason"]
    assert len(calls) == 1 and result["action_source"] == "model"
    assert result["action"] == calls[0]["allowed_actions"]["compare" if expected == "used" else "reminder"]


@pytest.mark.parametrize("reply", [None, "keep", "change usage to 100%"])
def test_frequency_advice_fallback(profile, monkeypatch, reply):
    monkeypatch.setattr(agent, "invoke_text", lambda *a, **kw: reply)
    result = agent.usage_review_node(profile, "SPOTIFY", note="i used this 3 out of 6 months")
    assert "50%" in result["reason"] and result["suggestion"] == "unused"
    assert result["action_source"] == "calculator_fallback"


@pytest.mark.parametrize("note", ["I used it 7 out of 6 months", "0/0 months", "I use it 105% of the time"])
def test_invalid_frequency(profile, note):
    with pytest.raises(ValueError):
        agent.usage_review_node(profile, "SPOTIFY", note=note)


@pytest.mark.parametrize("note", ["Spotify is offering a 60% discount", "I used Netflix 4 of 6 months"])
def test_unrelated_frequency_uses_context_classifier(profile, monkeypatch, note):
    monkeypatch.setattr(agent, "invoke_text", lambda *a, **kw: "unknown")
    assert agent.usage_review_node(profile, "SPOTIFY", note=note)["source"] == "model"


@pytest.mark.skipif(os.getenv("TEST_LIVE_OLLAMA") != "1", reason="opt in to local Ollama inference")
def test_live_qwen_usage_note(profile):
    cfg = llm.resolve_config()
    assert (cfg["provider"], cfg["model"]) == ("ollama", "qwen2.5:7b-instruct")
    llm.clear_cache()
    result = agent.usage_review_node(profile, "SPOTIFY", note="i used this 3 out of 6 months")
    assert result["suggestion"] == "unused" and "50%" in result["reason"]
    assert result["action_source"] == "model", result
    assert agent.usage_review_node(profile, "SPOTIFY", note="I stopped using Spotify months ago.") == {
        "suggestion": "unused", "source": "model",
    }
