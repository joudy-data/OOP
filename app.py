from contextlib import redirect_stdout
from io import StringIO

import streamlit as st

from conversation import Conversation
from exceptions import InvalidPromptError
from main import build_manager
from manager import ModelManager
from models import GPTModel, LanguageModel
from pricing import PricingPlan
from tokenizer import Tokenizer


st.set_page_config(page_title="Model Playground", page_icon="💬", layout="wide")
st.markdown(
    """
    <style>
    .playground-header {
        border-radius: 18px;
        padding: 26px 32px;
        margin-bottom: 18px;
        color: #f8fafc;
        background: linear-gradient(115deg, #162344, #283569 68%, #3c427d);
    }
    .playground-header h1 {color: #ffffff; padding: 0; font-size: 2.2rem;}
    .playground-header p {color: #dbe5fa; margin: 6px 0 0;}
    [data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.35);
        border-radius: 12px;
        padding: 12px 16px;
    }
    </style>
    <div class="playground-header">
        <h1>Model Playground</h1>
        <p>Manage GPT and Llama, test prompts, and inspect usage in a simulated workspace.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

state = st.session_state
if "manager" not in state:
    state.manager = build_manager()
if "model_catalog" not in state:
    state.model_catalog = dict(state.manager.models)
if "conversations" not in state:
    state.conversations = {}
if "usage" not in state:
    state.usage = {}
if "chat_errors" not in state:
    state.chat_errors = {}

manager = state.manager
catalog = state.model_catalog


def status_of(model):
    if manager.get_model(model.model_name) is None:
        return "Removed"
    return "Active" if model.is_active else "Inactive"


def usage_of(name):
    return state.usage.get(name, {"tokens": 0, "cost": 0.0})


def validate_input(model, prompt):
    model.validate_prompt(prompt)
    tokens = model.tokenizer.count_tokens(prompt)
    if tokens > model.max_tokens:
        raise InvalidPromptError(f"This model accepts up to {model.max_tokens:,} input tokens.")
    return tokens


def capture_output(function, *args):
    output = StringIO()
    with redirect_stdout(output):
        result = function(*args)
    return result, output.getvalue().strip()


def dashboard():
    st.subheader("Usage overview")
    counts = [len(chat.messages) // 2 for chat in state.conversations.values()]
    usage = list(state.usage.values())
    summary = st.columns(4)
    summary[0].metric("Active models", f"{sum(m.is_active for m in manager.models.values())} / {len(manager.models)}")
    summary[1].metric("Messages sent", sum(counts))
    summary[2].metric("Input tokens", f"{sum(row['tokens'] for row in usage):,}")
    summary[3].metric("Estimated cost", f"${sum(row['cost'] for row in usage):.5f}")

    rows = []
    for name, model in catalog.items():
        chat = state.conversations.get(name)
        rows.append({
