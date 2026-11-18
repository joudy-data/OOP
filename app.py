import streamlit as st

from conversation import Conversation
from exceptions import InvalidPromptError
from main import build_manager


st.set_page_config(page_title="Model Playground", page_icon="💬")

if "manager" not in st.session_state:
    st.session_state.manager = build_manager()
    st.session_state.conversations = {}
    st.session_state.usage = {}
    st.session_state.chat_errors = {}

manager = st.session_state.manager

with st.sidebar:
    st.header("Model settings")
    model_name = st.selectbox("Model", list(manager.models))
    model = manager.get_model(model_name)
    st.caption(f"{model.provider} · {model.max_tokens:,} token limit")
    model.temperature = st.slider(
        "Temperature", 0.0, 2.0, float(model.temperature), 0.1,
        key=f"temperature_{model_name}",
        help="The demo stores this setting. Simulated replies use a fixed format.",
    )
    model.is_active = st.toggle("Model active", value=model.is_active, key=f"active_{model_name}")
    if st.button("Clear this conversation"):
        st.session_state.conversations.pop(model_name, None)
        st.session_state.usage.pop(model_name, None)
        st.session_state.chat_errors.pop(model_name, None)

if model_name not in st.session_state.conversations:
    st.session_state.conversations[model_name] = Conversation(
        list(manager.models).index(model_name) + 1, model
    )
    st.session_state.usage[model_name] = {"tokens": 0, "cost": 0.0}

conversation = st.session_state.conversations[model_name]
usage = st.session_state.usage[model_name]

st.title("Model Playground")
st.caption("Simulated chat · One word counts as one input token")

turns, tokens, cost = st.columns(3)
turns.metric("Messages sent", len(conversation.messages) // 2)
tokens.metric("Input tokens", f"{usage['tokens']:,}")
cost.metric("Estimated cost", f"${usage['cost']:.5f}")

for message in conversation.messages:
    with st.chat_message(message["role"]):
        st.text(message["content"])

if not model.is_active:
    st.info("Turn on Model active in the sidebar to send a message.")

error = st.session_state.chat_errors.pop(model_name, None)
if error:
    st.error(error)

prompt = st.chat_input(f"Message {model_name}", disabled=not model.is_active)
if prompt is not None:
    try:
        model.validate_prompt(prompt)
        token_count = model.tokenizer.count_tokens(prompt)
        if token_count > model.max_tokens:
            raise InvalidPromptError(f"This model accepts up to {model.max_tokens:,} input tokens.")
        conversation.send_message(prompt)
        usage["tokens"] += token_count
        usage["cost"] += model.calculate_request_cost(prompt)
    except InvalidPromptError as error:
        st.session_state.chat_errors[model_name] = str(error)
    st.rerun()
