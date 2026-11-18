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
        model_usage = usage_of(name)
        row = {
            "Model": name,
            "Provider": model.provider,
            "Status": status_of(model),
            "Temperature": f"{model.temperature:.1f}",
            "Token limit": f"{len(model):,}",
            "Messages": len(chat.messages) // 2 if chat else 0,
            "Input tokens": model_usage["tokens"],
            "Estimated cost": f"${model_usage['cost']:.5f}",
        }
        rows.append(row)
    st.table(rows)

    if any(row["Input tokens"] for row in rows):
        st.subheader("Input tokens by model")
        st.bar_chart(rows, x="Model", y="Input tokens", color="#6966cf")
    else:
        st.info("Send a message in Chat to start tracking usage.")


def chat():
    st.subheader("Chat")
    if not manager.models:
        st.info("Restore a model in Model management to start a chat.")
        return

    name = st.selectbox("Chat model", list(manager.models), key="chat_model")
    model = manager.get_model(name)
    if name not in state.conversations:
        state.conversations[name] = Conversation(list(catalog).index(name) + 1, model)
        state.usage[name] = {"tokens": 0, "cost": 0.0}

    conversation = state.conversations[name]
    usage = state.usage[name]
    st.caption(f"{model.provider} · {status_of(model)} · {model.max_tokens:,} token limit · Temperature {model.temperature:.1f}")
    summary = st.columns(3)
    summary[0].metric("Messages sent", len(conversation.messages) // 2)
    summary[1].metric("Input tokens", f"{usage['tokens']:,}")
    summary[2].metric("Estimated cost", f"${usage['cost']:.5f}")

    if st.button("Clear this conversation"):
        conversation.messages.clear()
        usage.update(tokens=0, cost=0.0)
        state.chat_errors.pop(name, None)
        st.rerun()

    for message in conversation.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if not model.is_active:
        st.warning(f"{name} is inactive. Enable it in Model management to send messages.")

    error = state.chat_errors.pop(name, None)
    if error:
        st.error(error)

    prompt = st.chat_input(f"Message {name}", disabled=not model.is_active, key="chat_prompt")
    if prompt is not None and model.is_active:
        try:
            tokens = validate_input(model, prompt)
            conversation.send_message(prompt)
            cost = model.calculate_request_cost(prompt)
            usage["tokens"] += tokens
            usage["cost"] += cost
        except InvalidPromptError as error:
            state.chat_errors[name] = f"InvalidPromptError: {error}"
        st.rerun()


def model_management():
    st.subheader("Model management")
    st.caption("Settings apply immediately to Chat, the Dashboard, and model selection.")
    columns = st.columns(len(catalog))
    for column, (name, model) in zip(columns, catalog.items()):
        with column, st.container(border=True):
            st.markdown(f"### {name}")
            st.caption(f"{model.provider} · {model.max_tokens:,} token limit")
            status = status_of(model)
            st.write(f"**Status:** {status}")
            if status == "Removed":
                if st.button("Restore model", key=f"restore_{name}"):
                    manager.add_model(model)
                    st.rerun()
                continue
            previous_active = model.is_active
            model.is_active = st.toggle("Model active", value=previous_active, key=f"active_{name}")
            if model.is_active != previous_active:
                st.rerun()
            model.temperature = st.slider(
                "Temperature", 0.0, 2.0, float(model.temperature), 0.1,
                key=f"temperature_{name}",
                help="Validated by the model setter. Simulated replies use a fixed format.",
            )
            st.caption(f"Plan: {model.pricing_plan.plan_name} · ${model.pricing_plan.price_per_1000_tokens:.2f} per 1,000 tokens")
            if st.button("Remove from registry", key=f"remove_{name}"):
                manager.remove_model(name)
                st.rerun()

    st.subheader("Automatic model selection")
    with st.form("model_selection"):
        left, right = st.columns(2)
        provider = left.selectbox("Provider", sorted({model.provider for model in catalog.values()}))
        required = right.number_input("Required input tokens", min_value=1, value=100, step=1)
        submitted = st.form_submit_button("Find eligible model")
    if submitted:
        selected = manager.select_model(provider, int(required))
        if selected is None:
            st.warning(f"No active {provider} model has capacity for {required:,} tokens.")
        else:
            st.success(f"{selected.model_name} · {selected.max_tokens:,} token capacity")


STAGES = {
    1: ("Base class", "Inspect the four attributes defined by LanguageModel.", "model.display_info()"),
    2: ("Instance methods", "Generate a simulated reply and display model settings.", "model.generate_response(sample)"),
    3: ("Class attribute", "Read the shared count of model objects created.", "LanguageModel.total_models"),
    4: ("Class method", "Read the same counter through a method that receives the class.", "LanguageModel.get_total_models()"),
    5: ("Static method", "Check a temperature using only its numeric value.", "LanguageModel.validate_temperature(value)"),
    6: ("Encapsulation and properties", "Try the setter on a temporary model and read the stored value.", "demo.temperature = value"),
    7: ("Inheritance", "Inspect a subclass and the settings initialized through super().", "isinstance(model, LanguageModel)"),
    8: ("Method overriding", "Run the selected subclass's implementation.", "model.generate_response(sample)"),
    9: ("Polymorphism", "Send the same sample to both models through one method call.", "for model in models:\n    model.generate_response(sample)"),
    10: ("Abstraction", "Try creating LanguageModel directly and see why Python rejects it.", "LanguageModel(\"Demo\", \"Local\", 100, 0.7)"),
    11: ("Tokenizer", "Split the sample into words and count its tokens.", "Tokenizer().count_tokens(sample)"),
    12: ("Composition", "Check whether both models share the same internally created tokenizer.", "gpt.tokenizer is llama.tokenizer"),
    13: ("Pricing plan", "Calculate a cost using PricingPlan and the sample's token count.", "plan.calculate_cost(tokens)"),
    14: ("Aggregation", "Inspect the shared pricing object and calculate one request's cost.", "gpt.pricing_plan is llama.pricing_plan"),
    15: ("Magic methods", "Inspect the display text, debug representation, length, and equality.", "str(model), repr(model), len(model), gpt == llama"),
    16: ("Multiple inheritance", "Capture the log that GPT inherits from LoggingMixin.", "gpt.generate_response(sample)"),
    17: ("Custom exceptions", "Pass an actual invalid value through the chat's validation function.", "validate_input(model, example)"),
    18: ("Conversation", "Send a sample in a temporary conversation and inspect the two stored messages.", "conversation.send_message(sample)"),
    19: ("Model manager", "Add, list, find, and remove a model in a temporary registry.", "manager.add_model(model)"),
    20: ("Select the best model", "Try to select the chosen model using its provider and token capacity.", "manager.select_model(model.provider, required_tokens)"),
}


def run_stage(number, model, value=0.7, required=100, example=""):
    gpt, llama = catalog["GPT-Model"], catalog["Llama-Model"]
    sample = "Explain Python classes"
    if number == 1:
        return capture_output(model.display_info)[1]
    if number == 2:
        return capture_output(model.display_info)[1] + "\n" + capture_output(model.generate_response, sample)[0]
    if number == 3:
        return f"LanguageModel.total_models = {LanguageModel.total_models}"
    if number == 4:
        return f"LanguageModel.get_total_models() = {LanguageModel.get_total_models()}"
    if number == 5:
        return f"{value:.1f} is valid: {LanguageModel.validate_temperature(value)}"
    if number == 6:
        demo = GPTModel("Temperature-Demo", "Demo", 100, 0.7, "demo")
        try:
            demo.temperature = value
        except ValueError as error:
            st.error(f"{type(error).__name__}: {error}")
        return f"Attempted: {value:.1f}\nStored: {demo.temperature:.1f}"
    if number == 7:
        return f"Class: {type(model).__name__}\nIs a LanguageModel: {isinstance(model, LanguageModel)}\n" + capture_output(model.display_info)[1]
    if number == 8:
        return capture_output(model.generate_response, sample)[0]
    if number == 9:
        return "\n\n".join(capture_output(item.generate_response, sample)[0] for item in (gpt, llama))
    if number == 10:
        try:
            LanguageModel("Demo", "Local", 100, 0.7)
        except TypeError as error:
            st.error(f"TypeError: {error}")
        return "Concrete subclasses: GPTModel, LlamaModel"
    if number == 11:
        return f"Words: {sample.split()}\nTokens: {Tokenizer().count_tokens(sample)}"
    if number == 12:
        return f"Share a tokenizer: {gpt.tokenizer is llama.tokenizer}\nGPT token count: {gpt.tokenizer.count_tokens(sample)}\nLlama token count: {llama.tokenizer.count_tokens(sample)}"
    if number == 13:
        plan = PricingPlan("Standard", 0.02)
        tokens = Tokenizer().count_tokens(sample)
        return f"{tokens} input tokens · ${plan.calculate_cost(tokens):.5f}"
    if number == 14:
        return f"Share a pricing plan: {gpt.pricing_plan is llama.pricing_plan}\n" + capture_output(model.calculate_request_cost, sample)[1]
    if number == 15:
        return f"str: {model}\nrepr: {model!r}\nlen: {len(model)}\nGPT == Llama: {gpt == llama}"
    if number == 16:
        reply, log = capture_output(gpt.generate_response, sample)
        return f"{log}\n{reply}"
    if number == 17:
        validate_input(model, example)
        st.success("Input accepted.")
        return ""
    if number == 18:
        conversation = Conversation(0, model)
        capture_output(conversation.send_message, sample)
        return capture_output(conversation.display_history)[1]
    if number == 19:
        demo = ModelManager()
        demo.add_model(model)
        _, names = capture_output(demo.list_models)
        found = demo.get_model(model.model_name) is model
        demo.remove_model(model.model_name)
        return f"Listed: {names}\nFound before removal: {found}\nAfter removal: {demo.get_model(model.model_name)}"
    selected = manager.select_model(model.provider, required)
    return f"Selected: {selected}" if selected is not None else "Selected: None (no active model meets the request)"


def stage_lab():
    st.subheader("OOP stage lab")
    st.caption("Choose a stage and run it against the project classes. Chat history and usage are separate from these examples.")
    number = st.selectbox("Stage", list(STAGES), format_func=lambda n: f"{n:02d} · {STAGES[n][0]}")
    name = st.selectbox("Demo model", list(catalog))
    model = catalog[name]
    _, description, call = STAGES[number]
    st.write(description)
    st.code(call, language="python")
    value, required, example = 0.7, 100, ""
    if number in (5, 6):
        value = st.number_input("Temperature to test", value=4.0, step=0.1)
    elif number == 17:
        cases = {
            "Empty message": "",
            "Spaces only": "   ",
            "Number (int)": 88,
            "Over the token limit": "test " * (model.max_tokens + 1),
            "Valid text": "Explain Python classes",
        }
        case = st.selectbox("Test case", list(cases))
        example = cases[case]
    elif number == 20:
        required = int(st.number_input("Required input tokens", min_value=1, value=100, step=1))
    if st.button("Run stage", type="primary"):
        try:
            result = run_stage(number, model, value, required, example)
            if result:
                st.code(result, language="text")
        except InvalidPromptError as error:
            st.error(f"InvalidPromptError: {error}")


page = st.radio(
    "Workspace", ["Dashboard", "Chat", "Model management", "OOP stage lab"],
    horizontal=True, label_visibility="collapsed",
)
pages = {
    "Dashboard": dashboard,
    "Chat": chat,
    "Model management": model_management,
    "OOP stage lab": stage_lab,
}
pages[page]()
