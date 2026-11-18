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
