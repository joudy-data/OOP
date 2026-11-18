from conversation import Conversation
from exceptions import InvalidPromptError
from manager import ModelManager
from models import GPTModel, LanguageModel, LlamaModel
from pricing import PricingPlan


def build_manager() -> ModelManager:
    """Create the two simulated models and register them."""
    pricing = PricingPlan("Standard", 0.02)
    gpt = GPTModel("GPT-Model", "OpenAI", 4096, 0.7, "demo-key", pricing)
    llama = LlamaModel("Llama-Model", "Local", 8192, 0.5, "/models/llama", "4-bit", pricing)

    manager = ModelManager()
    manager.add_model(gpt)
    manager.add_model(llama)
    return manager


def main() -> None:
    """Run the local demonstration and check the project behavior."""
    model_count = LanguageModel.get_total_models()
    manager = build_manager()
    assert LanguageModel.get_total_models() == model_count + 2
    manager.list_models()

    gpt = manager.get_model("GPT-Model")
    llama = manager.get_model("Llama-Model")
    assert isinstance(gpt, GPTModel)
    assert isinstance(llama, LlamaModel)
    assert gpt.pricing_plan is llama.pricing_plan
    assert gpt.tokenizer is not llama.tokenizer
    gpt.display_info()
    print(str(gpt), repr(gpt), len(gpt), gpt == llama, sep="\n")

    for model in (gpt, llama):
        print(model.generate_response("Explain OOP"))

    gpt.temperature = 1.2
    try:
        gpt.temperature = 4
    except ValueError as error:
        print(error)
    assert gpt.temperature == 1.2

    assert manager.select_model("OpenAI", 4096) is gpt
    assert manager.select_model("OpenAI", 4097) is None
    gpt.is_active = False
    assert manager.select_model("OpenAI", 1000) is None
    gpt.is_active = True
    assert manager.remove_model("Llama-Model") is llama
    assert manager.get_model("Llama-Model") is None
    manager.add_model(llama)

    conversation = Conversation(1, gpt)
    for prompt in ("Explain object-oriented programming", "", "   ", 123, "What is inheritance?"):
        try:
            conversation.send_message(prompt)
            gpt.calculate_request_cost(prompt)
        except InvalidPromptError as error:
            print(error)
    assert len(conversation.messages) == 4
    conversation.display_history()


if __name__ == "__main__":
    main()
