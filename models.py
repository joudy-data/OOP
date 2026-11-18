from abc import ABC, abstractmethod

from exceptions import InvalidPromptError
from pricing import PricingPlan
from tokenizer import Tokenizer


class LanguageModel(ABC):
    """Define shared settings and behavior for simulated models."""

    total_models = 0

    def __init__(
        self, model_name: str, provider: str, max_tokens: int,
        temperature: float, pricing_plan: PricingPlan | None = None,
    ) -> None:
        """Save the settings and create the model's own tokenizer."""
        self.model_name = model_name
        self.provider = provider
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.tokenizer = Tokenizer()
        self.pricing_plan = pricing_plan
        self.__is_active = True
        LanguageModel.total_models += 1

    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        """Return a simulated response using the subclass's format."""

    def display_info(self) -> None:
        """Print the model's main settings."""
        print("Model Name:", self.model_name)
        print("Provider:", self.provider)
        print("Maximum Tokens:", self.max_tokens)
        print("Temperature:", self.temperature)

    @classmethod
    def get_total_models(cls) -> int:
        """Return the shared count of models created."""
        return cls.total_models

    @staticmethod
    def validate_temperature(temperature: float) -> bool:
        """Check whether the setting is between zero and two."""
        return 0 <= temperature <= 2

    @property
    def temperature(self) -> float:
        """Return the stored temperature."""
        return self.__temperature

    @temperature.setter
    def temperature(self, value: float) -> None:
        """Validate a new temperature before saving it."""
        if not self.validate_temperature(value):
            raise ValueError("Temperature must be between 0 and 2.")
        self.__temperature = value

    @property
    def is_active(self) -> bool:
        """Return whether the model is available for selection."""
        return self.__is_active

    @is_active.setter
    def is_active(self, value: bool) -> None:
        """Set the model's availability."""
        if not isinstance(value, bool):
            raise TypeError("Model status must be True or False.")
        self.__is_active = value

    def validate_prompt(self, prompt: str) -> None:
        """Reject non-text and blank prompts."""
        if not isinstance(prompt, str):
            raise InvalidPromptError("Prompt must be a string.")
        if not prompt.strip():
            raise InvalidPromptError("Prompt cannot be empty.")

    def format_response(self, prefix: str, prompt: str) -> str:
        """Combine simulated response text and the prompt's token count."""
        tokens = self.tokenizer.count_tokens(prompt)
        return f"{prefix}{prompt}\nInput tokens: {tokens}"

    def calculate_request_cost(self, prompt: str) -> float:
        """Print and return the simulated cost of the input prompt."""
        self.validate_prompt(prompt)
        if self.pricing_plan is None:
            raise ValueError("Assign a pricing plan before calculating the cost.")
        tokens = self.tokenizer.count_tokens(prompt)
        cost = self.pricing_plan.calculate_cost(tokens)
        print(f"Request cost: {cost:.5f}")
        return cost

    def __str__(self) -> str:
        """Return a readable model label."""
        return f"{self.model_name} by {self.provider}"

    def __repr__(self) -> str:
        """Return the model class, name, and provider for debugging."""
        return f"{type(self).__name__}(model_name={self.model_name!r}, provider={self.provider!r})"

    def __len__(self) -> int:
        """Return the model's token capacity."""
        return self.max_tokens

    def __eq__(self, other: object) -> bool:
        """Compare models by name and provider."""
        if not isinstance(other, LanguageModel):
            return NotImplemented
        return self.model_name == other.model_name and self.provider == other.provider


class LoggingMixin:
    """Add simple console logging to a model."""

    def log(self, message: str) -> None:
        """Print a message with a log label."""
        print(f"[LOG] {message}")


class GPTModel(LanguageModel, LoggingMixin):
    """Simulate a response from GPT."""

    def __init__(
        self, model_name: str, provider: str, max_tokens: int, temperature: float,
        api_key: str, pricing_plan: PricingPlan | None = None,
    ) -> None:
        """Set shared settings and retain the demo key."""
        super().__init__(model_name, provider, max_tokens, temperature, pricing_plan)
        self.api_key = api_key

    def generate_response(self, prompt: str) -> str:
        """Validate the prompt, log the request, and return simulated text."""
        self.validate_prompt(prompt)
        self.log(f"Sending request to {self.model_name}")
        return self.format_response("GPT API response to: ", prompt)


class LlamaModel(LanguageModel):
    """Simulate a response from a local Llama model."""

    def __init__(
        self, model_name: str, provider: str, max_tokens: int, temperature: float,
        model_path: str, quantization: str, pricing_plan: PricingPlan | None = None,
    ) -> None:
        """Set shared settings and local model options."""
        super().__init__(model_name, provider, max_tokens, temperature, pricing_plan)
        self.model_path = model_path
        self.quantization = quantization

    def generate_response(self, prompt: str) -> str:
        """Validate the prompt and return a simulated local response."""
        self.validate_prompt(prompt)
        return self.format_response("Local Llama response to: ", prompt)
