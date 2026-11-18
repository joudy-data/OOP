from models import LanguageModel


class ModelManager:
    """Manage a collection of models indexed by name."""

    def __init__(self) -> None:
        """Start with an empty registry."""
        self.models: dict[str, LanguageModel] = {}

    def add_model(self, model: LanguageModel) -> None:
        """Register a model under its name."""
        self.models[model.model_name] = model

    def remove_model(self, model_name: str) -> LanguageModel | None:
        """Remove and return a registered model if it exists."""
        return self.models.pop(model_name, None)

    def get_model(self, model_name: str) -> LanguageModel | None:
        """Look up a model by name."""
        return self.models.get(model_name)

    def list_models(self) -> None:
        """Print the registered model names."""
        for model_name in self.models:
            print(model_name)

    def select_model(self, provider: str, required_tokens: int) -> LanguageModel | None:
        """Return the first active model that meets the request."""
        for model in self.models.values():
            if (
                model.provider == provider
                and model.max_tokens >= required_tokens
                and model.is_active
            ):
                return model
        return None
