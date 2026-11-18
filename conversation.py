from models import LanguageModel


class Conversation:
    """Keep the messages exchanged with one model."""

    def __init__(self, conversation_id: int, model: LanguageModel) -> None:
        """Start an empty history for the selected model."""
        self.conversation_id = conversation_id
        self.model = model
        self.messages: list[dict[str, str]] = []

    def send_message(self, prompt: str) -> str:
        """Store both messages after the model generates a response."""
        response = self.model.generate_response(prompt)
        self.messages.append({"role": "user", "content": prompt})
        self.messages.append({"role": "assistant", "content": response})
        return response

    def display_history(self) -> None:
        """Print the messages in the order they were sent."""
        for message in self.messages:
            print(f"{message['role']}: {message['content']}")
