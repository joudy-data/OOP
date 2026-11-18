class Tokenizer:
    """Count words as tokens for the simulation."""

    def count_tokens(self, text: str) -> int:
        """Return the number of whitespace-separated words."""
        return len(text.split())
