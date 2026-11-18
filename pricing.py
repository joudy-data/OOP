class PricingPlan:
    """Store a pricing plan that multiple models can share."""

    def __init__(self, plan_name: str, price_per_1000_tokens: float) -> None:
        """Set the name and rate for the plan."""
        self.plan_name = plan_name
        self.price_per_1000_tokens = price_per_1000_tokens

    def calculate_cost(self, tokens: int) -> float:
        """Return the simulated charge for a token count."""
        return tokens / 1000 * self.price_per_1000_tokens
