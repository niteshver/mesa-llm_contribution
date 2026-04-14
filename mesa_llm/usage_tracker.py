"""
Usage tracking and budget control system for Mesa-LLM.

This module provides functionality to track LLM API usage (tokens, costs) and enforce budget limits
to prevent excessive spending during simulations.
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class UsageTracker:
    """
    Tracks LLM usage statistics and enforces budget limits.

    Attributes:
        total_tokens (int): Total tokens used across all LLM calls
        prompt_tokens (int): Total prompt tokens used
        completion_tokens (int): Total completion tokens used
        total_cost (float): Estimated total cost in USD
        call_count (int): Number of LLM calls made
        budget_limit (float): Maximum allowed cost in USD (None for unlimited)
        token_limit (int): Maximum allowed tokens (None for unlimited)
    """

    def __init__(self, budget_limit: Optional[float] = None, token_limit: Optional[int] = None):
        """
        Initialize the usage tracker.

        Args:
            budget_limit: Maximum allowed cost in USD
            token_limit: Maximum allowed total tokens
        """
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_cost = 0.0
        self.call_count = 0
        self.budget_limit = budget_limit
        self.token_limit = token_limit

    def update_usage(self, usage: Dict[str, int], model: str) -> None:
        """
        Update usage statistics from an LLM response.

        Args:
            usage: Usage dict from litellm response containing 'prompt_tokens', 'completion_tokens', 'total_tokens'
            model: The model name used for cost calculation

        Raises:
            BudgetExceededError: If budget or token limit is exceeded
        """
        prompt_tokens = usage.get('prompt_tokens', 0)
        completion_tokens = usage.get('completion_tokens', 0)
        total_tokens = usage.get('total_tokens', prompt_tokens + completion_tokens)

        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += total_tokens
        self.call_count += 1

        # Estimate cost (simplified - in production, use proper pricing)
        cost = self._estimate_cost(model, prompt_tokens, completion_tokens)
        self.total_cost += cost

        # Check limits
        if self.budget_limit and self.total_cost > self.budget_limit:
            raise BudgetExceededError(
                f"Budget limit exceeded: ${self.total_cost:.4f} > ${self.budget_limit:.4f}"
            )

        if self.token_limit and self.total_tokens > self.token_limit:
            raise TokenLimitExceededError(
                f"Token limit exceeded: {self.total_tokens} > {self.token_limit}"
            )

        logger.debug(
            f"Usage updated: {total_tokens} tokens (${cost:.4f}), "
            f"Total: {self.total_tokens} tokens (${self.total_cost:.4f})"
        )

    def _estimate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """
        Estimate cost for a single LLM call. This is a simplified implementation.
        In production, use the actual pricing from the provider.

        Args:
            model: Model name
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens

        Returns:
            Estimated cost in USD
        """
        # Simplified pricing - replace with actual pricing logic
        pricing = {
            'gpt-4o': {'prompt': 5e-6, 'completion': 15e-6},  # $5/$15 per million tokens
            'gpt-4o-mini': {'prompt': 0.15e-6, 'completion': 0.6e-6},  # $0.15/$0.6 per million
            'gemini/gemini-2.0-flash': {'prompt': 0.1e-6, 'completion': 0.4e-6},  # Approximate
        }

        # Default pricing if model not found
        default_pricing = {'prompt': 1e-6, 'completion': 4e-6}  # $1/$4 per million tokens

        rates = pricing.get(model.split('/')[-1], default_pricing)

        cost = (prompt_tokens * rates['prompt']) + (completion_tokens * rates['completion'])
        return cost

    def get_usage_summary(self) -> Dict[str, any]:
        """
        Get a summary of current usage.

        Returns:
            Dict containing usage statistics
        """
        return {
            'total_tokens': self.total_tokens,
            'prompt_tokens': self.prompt_tokens,
            'completion_tokens': self.completion_tokens,
            'total_cost': self.total_cost,
            'call_count': self.call_count,
            'budget_limit': self.budget_limit,
            'token_limit': self.token_limit,
            'budget_remaining': self.budget_limit - self.total_cost if self.budget_limit else None,
            'tokens_remaining': self.token_limit - self.total_tokens if self.token_limit else None,
        }

    def reset(self) -> None:
        """Reset all usage statistics."""
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_cost = 0.0
        self.call_count = 0
        logger.info("Usage tracker reset")


class BudgetExceededError(Exception):
    """Raised when budget limit is exceeded."""
    pass


class TokenLimitExceededError(Exception):
    """Raised when token limit is exceeded."""
    pass