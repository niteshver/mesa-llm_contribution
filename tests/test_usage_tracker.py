from mesa_llm.usage_tracker import UsageTracker, BudgetExceededError, TokenLimitExceededError

# Simulate a response from an LLM API (like litellm)
class DummyResponse:
    def __init__(self, prompt_tokens, completion_tokens):
        self.usage = {
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': prompt_tokens + completion_tokens,
        }


def test_usage_tracker():
    # Set a high enough budget for first two tests
    tracker = UsageTracker(budget_limit=10, token_limit=100)
    model = 'gpt-4o'
    # Should be fine
    resp1 = DummyResponse(50, 50)  # 100 tokens
    try:
        tracker.update_usage(resp1.usage, model)
        print("First update: PASS", tracker.get_usage_summary())
    except Exception as e:
        print("First update: FAIL", e)

    # Should hit token limit
    resp2 = DummyResponse(60, 40)  # 100 tokens, total now 200
    try:
        tracker.update_usage(resp2.usage, model)
        print("Second update: FAIL (should have raised TokenLimitExceededError)")
    except TokenLimitExceededError as e:
        print("Second update: PASS (token limit)", e)
    except Exception as e:
        print("Second update: FAIL (wrong error)", e)

    # Should hit budget limit
    tracker.reset()
    tracker.budget_limit = 0.000001  # Very low budget
    resp3 = DummyResponse(10000, 10000)
    try:
        tracker.update_usage(resp3.usage, model)
        print("Third update: FAIL (should have raised BudgetExceededError)")
    except BudgetExceededError as e:
        print("Third update: PASS (budget limit)", e)
    except Exception as e:
        print("Third update: FAIL (wrong error)", e)

if __name__ == "__main__":
    test_usage_tracker()
