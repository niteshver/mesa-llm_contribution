from examples.retry_tool_model import run_demo


def test_retry_tool_model_runs():
    result = run_demo()

    assert result["tool_calls"] == [
        {
            "tool_call_id": "call_retry",
            "role": "tool",
            "name": "fetch_assignment_status",
            "response": "Assignment 1 is submitted.",
        }
    ]
    assert result["action_memory"]["tool_retry_count"] == 1
    assert result["action_memory"]["tool_retry_history"] == [
        [
            {
                "name": "fetch_assignment_status",
                "response": "Error: Unknown assignment_id: 999",
            }
        ]
    ]
