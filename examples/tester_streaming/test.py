
from litellm import completion

MODEL = "ollama/llama3-groq-tool-use:8b"
API_BASE = "http://localhost:11434"


def stream_response(prompt: str) -> str:
    """Stream the model's response and print text as it arrives."""

    response = completion(
        model=MODEL,
        api_base=API_BASE,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful AI assistant.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        stream=True,
    )

    full_response = []

    for chunk in response:
        choices = getattr(chunk, "choices", None) or []

        if not choices:
            continue

        delta = getattr(choices[0], "delta", None)
        content = getattr(delta, "content", None) if delta else None

        if content:
            print(content, end="", flush=True)
            full_response.append(content)

    print()
    return "".join(full_response)


if __name__ == "__main__":
    print(f"Model: {MODEL}")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            prompt = input("You: ").strip()

            if prompt.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break

            if not prompt:
                continue

            print("\nAssistant: ", end="", flush=True)

            answer = stream_response(prompt)

            print("\n--- Response complete ---\n")

        except KeyboardInterrupt:
            print("\nExiting...")
            break

        except Exception as exc:
            print(f"\nStreaming error: {exc}\n")
