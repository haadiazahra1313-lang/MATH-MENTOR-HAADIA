import os

from dotenv import load_dotenv
from groq import Groq


# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY was not found. "
        "Please add it to your .env file."
    )

# Create Groq client
client = Groq(api_key=api_key)


# Conversation memory
messages = [
    {
        "role": "system",
        "content": "You are a helpful, intelligent AI assistant."
    }
]


print("=" * 60)
print("              MY AI CHATBOT")
print("=" * 60)
print("Powered by Groq + Llama 3.1 8B")
print("Type 'quit', 'exit', or 'bye' to stop.")
print("=" * 60)


while True:

    user_input = input("\nYou: ")

    # Exit chatbot
    if user_input.lower().strip() in ["quit", "exit", "bye"]:
        print("\nChatbot: Goodbye!")
        break

    # Ignore empty messages
    if not user_input.strip():
        continue

    # Add user's message to conversation
    messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    try:

        print("\nChatbot: ", end="", flush=True)

        # Send conversation to LLM
        stream = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
            stream=True
        )

        # Store complete response
        assistant_response = ""

        # Display response as it arrives
        for chunk in stream:

            if chunk.choices[0].delta.content:

                content = chunk.choices[0].delta.content

                print(
                    content,
                    end="",
                    flush=True
                )

                assistant_response += content

        # Save assistant response to memory
        messages.append(
            {
                "role": "assistant",
                "content": assistant_response
            }
        )

        print()

    except Exception as e:

        print(f"\n\nError: {e}")

        # Remove the failed user message
        messages.pop()
        