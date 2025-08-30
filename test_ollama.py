import asyncio
import litellm

async def test_ollama_connection():
    print("Attempting to connect to Ollama...")
    try:
        response = await litellm.acompletion(
            model="ollama/llama3",
            messages=[{"role": "user", "content": "Hello, Ollama!"}],
            api_base="http://localhost:11434"
        )
        print("Successfully connected to Ollama!")
        print(f"Ollama response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"Failed to connect to Ollama: {e}")

if __name__ == "__main__":
    asyncio.run(test_ollama_connection())
