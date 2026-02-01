import argparse
import os

from dotenv import load_dotenv
from google import genai


def main():
    parser = argparse.ArgumentParser(description="Chatbot")
    parser.add_argument("user_prompt", type=str, help="User prompt")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("No api key found")
    client = genai.Client(api_key=api_key)

    print(f"User prompt: {args.user_prompt}")
    response = client.models.generate_content(
        model="gemini-3-flash-preview", contents=args.user_prompt
    )
    if not response.usage_metadata:
        raise RuntimeError("API request has failed")
    print(f"Prompt tokens: {response.usage_metadata.prompt_token_count}")
    print(f"Response tokens: {response.usage_metadata.candidates_token_count}")
    print("Response:")
    print(response.text)


if __name__ == "__main__":
    main()
