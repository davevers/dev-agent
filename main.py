import argparse
import os
import sys

from dotenv import load_dotenv
from google import genai
from google.genai import types

from call_functions import available_functions, call_function
from config import MAX_ITERATIONS
from prompt import system_prompt


def main():
    parser = argparse.ArgumentParser(description="Chatbot")
    parser.add_argument("user_prompt", type=str, help="User prompt")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not found")
    client = genai.Client(api_key=api_key)

    if args.verbose:
        print(f"User prompt: {args.user_prompt}")

    messages = [types.Content(role="user", parts=[types.Part(text=args.user_prompt)])]
    for i in range(MAX_ITERATIONS):
        try:
            final_response = generate_content(client, messages, args.verbose)
            if final_response:
                print("Final response:")
                print(final_response)
                return
        except Exception as e:
            print(f"Error generating content: {e}")

    print(
        f"Maximum number of iterations {MAX_ITERATIONS} reached without a final response"
    )
    sys.exit(1)


def generate_content(client, messages, verbose):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=messages,
        config=types.GenerateContentConfig(
            tools=[available_functions], system_instruction=system_prompt
        ),
    )

    if not response.usage_metadata:
        raise RuntimeError("API request has failed")
    if verbose:
        print(f"Prompt tokens: {response.usage_metadata.prompt_token_count}")
        print(f"Response tokens: {response.usage_metadata.candidates_token_count}")
    if not response.function_calls:
        return response.text

    if response.candidates:
        for candidate in response.candidates:
            if candidate.content:
                messages.append(candidate.content)

    function_responses = []
    for function_call in response.function_calls:
        result = call_function(function_call, verbose)
        if (
            not result.parts
            or not result.parts[0].function_response
            or not result.parts[0].function_response.response
        ):
            raise Exception(f"Empty function call result for {function_call.name}")
        function_responses.append(result.parts[0])
        if verbose:
            print(f"-> {result.parts[0].function_response.response}")

    messages.append(types.Content(role="user", parts=function_responses))


if __name__ == "__main__":
    main()
