import os

from dotenv import load_dotenv
from google import genai
from google.genai.types import UsageMetadata

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("No api key found")

client = genai.Client(api_key=api_key)

prompt = "Why is Boot.dev such a great place to learn backend development? Use one paragraph maximum."
print(f"User prompt: {prompt}")
response = client.models.generate_content(
    model="gemini-3-flash-preview", contents=prompt
)
if not response.usage_metadata:
    raise RuntimeError("API request has failed")
print(
    f"Prompt tokens: {response.usage_metadata.prompt_token_count}\nResponse tokens: {response.usage_metadata.candidates_token_count}"
)
print(f"Response:\n{response.text}")
