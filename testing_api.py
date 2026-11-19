import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


# from google import genai
# client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
# x=''
# while x not in ['bye','exit','finish']:
#   x=input('enter your query:')
#   response = client.models.generate_content(
#     model="gemini-3.1-flash-lite",
#     contents=x,
#    )
#   print(response.text)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    default_headers={
        "HTTP-Referer": "http://localhost",
        "X-Title": "corrective-rag",
    },
)

model = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")

while True:
    query = input('enter your query:')
    if query in ['bye', 'exit', 'finish']:
        break
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": query}],
    )
    print(response.choices[0].message.content)
