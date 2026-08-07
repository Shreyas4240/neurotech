import re
import os
from huggingface_hub import InferenceClient

client = InferenceClient(
    model="HuggingFaceTB/SmolLM3-3B",
    token=os.environ.get("HF_TOKEN")
)

messages = [
    {"role": "system", "content": "You are a concise, helpful assistant. Do not show your reasoning."},
    {"role": "user", "content": "Find the radius of convergence of \sum_{n=1}^{\infty}{\frac{(n+1)x^{n}}{n^{2}\,6^{n}}}: "}
]

response = client.chat.completions.create(
    model="HuggingFaceTB/SmolLM3-3B",
    messages=messages,
    max_tokens=1000,
    temperature=0.1
)

raw_text = response.choices[0].message.content

# Remove <think>...</think> sections
clean_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()

if not clean_text:
    clean_text = "Sorry, I could not generate a response."

print(clean_text)