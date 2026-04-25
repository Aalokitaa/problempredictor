import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set")

genai.configure(api_key=api_key)

print("Checking available models for your key...")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(f"ID: {m.name}")