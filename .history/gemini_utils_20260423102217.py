import google.generativeai as genai
import os

# 1. Using the 'latest' alias is more stable than specific versions
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")
genai.configure(api_key=API_KEY)

def test_connection():
    try:
        # Try 'gemini-1.5-flash-latest' or 'gemini-1.5-pro-latest'
        # These aliases automatically point to the best available version
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        
        print("Sending test prompt to Gemini...")
        response = model.generate_content("Hello! I am building KACHUA. Say hi back!")
        
        print("-" * 30)
        print("SUCCESS! Gemini says:")
        print(response.text)
        print("-" * 30)
        
    except Exception as e:
        # If 1.5-flash still fails, let's try the Pro model you're paying for
        print(f"Flash failed, trying Gemini 1.5 Pro...")
        try:
            model_pro = genai.GenerativeModel('gemini-1.5-pro-latest')
            response = model_pro.generate_content("Hello from KACHUA!")
            print(f"SUCCESS with Pro: {response.text}")
        except Exception as e2:
            print(f"CRITICAL ERROR: {e2}")

if __name__ == "__main__":
    test_connection()