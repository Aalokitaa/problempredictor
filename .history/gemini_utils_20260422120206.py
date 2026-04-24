import google.generativeai as genai
import os

# 1. Configuration (Use your key here)
API_KEY = "AIzaSyDs1U5X827EH5c966Dxo8w19SZcDxucDLs" 
genai.configure(api_key=API_KEY)

def test_connection():
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # 3. Simple test prompt
        print("Sending test prompt to Gemini...")
        response = model.generate_content("Hello! I am building KACHUA. Say hi back!")
        
        print("-" * 30)
        print("SUCCESS! Gemini says:")
        print(response.text)
        print("-" * 30)
        
    except Exception as e:
        print(f"ERROR: Could not connect to Gemini. Details: {e}")

if __name__ == "__main__":
    test_connection()