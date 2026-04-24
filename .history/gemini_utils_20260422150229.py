import google.generativeai as genai  # ADDED 'ai' AT THE END
import os

# 1. Configuration
API_KEY = "AIzaSyA4fLK_34Of33Y7i0kH1S1IXqyNWz4h_Hc" 
genai.configure(api_key=API_KEY)

def test_connection():
    try:
        # Use 1.5-flash for a quick connectivity test
        model = genai.GenerativeModel('gemini-1.5-flash')
        
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