import google.generativeai as genai

genai.configure(api_key="AIzaSyA4fLK_34Of33Y7i0kH1S1IXqyNWz4h_Hc")

print("Checking available models for your key...")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(f"ID: {m.name}")