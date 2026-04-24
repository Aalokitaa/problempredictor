import google.generativeai as genai
import os

# Use your actual API Key here or set it as an environment variable
genai.configure(api_key="AIzaSyDs1U5X827EH5c966Dxo8w19SZcDxucDLs")

def get_kachua_advice(risk_score, top_drivers):
    model = genai.GenerativeModel('gemini-1.5-pro')