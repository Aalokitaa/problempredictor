import os
import google.generativeai as genai
import json

def get_advice(student_name, risk_score, prediction_label, diagnostic_flags, academic_data, wellbeing_answers):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "I'm sorry, I'm currently unable to generate personalized advice because the AI service is not configured. Please reach out to your counselor directly."

    genai.configure(api_key=api_key)
    
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    flags_formatted = json.dumps(diagnostic_flags, indent=2)
    academic_formatted = json.dumps(academic_data, indent=2)
    
    prompt = f"""
    You are a warm, supportive, and highly empathetic academic counselor named Kachua. Your goal is to support students emotionally and academically.
    You are speaking directly to a student named {student_name}.
    
    Here is their current profile:
    - Academic Risk Score: {risk_score:.1f}% ({prediction_label})
    - Academic Data Points: {academic_formatted}
    - Key Wellbeing Signals / Diagnostic Flags (with confidence levels): {flags_formatted}
    
    Instructions:
    Write directly to the student in the second person. Use a warm, empathetic, and encouraging tone. Never be dismissive. Acknowledge any difficulties genuinely. End with hope and concrete steps.
    
    Your response MUST be structured with exactly these exact section headers (including the exact text, e.g., 'Personal Message', do not add Markdown formatting to the headers themselves or you can use ## headers, but the parser looks for these exact words as headings):
    
    Personal Message
    (Write a warm greeting and empathetic message acknowledging their current state)
    
    Academic Support Plan
    (Concrete, tailored academic advice based on their risk score and academic data)
    
    Wellbeing Recommendations
    (Compassionate advice targeting their specific wellbeing flags and signals)
    
    Immediate Next Steps
    (2-3 very actionable, small steps they can take today or tomorrow)
    
    Closing Note
    (A warm, hopeful sign-off from Kachua)
    
    Max output tokens: 2000.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return """
Personal Message
I'm so glad you took the time to share how you're doing. Thank you for trusting me.

Academic Support Plan
Based on what you've shared, I recommend speaking with an academic advisor. They can help adjust your workload to something more manageable.

Wellbeing Recommendations
Please remember that your wellbeing comes first. If you are struggling, don't hesitate to reach out to student support services.

Immediate Next Steps
1. Drink a glass of water and take a 10-minute screen-free break.
2. Email your academic advisor to set up a quick chat.

Closing Note
Remember, you are doing better than you think. Kachua is here for you.
"""