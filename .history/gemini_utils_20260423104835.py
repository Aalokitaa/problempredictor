"""
KACHUA Gemini AI Integration
Provides empathetic, personalized academic support advice using Google Gemini API
"""

import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Gemini API
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")

genai.configure(api_key=API_KEY)

def get_advice(student_name, risk_score, prediction_label, diagnostic_flags, academic_data, wellbeing_answers):
    """
    Generate personalized, empathetic Gemini advice for a student.
    
    Args:
        student_name: Student's first name
        risk_score: Normalized risk score (0-1)
        prediction_label: "SAFE" or "AT RISK"
        diagnostic_flags: List of dicts with {condition, confidence, description}
        academic_data: Dict with academic assessment answers
        wellbeing_answers: Dict with wellbeing survey answers
    
    Returns:
        String containing structured advice with sections:
        - Personal Message
        - Academic Support Plan
        - Wellbeing Recommendations
        - Immediate Next Steps
        - Closing Note
    """
    
    try:
        # Build diagnostic flags summary
        flags_summary = ""
        if diagnostic_flags:
            for flag in diagnostic_flags:
                flags_summary += f"- {flag.get('condition', 'Unknown')}: {flag.get('confidence', 'Possible')} (confidence)\n"
        else:
            flags_summary = "No significant wellbeing concerns flagged at this time."
        
        # Extract key academic data points
        academic_summary = f"""
Academic Performance: {academic_data.get('G1_Score', 'N/A')}/100
Study Time: {academic_data.get('StudyTime', 0)} hours/week
Absences: {academic_data.get('Absences', 0)} sessions
Past Failures: {academic_data.get('PastFailures', 0)}
Family Support: {academic_data.get('FamSup', 'Unknown')}
School Support: {academic_data.get('SchoolSup', 'Unknown')}
"""
        
        # Extract key wellbeing signals
        wellbeing_summary = f"""
Sleep Quality: {wellbeing_answers.get('sleep_quality', 'Not assessed')}
Stress Level: {wellbeing_answers.get('stress_level', 'Not assessed')}
Social Support: {wellbeing_answers.get('has_support', 'Not assessed')}
Physical Health: {wellbeing_answers.get('physical_concerns', 'Not assessed')}
"""
        
        # Build comprehensive prompt
        prompt = f"""
You are Kachua, a warm, empathetic academic counselor. You're responding to a student named {student_name} who has completed an academic and wellbeing assessment.

STUDENT CONTEXT:
- Name: {student_name}
- Academic Risk Level: {prediction_label}
- Risk Score: {risk_score:.1%}

FLAGGED WELLBEING CONDITIONS:
{flags_summary}

ACADEMIC SNAPSHOT:
{academic_summary}

WELLBEING INDICATORS:
{wellbeing_summary}

Please provide a structured response that is:
1. Genuinely empathetic and never dismissive
2. Warm and hopeful in tone
3. Practical with specific actionable steps
4. Written in second person directly to {student_name}
5. Acknowledging real challenges without minimizing them

Format your response with these exact section headers (use ### for headers):

### Personal Message
A warm, personalized opening addressing {student_name} by name. Acknowledge their current situation genuinely. If at risk, express care. If safe, celebrate their progress.

### Academic Support Plan
Specific, concrete steps to strengthen academic performance. Reference their particular situation (e.g., if low study time, suggest structured techniques like Pomodoro). Include institutional resources they can access.

### Wellbeing Recommendations
Based on the flagged conditions, provide specific self-care and support strategies. If no conditions flagged, focus on maintaining wellbeing. Include crisis resources for India if applicable.

### Immediate Next Steps
3-5 bullet points of things to do this week. Be realistic and achievable.

### Closing Note
End with genuine hope, reminding them that challenges are temporary and help is available. Sign off as "Kachua" with the tagline "We're there for you ;)"

Max output: 2000 tokens. Be concise but comprehensive.
"""
        
        # Call Gemini model
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(
            prompt,
            generation_config={
                'max_output_tokens': 2000,
                'temperature': 0.7,
            }
        )
        
        return response.text
        
    except Exception as e:
        print(f"Gemini API Error: {e}")
        
        # Fallback message if API fails
        fallback = f"""
### Personal Message
Hi {student_name}, thank you for taking the time to complete this assessment. Kachua sees you and recognizes that you're reaching out for support. That takes courage.

### Academic Support Plan
Consider reaching out to your institution's academic support services. They can help you develop a personalized study plan, connect you with tutoring resources, and explore options like deadline extensions if needed. Many students benefit from structured study techniques and group study sessions.

### Wellbeing Recommendations
Take time for activities that help you recharge. This might be time outdoors, creative pursuits, time with people you trust, or simply rest. If you're experiencing difficulties with mood, sleep, focus, or other aspects of wellbeing, speaking with a counselor or doctor can make a real difference.

### Immediate Next Steps
- Reach out to your institution's student support office
- Schedule a meeting with an academic advisor if available
- Identify one study technique you'd like to try this week
- Speak with someone you trust about what you're experiencing
- Schedule a health or wellness check if you haven't recently

### Closing Note
Remember that struggles are temporary, and they don't define your capability or potential. Resources exist to support you, and reaching out is a sign of strength, not weakness.

Kachua is here for you ;)
"""
        return fallback

def test_connection():
    """Test Gemini API connection."""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        print("Sending test prompt to Gemini...")
        response = model.generate_content("Hello! I am building KACHUA. Say hi back!")
        
        print("-" * 30)
        print("SUCCESS! Gemini says:")
        print(response.text)
        print("-" * 30)
        
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")

if __name__ == "__main__":
    test_connection()