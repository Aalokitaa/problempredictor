"""
KACHUA Flask Application
Complete AI-powered student risk assessment and intervention system
Author: Aalokita Chibb
"""

from flask import Flask, render_template, request, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import numpy as np
import json
import os
import uuid
from datetime import datetime
from dotenv import load_dotenv
import joblib
from io import StringIO, BytesIO

# Import custom modules
from models import db, StudentProfile, TeacherAction, WellbeingResponse, DiagnosticSession
from gemini_utils import get_advice

load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///kachua.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'kachua_secret_2024')

# Initialize database
db.init_app(app)

# Global model and encoder cache
MODEL = None
ENCODERS = {}
FEATURE_NAMES = None

def load_model_and_encoders():
    """Load trained RandomForest model and label encoders at startup."""
    global MODEL, ENCODERS, FEATURE_NAMES
    
    try:
        MODEL = joblib.load('models/rf_model.pkl')
        
        encoder_files = ['Gender', 'AddressType', 'SchoolSup', 'FamSup', 'Internet', 'HigherEd']
        for col in encoder_files:
            encoder_path = f'models/encoder_{col}.pkl'
            if os.path.exists(encoder_path):
                ENCODERS[col] = joblib.load(encoder_path)
        
        FEATURE_NAMES = ['Age', 'Gender', 'AddressType', 'G1_Score', 'PastFailures', 
                        'StudyTime', 'Absences', 'GoOut', 'Dalc', 'Walc', 'FreeTime', 
                        'SchoolSup', 'FamSup', 'Internet', 'HigherEd']
        
        print("✓ Model and encoders loaded successfully")
        return True
    except Exception as e:
        print(f"⚠ Warning: Could not load model - {e}")
        return False

def score_wellbeing_responses(responses):
    """
    Score wellbeing responses and generate diagnostic flags.
    Implements the full diagnostic scoring logic for all 19 conditions.
    """
    flags = []
    recommendations = {}
    
    # Map question responses
    q_responses = responses
    
    # ADHD detection (Q13, Q14, Q15, Q16, Q17, Q70, Q71, Q75)
    adhd_count = sum([
        q_responses.get('Q13', False),
        q_responses.get('Q14', False),
        q_responses.get('Q15', False),
        q_responses.get('Q16', False),
        q_responses.get('Q17', False),
        q_responses.get('Q70', False),
        q_responses.get('Q71', False),
        q_responses.get('Q75', False),
    ])
    if adhd_count >= 4:
        confidence = 'Likely' if adhd_count >= 6 else 'Possible'
        flags.append({'condition': 'ADHD Indicators', 'confidence': confidence, 'description': 'Attention and executive function patterns'})
        recommendations['ADHD'] = "Consider speaking with a counselor about an ADHD screening. In the meantime try body doubling — studying alongside someone else even silently. Use the Pomodoro technique: 25 minutes of focused work then a 5-minute break. Try external tools like timers, colour-coded planners, and app blockers during study. Inform your institution's disability support office — you may be entitled to extended exam time or alternative assessment formats. Remember — these results are not a diagnosis. Please speak to a qualified professional."
    
    # Dyslexia detection (Q19, Q20, Q23, Q24, Q65)
    dyslexia_count = sum([
        q_responses.get('Q19', False),
        q_responses.get('Q20', False),
        q_responses.get('Q23', False),
        q_responses.get('Q24', False),
        q_responses.get('Q65', False),
    ])
    if dyslexia_count >= 3:
        confidence = 'Likely' if dyslexia_count >= 4 else 'Possible'
        flags.append({'condition': 'Dyslexia Indicators', 'confidence': confidence, 'description': 'Reading and written expression patterns'})
        recommendations['Dyslexia'] = "Request a psychoeducational assessment through your institution. Use text-to-speech tools like Natural Reader or built-in phone accessibility features when reading. Try audiobooks and recorded lectures instead of dense written text. Write notes using mind maps or voice memos rather than linear text. Font choice matters — try OpenDyslexic or Arial at larger sizes. Disability support at your college may offer scribes or extra time. Remember — these results are not a diagnosis. Please speak to a qualified professional."
    
    # Anxiety Disorder (Q25, Q26, Q27, Q31, Q81)
    anxiety_count = sum([
        q_responses.get('Q25', False),
        q_responses.get('Q26', False),
        q_responses.get('Q27', False),
        q_responses.get('Q31', False),
        q_responses.get('Q81', False),
    ])
    if anxiety_count >= 3:
        confidence = 'Likely' if anxiety_count >= 4 else 'Possible'
        flags.append({'condition': 'Anxiety Disorder', 'confidence': confidence, 'description': 'Persistent worry affecting functioning'})
        recommendations['Anxiety'] = "Speak to a counselor or mental health professional — anxiety is very treatable and you deserve support. Practice slow diaphragmatic breathing before high-pressure situations: inhale 4 counts, hold 4, exhale 6. Reduce caffeine which amplifies anxiety symptoms. Try journaling your worries to externalize them. Apps like Headspace or Calm offer free student plans. In India you can contact iCall at 9152987821 for free confidential counseling. Remember — these results are not a diagnosis. Please speak to a qualified professional."
    
    # Depression (Q28, Q29, Q31, Q78, Q79)
    depression_indicators = sum([
        q_responses.get('Q28', False),
        q_responses.get('Q29', False),
    ])
    depression_additional = sum([
        q_responses.get('Q31', False),
        q_responses.get('Q78', False),
        q_responses.get('Q79', False),
    ])
    if depression_indicators >= 2 and depression_additional >= 1:
        flags.append({'condition': 'Depression Indicators', 'confidence': 'Possible', 'description': 'Persistent low mood and loss of interest'})
        recommendations['Depression'] = "Please reach out to a mental health professional or counselor. You do not have to manage this alone. Try to maintain a basic daily structure even on low days: a consistent wake time, one meal, one small task. Physical movement — even a 10-minute walk — has clinically proven effects on mood. Avoid isolating. If you are in India you can contact iCall at 9152987821 for free support. Remember — these results are not a diagnosis. Please speak to a qualified professional."
    
    # Sleep Disorder (Q35, Q36, Q37, Q38, Q39, Q41)
    sleep_concerns = sum([
        q_responses.get('Q35', False),  # Feels tired
        q_responses.get('Q36', False),  # Delayed sleep phase
        q_responses.get('Q37', False),  # Snoring/apnea
        q_responses.get('Q38', False),  # Daytime sleep attacks
        q_responses.get('Q39', False),  # Restless legs
        q_responses.get('Q41', False),  # Inconsistent schedule
    ])
    if sleep_concerns >= 3:
        confidence = 'Likely' if sleep_concerns >= 4 else 'Possible'
        flags.append({'condition': 'Sleep Disorder', 'confidence': confidence, 'description': 'Significant sleep pattern disruptions'})
        recommendations['Sleep'] = "Maintain a consistent sleep and wake time even on weekends. Avoid screens for at least 30 minutes before bed. Keep your room cool and dark. Avoid caffeine after 2pm. If you suspect sleep apnea or narcolepsy based on your answers please see a doctor — these are medical conditions with very effective treatments. Remember — these results are not a diagnosis. Please speak to a qualified professional."
    
    # Screen Addiction/Digital Overwhelm (Q69, Q70, Q71, Q73, Q74, Q75)
    screen_concerns = sum([
        q_responses.get('Q69', False),  # High screen time
        q_responses.get('Q70', False),  # Phone during study
        q_responses.get('Q71', False),  # Notifications interrupt
        q_responses.get('Q73', False),  # Social media makes worse
        q_responses.get('Q74', False),  # Screens after 10pm
        q_responses.get('Q75', False),  # Focus getting shorter
    ])
    if screen_concerns >= 3:
        flags.append({'condition': 'Digital Overwhelm', 'confidence': 'Possible', 'description': 'Excessive screen time affecting attention'})
        recommendations['Screen'] = "Use your phone's built-in screen time or digital wellbeing tools to set hard app limits. Delete social media apps from your home screen — friction reduces impulsive use. Try a 30-minute phone-free morning routine. Use website blockers like Cold Turkey or Freedom during study blocks. Batch social media into one 20-minute slot per day. Remember — these results are not a diagnosis. Please speak to a qualified professional."
    
    # Burnout (Q78, Q77, Q81)
    if q_responses.get('Q78', False) and q_responses.get('Q77', False) and q_responses.get('Q81', False):
        flags.append({'condition': 'Burnout', 'confidence': 'Possible', 'description': 'Exhaustion and disengagement from studies'})
        recommendations['Burnout'] = "Burnout requires genuine rest — not just sleep but psychological detachment from academic pressure. Schedule non-negotiable rest activities that are not screens: walking, cooking, music, creative hobbies. Talk to your academic advisor about your workload. It is okay to ask for extensions. Recovery from burnout takes weeks — be patient with yourself. Remember — these results are not a diagnosis. Please speak to a qualified professional."
    
    # Imposter Syndrome (Q79, Q80)
    if q_responses.get('Q79', False) and q_responses.get('Q80', False):
        flags.append({'condition': 'Imposter Syndrome', 'confidence': 'Possible', 'description': 'Feeling undeserving despite academic capability'})
        recommendations['Imposter'] = "Imposter syndrome is extremely common especially among high-achieving students and first-generation college students. Your presence here is not an accident. Try writing down three things you did well each week — evidence counters the feeling. Talk openly about it with someone you trust. Understanding it cognitively helps reduce its power. You belong here. Remember — these results are not a diagnosis. Please speak to a qualified professional."
    
    # Environmental Risk (Q61, Q62, Q63, Q64)
    env_concerns = sum([
        not q_responses.get('Q61', True),  # Not safe study environment
        q_responses.get('Q62', False),  # Pressure at home
        q_responses.get('Q63', False),  # Bullying
        q_responses.get('Q64', False),  # Too many work hours
    ])
    if env_concerns >= 2:
        flags.append({'condition': 'Environmental Risk', 'confidence': 'Possible', 'description': 'External barriers to learning'})
        recommendations['Environment'] = "Your home or social environment appears to be significantly affecting your ability to learn. Please speak to your institution's student welfare office — many offer quiet study spaces, counseling, and emergency support. If you are experiencing bullying report it formally and seek support from a trusted staff member. If financial pressure is a concern ask about emergency bursaries or scholarship support. Remember — these results are not a diagnosis. Please speak to a qualified professional."
    
    # Chronic Fatigue (Q48, Q51, Q50)
    if q_responses.get('Q48', False) and q_responses.get('Q51', False):
        flags.append({'condition': 'Chronic Fatigue', 'confidence': 'Possible', 'description': 'Persistent exhaustion and brain fog'})
        recommendations['Fatigue'] = "Ask your doctor for a blood test checking iron, B12, vitamin D, and thyroid function. These are very commonly low in students and directly cause brain fog, fatigue, and difficulty concentrating. Eat at least two full meals per day. Carry snacks. Aim for 2 litres of water daily. Dehydration alone causes measurable cognitive impairment. Remember — these results are not a diagnosis. Please speak to a qualified professional."
    
    # OCD (Q30, Q25)
    if q_responses.get('Q30', False) and q_responses.get('Q25', False):
        flags.append({'condition': 'OCD Indicators', 'confidence': 'Possible', 'description': 'Intrusive thoughts and repetitive behaviors'})
        recommendations['OCD'] = "OCD is a recognized and treatable condition. Cognitive Behavioral Therapy, specifically a technique called ERP, is highly effective. Please speak to a mental health professional. Avoid reassurance-seeking behaviors which can reinforce the cycle. You are not your thoughts — they are symptoms, not facts. Remember — these results are not a diagnosis. Please speak to a qualified professional."
    
    # Vision/Hearing Concerns (Q56, Q57, Q58)
    sensory_concerns = sum([
        q_responses.get('Q56', False),
        q_responses.get('Q57', False),
        q_responses.get('Q58', False),
    ])
    if sensory_concerns >= 1:
        flags.append({'condition': 'Sensory Concerns', 'confidence': 'Possible', 'description': 'Vision or hearing affecting learning'})
        recommendations['Sensory'] = "Please get a professional eye or hearing test if you have not had one in the past year — these are often the simplest fixes with the biggest academic impact. Sit closer to the front of the class. Ask for lecture slides in advance. Use subtitles on all video content. Inform your institution's disability office even before a formal diagnosis. Remember — these results are not a diagnosis. Please speak to a qualified professional."
    
    # Eating Disorder Indicators (Q33, Q54)
    if q_responses.get('Q33', False) and q_responses.get('Q54', False):
        flags.append({'condition': 'Eating Disorder', 'confidence': 'Possible', 'description': 'Disordered eating affecting health'})
        recommendations['Eating'] = "Disordered eating patterns affect brain function, energy, and mood significantly. Please speak to a doctor and a counselor. Eating disorders are medical conditions not choices. In India the Vandrevala Foundation helpline at 1860-2662-345 is available 24 hours. Recovery is possible and support is available. Remember — these results are not a diagnosis. Please speak to a qualified professional."
    
    # Gifted and Understimulated (Q82, Q77, Q15, Q83)
    if q_responses.get('Q82', False) and q_responses.get('Q77', False) and q_responses.get('Q15', False):
        flags.append({'condition': 'Potential Giftedness', 'confidence': 'Possible', 'description': 'May be understimulated by current coursework'})
        recommendations['Gifted'] = "You may be experiencing boredom or under-stimulation rather than academic difficulty. Seek out advanced electives, research opportunities, or independent study projects. Speak to a professor whose work interests you about getting involved. Enrichment outside class — online courses, competitions, personal projects — can re-engage your mind. Consider whether your current course is the right fit and speak to an academic advisor. Remember — these results are not a diagnosis. Please speak to a qualified professional."
    
    return flags, recommendations

@app.with_appcontext
def init_db():
    """Initialize database tables."""
    db.create_all()

# Routes

@app.route('/')
def index():
    """Render main landing page."""
    return render_template('index.html')

@app.route('/analysis')
def analysis():
    """Render feature importance analysis page."""
    feature_importances = []
    if MODEL:
        importances = MODEL.feature_importances_
        indices = np.argsort(importances)[::-1]
        for idx in indices:
            feature_importances.append({
                'name': FEATURE_NAMES[idx],
                'importance': float(importances[idx]),
                'percentage': float(importances[idx]) * 100
            })
    
    return render_template('analysis.html', features=feature_importances)

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Academic assessment prediction endpoint.
    Accepts academic data and student info, returns risk prediction.
    """
    try:
        data = request.get_json()
        
        if not MODEL or not ENCODERS:
            return jsonify({'error': 'Model not loaded'}), 500
        
        student_name = data.get('student_name', 'Student')
        student_uid = data.get('student_uid', str(uuid.uuid4()))
        student_contact = data.get('student_contact', '')
        
        # Prepare features for prediction
        features_dict = {
            'Age': int(data.get('Age', 18)),
            'Gender': data.get('Gender', 'Male'),
            'AddressType': data.get('AddressType', 'Urban'),
            'G1_Score': int(data.get('G1_Score', 50)),
            'PastFailures': int(data.get('PastFailures', 0)),
            'StudyTime': int(data.get('StudyTime', 2)),
            'Absences': int(data.get('Absences', 0)),
            'GoOut': int(data.get('GoOut', 3)),
            'Dalc': int(data.get('Dalc', 1)),
            'Walc': int(data.get('Walc', 1)),
            'FreeTime': int(data.get('FreeTime', 3)),
            'SchoolSup': data.get('SchoolSup', 'No'),
            'FamSup': data.get('FamSup', 'No'),
            'Internet': data.get('Internet', 'Yes'),
            'HigherEd': data.get('HigherEd', 'Yes'),
        }
        
        # Encode categorical features
        for col in ['Gender', 'AddressType', 'SchoolSup', 'FamSup', 'Internet', 'HigherEd']:
            if col in ENCODERS:
                features_dict[col] = ENCODERS[col].transform([features_dict[col]])[0]
        
        # Prepare feature array
        features_array = np.array([
            [features_dict['Age'], features_dict['Gender'], features_dict['AddressType'],
             features_dict['G1_Score'], features_dict['PastFailures'], features_dict['StudyTime'],
             features_dict['Absences'], features_dict['GoOut'], features_dict['Dalc'],
             features_dict['Walc'], features_dict['FreeTime'], features_dict['SchoolSup'],
             features_dict['FamSup'], features_dict['Internet'], features_dict['HigherEd']]
        ])
        
        # Make prediction
        prediction = MODEL.predict(features_array)[0]
        probabilities = MODEL.predict_proba(features_array)[0]
        risk_score = probabilities[1]  # Probability of At Risk
        prediction_label = 'AT RISK' if prediction == 1 else 'SAFE'
        
        # Save to database
        student = StudentProfile(
            name=student_name,
            uid=student_uid,
            contact=student_contact,
            risk_score=float(risk_score),
            prediction_label=prediction_label
        )
        db.session.add(student)
        db.session.commit()
        
        return jsonify({
            'risk_score': float(risk_score),
            'prediction_label': prediction_label,
            'probability_safe': float(probabilities[0]),
            'probability_atrisk': float(probabilities[1]),
            'student_id': student.id,
            'academic_data': features_dict
        }), 200
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/wellbeing', methods=['POST'])
def wellbeing():
    """
    Wellbeing assessment scoring endpoint.
    Scores 83 wellbeing questions and returns diagnostic flags and recommendations.
    """
    try:
        data = request.get_json()
        
        student_name = data.get('student_name', 'Student')
        student_uid = data.get('student_uid', str(uuid.uuid4()))
        share_with_teacher = data.get('share_with_teacher', False)
        responses = data.get('responses', {})
        
        # Score wellbeing responses
        flags, recommendations = score_wellbeing_responses(responses)
        
        # Save to database
        wellbeing = WellbeingResponse(
            student_name=student_name,
            student_uid=student_uid,
            share_with_teacher=share_with_teacher
        )
        wellbeing.set_responses(responses)
        wellbeing.set_diagnostic_flags(flags)
        wellbeing.set_recommendations(recommendations)
        
        db.session.add(wellbeing)
        db.session.commit()
        
        return jsonify({
            'diagnostic_flags': flags,
            'recommendations': recommendations,
            'wellbeing_id': wellbeing.id,
            'flag_count': len(flags)
        }), 200
    
    except Exception as e:
        print(f"Wellbeing error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/gemini_advice', methods=['POST'])
def gemini_advice():
    """
    Generate Gemini AI personalized advice.
    Combines academic and wellbeing data for comprehensive support plan.
    """
    try:
        data = request.get_json()
        
        student_name = data.get('student_name', 'Student')
        risk_score = float(data.get('risk_score', 0.5))
        prediction_label = data.get('prediction_label', 'SAFE')
        diagnostic_flags = data.get('diagnostic_flags', [])
        academic_data = data.get('academic_data', {})
        wellbeing_answers = data.get('wellbeing_answers', {})
        
        # Generate advice from Gemini
        advice = get_advice(student_name, risk_score, prediction_label, diagnostic_flags, academic_data, wellbeing_answers)
        
        # Create session and save
        session_token = str(uuid.uuid4())
        session = DiagnosticSession(
            session_token=session_token,
            combined_risk_score=risk_score,
            gemini_advice=advice
        )
        session.set_academic_data(academic_data)
        session.set_wellbeing_data(wellbeing_answers)
        
        db.session.add(session)
        db.session.commit()
        
        return jsonify({
            'advice': advice,
            'session_id': session_token,
            'created_at': session.created_at.isoformat()
        }), 200
    
    except Exception as e:
        print(f"Gemini advice error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/result/<session_id>')
def result(session_id):
    """Render personalized result report."""
    try:
        session = DiagnosticSession.query.filter_by(session_token=session_id).first()
        if not session:
            return "Session not found", 404
        
        return render_template('result.html',
            session_id=session_id,
            risk_score=session.combined_risk_score,
            prediction_label='AT RISK' if session.combined_risk_score > 0.5 else 'SAFE',
            diagnostic_flags=session.get_wellbeing_data().get('diagnostic_flags', []),
            gemini_advice=session.gemini_advice,
            created_at=session.created_at.isoformat()
        )
    except Exception as e:
        print(f"Result error: {e}")
        return str(e), 500

@app.route('/api/students')
def get_students():
    """Fetch all student profiles."""
    try:
        students = StudentProfile.query.all()
        return jsonify([s.to_dict() for s in students]), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/wellbeing_reports')
def get_wellbeing_reports():
    """Fetch wellbeing reports shared with teachers."""
    try:
        reports = WellbeingResponse.query.filter_by(share_with_teacher=True).all()
        return jsonify([r.to_dict() for r in reports]), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/remark', methods=['POST'])
def log_remark():
    """Log teacher intervention remark."""
    try:
        data = request.get_json()
        
        student_id = data.get('student_id')
        remark_text = data.get('remark_text', '')
        is_bulk = data.get('is_bulk', False)
        
        action = TeacherAction(
            student_id=student_id,
            remark_text=remark_text,
            is_bulk=is_bulk
        )
        db.session.add(action)
        db.session.commit()
        
        return jsonify({'success': True, 'action_id': action.id}), 200
    
    except Exception as e:
        print(f"Remark error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/bulk_analyze', methods=['POST'])
def bulk_analyze():
    """
    Bulk student data analysis endpoint.
    Accepts CSV/Excel file, runs predictions, returns comprehensive analysis.
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not MODEL or not ENCODERS:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Determine file type
        filename = file.filename.lower()
        
        try:
            if filename.endswith('.csv'):
                df = pd.read_csv(file)
            elif filename.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file)
            elif filename.endswith('.json'):
                df = pd.read_json(file)
            else:
                return jsonify({'error': 'Unsupported file format'}), 400
        except Exception as e:
            return jsonify({'error': f'File parsing error: {str(e)}'}), 400
        
        # Validate required columns
        required_cols = ['Age', 'Gender', 'AddressType', 'G1_Score', 'PastFailures', 
                        'StudyTime', 'Absences', 'GoOut', 'Dalc', 'Walc', 'FreeTime',
                        'SchoolSup', 'FamSup', 'Internet', 'HigherEd']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return jsonify({'error': f'Missing columns: {", ".join(missing_cols)}'}), 400
        
        # Fill missing values
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown', inplace=True)
            else:
                df[col].fillna(df[col].median(), inplace=True)
        
        # Prepare data for prediction
        df_pred = df.copy()
        
        for col in ['Gender', 'AddressType', 'SchoolSup', 'FamSup', 'Internet', 'HigherEd']:
            if col in ENCODERS:
                df_pred[col] = df[col].apply(lambda x: ENCODERS[col].transform([x])[0])
        
        # Make predictions
        X = df_pred[required_cols].values
        predictions = MODEL.predict(X)
        probabilities = MODEL.predict_proba(X)
        
        # Add results to dataframe
        df['Risk_Score'] = probabilities[:, 1]
        df['Risk_Label'] = predictions
        df['Prediction'] = df['Risk_Label'].map({1: 'AT RISK', 0: 'SAFE'})
        
        # Save all to database
        for idx, row in df.iterrows():
            student = StudentProfile(
                name=row.get('Name', f'Student_{idx}'),
                uid=str(uuid.uuid4()),
                risk_score=float(row['Risk_Score']),
                prediction_label=row['Prediction'],
                bulk_upload=True
            )
            db.session.add(student)
        
        db.session.commit()
        
        # Calculate statistics
        total_students = len(df)
        at_risk_count = (predictions == 1).sum()
        safe_count = (predictions == 0).sum()
        at_risk_percentage = (at_risk_count / total_students * 100) if total_students > 0 else 0
        average_risk_score = probabilities[:, 1].mean()
        
        # Top risk factors
        importances = MODEL.feature_importances_
        indices = np.argsort(importances)[::-1][:5]
        top_risk_factors = [
            {'factor': FEATURE_NAMES[idx], 'importance': float(importances[idx])}
            for idx in indices
        ]
        
        # Risk distribution
        risk_distribution = {
            '0-20%': ((probabilities[:, 1] < 0.2).sum()),
            '20-40%': (((probabilities[:, 1] >= 0.2) & (probabilities[:, 1] < 0.4)).sum()),
            '40-60%': (((probabilities[:, 1] >= 0.4) & (probabilities[:, 1] < 0.6)).sum()),
            '60-80%': (((probabilities[:, 1] >= 0.6) & (probabilities[:, 1] < 0.8)).sum()),
            '80-100%': ((probabilities[:, 1] >= 0.8).sum()),
        }
        
        # Highest risk students
        df_sorted = df.sort_values('Risk_Score', ascending=False).head(10)
        highest_risk_students = df_sorted[['Name', 'Risk_Score', 'G1_Score', 'Absences']].to_dict('records')
        
        return jsonify({
            'total_students': total_students,
            'at_risk_count': int(at_risk_count),
            'safe_count': int(safe_count),
            'at_risk_percentage': float(at_risk_percentage),
            'average_risk_score': float(average_risk_score),
            'top_risk_factors': top_risk_factors,
            'risk_distribution': risk_distribution,
            'highest_risk_students': highest_risk_students,
            'message': 'Analysis complete'
        }), 200
    
    except Exception as e:
        print(f"Bulk analysis error: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    with app.app_context():
        init_db()
        load_model_and_encoders()
        app.run(debug=True, host='0.0.0.0', port=5000)
