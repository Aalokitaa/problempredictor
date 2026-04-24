import os
import uuid
import json
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

import gemini_utils
from models import db, StudentProfile, TeacherAction, WellbeingResponse, DiagnosticSession

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'kachua_secret_2024')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///kachua.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

# Load Models
rf_model = None
encoders = {}
feature_importances = []
feature_names = []

def load_ml_assets():
    global rf_model, encoders, feature_importances, feature_names
    try:
        rf_model = joblib.load('models/rf_model.pkl')
        categorical_cols = ['Gender', 'AddressType', 'SchoolSup', 'FamSup', 'Internet', 'HigherEd']
        for col in categorical_cols:
            encoders[col] = joblib.load(f'models/encoder_{col}.pkl')
            
        importances = rf_model.feature_importances_
        # Expected feature order based on train_model.py
        expected_features = ['Age', 'Gender', 'AddressType', 'G1_Score', 'PastFailures', 
                             'StudyTime', 'Absences', 'GoOut', 'Dalc', 'Walc', 'FreeTime', 
                             'SchoolSup', 'FamSup', 'Internet', 'HigherEd']
        
        indices = np.argsort(importances)[::-1]
        feature_names = [expected_features[i] for i in indices]
        feature_importances = [float(importances[i]) * 100 for i in indices]
    except Exception as e:
        print(f"Warning: Could not load machine learning models. Please run train_model.py first. Error: {e}")

with app.app_context():
    db.create_all()
    load_ml_assets()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analysis')
def analysis():
    return render_template('analysis.html', 
                           feature_names=feature_names, 
                           feature_importances=feature_importances)

@app.route('/result/<session_id>')
def result(session_id):
    session = DiagnosticSession.query.filter_by(session_token=session_id).first_or_404()
    
    academic_data = json.loads(session.academic_data) if session.academic_data else {}
    wellbeing_data = json.loads(session.wellbeing_data) if session.wellbeing_data else {}
    
    student_name = academic_data.get('student_name', wellbeing_data.get('student_name', 'Student'))
    
    # We parse the diagnostic flags if they are stored in the wellbeing_data or if we saved them separately.
    # To keep it simple, we'll try to extract them from wellbeing_data if they were passed that way, 
    # or expect the frontend to have passed them to Gemini which formatted them.
    # We'll just pass the raw data to template and parse Gemini text
    
    return render_template('result.html',
                           student_name=student_name,
                           risk_score=session.combined_risk_score,
                           prediction_label=academic_data.get('prediction_label', 'UNKNOWN'),
                           diagnostic_flags=wellbeing_data.get('diagnostic_flags', []),
                           gemini_advice=session.gemini_advice,
                           session_id=session.session_token,
                           created_at=session.created_at.strftime("%B %d, %Y"))

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        # Extract fields
        df_input = pd.DataFrame([{
            'Age': data.get('Age', 18),
            'Gender': data.get('Gender', 'Prefer not to say'),
            'AddressType': data.get('AddressType', 'Urban area'),
            'G1_Score': data.get('G1_Score', 50),
            'PastFailures': data.get('PastFailures', 0),
            'StudyTime': data.get('StudyTime', 3),
            'Absences': data.get('Absences', 0),
            'GoOut': data.get('GoOut', 3),
            'Dalc': data.get('Dalc', 1),
            'Walc': data.get('Walc', 1),
            'FreeTime': data.get('FreeTime', 3),
            'SchoolSup': data.get('SchoolSup', 'No'),
            'FamSup': data.get('FamSup', 'No'),
            'Internet': data.get('Internet', 'Yes'),
            'HigherEd': data.get('HigherEd', 'Yes')
        }])
        
        # Encode categorical
        for col, le in encoders.items():
            if col in df_input.columns:
                # Handle unknown classes gracefully
                if df_input[col][0] not in le.classes_:
                    df_input[col] = le.classes_[0]
                df_input[col] = le.transform(df_input[col])
                
        # Predict
        if rf_model:
            prob = rf_model.predict_proba(df_input)[0]
            prob_safe = float(prob[0]) * 100
            prob_risk = float(prob[1]) * 100
            pred = rf_model.predict(df_input)[0]
        else:
            prob_safe, prob_risk, pred = 50.0, 50.0, 0
            
        label = "AT RISK" if pred == 1 else "SAFE"
        risk_score = prob_risk
        
        # Save to DB
        student = StudentProfile(
            name=data.get('student_name', f"Student_{str(uuid.uuid4())[:8]}"),
            uid=data.get('student_uid', str(uuid.uuid4())),
            contact=data.get('student_contact', ''),
            risk_score=risk_score,
            prediction_label=label
        )
        db.session.add(student)
        db.session.commit()
        
        return jsonify({
            'risk_score': risk_score,
            'prediction_label': label,
            'probability_safe': prob_safe,
            'probability_atrisk': prob_risk,
            'student_id': student.id
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/wellbeing', methods=['POST'])
def wellbeing():
    data = request.json
    # Frontend provides all answers.
    # Diagnostic logic
    flags = data.get('diagnostic_flags', [])
    recommendations = data.get('recommendations', {})
    
    try:
        response_record = WellbeingResponse(
            student_name=data.get('student_name', 'Anonymous'),
            student_uid=data.get('student_uid', str(uuid.uuid4())),
            responses=json.dumps(data.get('answers', {})),
            diagnostic_flags=json.dumps(flags),
            recommendations=json.dumps(recommendations),
            share_with_teacher=data.get('share_with_teacher', False)
        )
        db.session.add(response_record)
        db.session.commit()
        
        return jsonify({'success': True, 'id': response_record.id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/gemini_advice', methods=['POST'])
def gemini_advice():
    data = request.json
    try:
        student_name = data.get('student_name', 'Student')
        academic_data = data.get('academic_data', {})
        risk_score = academic_data.get('risk_score', 0)
        prediction_label = academic_data.get('prediction_label', 'UNKNOWN')
        diagnostic_flags = data.get('diagnostic_flags', [])
        wellbeing_answers = data.get('wellbeing_answers', {})
        
        advice_text = gemini_utils.get_advice(
            student_name, risk_score, prediction_label, 
            diagnostic_flags, academic_data, wellbeing_answers
        )
        
        session_token = str(uuid.uuid4())
        ds = DiagnosticSession(
            session_token=session_token,
            academic_data=json.dumps(academic_data),
            wellbeing_data=json.dumps({
                'student_name': student_name,
                'diagnostic_flags': diagnostic_flags,
                'answers': wellbeing_answers
            }),
            combined_risk_score=risk_score,
            gemini_advice=advice_text
        )
        db.session.add(ds)
        db.session.commit()
        
        return jsonify({'session_id': session_token})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/students', methods=['GET'])
def get_students():
    students = StudentProfile.query.order_by(StudentProfile.created_at.desc()).all()
    return jsonify([{
        'id': s.id,
        'name': s.name,
        'uid': s.uid,
        'risk_score': s.risk_score,
        'prediction_label': s.prediction_label,
        'created_at': s.created_at.isoformat()
    } for s in students])

@app.route('/api/wellbeing_reports', methods=['GET'])
def get_wellbeing_reports():
    reports = WellbeingResponse.query.filter_by(share_with_teacher=True).order_by(WellbeingResponse.submitted_at.desc()).all()
    return jsonify([{
        'id': r.id,
        'student_name': r.student_name,
        'student_uid': r.student_uid,
        'diagnostic_flags': json.loads(r.diagnostic_flags),
        'submitted_at': r.submitted_at.isoformat()
    } for r in reports])

@app.route('/api/remark', methods=['POST'])
def remark():
    data = request.json
    try:
        action = TeacherAction(
            student_id=data.get('student_id'),
            remark_text=data.get('remark_text'),
            is_bulk=data.get('is_bulk', False)
        )
        db.session.add(action)
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/bulk_analyze', methods=['POST'])
def bulk_analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file)
        elif file.filename.endswith('.json'):
            df = pd.read_json(file)
        else:
            return jsonify({'error': 'Unsupported file format'}), 400
            
        expected_cols = ['Age', 'Gender', 'AddressType', 'G1_Score', 'PastFailures', 
                         'StudyTime', 'Absences', 'GoOut', 'Dalc', 'Walc', 'FreeTime', 
                         'SchoolSup', 'FamSup', 'Internet', 'HigherEd']
                         
        # Validate columns
        missing_cols = [c for c in expected_cols if c not in df.columns]
        if missing_cols:
            return jsonify({'error': f'Missing columns: {", ".join(missing_cols)}'}), 400
            
        # Fill missing
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna(df[col].median(), inplace=True)
                
        df_ml = df[expected_cols].copy()
        
        # Apply encoders
        for col, le in encoders.items():
            if col in df_ml.columns:
                df_ml[col] = df_ml[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                df_ml[col] = le.transform(df_ml[col])
                
        if rf_model:
            probs = rf_model.predict_proba(df_ml)
            preds = rf_model.predict(df_ml)
            df['Risk_Score'] = probs[:, 1] * 100
            df['Risk_Label'] = ['AT RISK' if p == 1 else 'SAFE' for p in preds]
        else:
            df['Risk_Score'] = 0.0
            df['Risk_Label'] = 'SAFE'
            
        # Save to DB
        students_to_insert = []
        for _, row in df.iterrows():
            name = row.get('Name', f"Bulk_{str(uuid.uuid4())[:8]}")
            uid = row.get('UID', str(uuid.uuid4()))
            contact = row.get('Contact', '')
            students_to_insert.append(StudentProfile(
                name=str(name),
                uid=str(uid),
                contact=str(contact),
                risk_score=float(row['Risk_Score']),
                prediction_label=str(row['Risk_Label']),
                bulk_upload=True
            ))
            
        db.session.bulk_save_objects(students_to_insert)
        db.session.commit()
        
        total_students = len(df)
        at_risk_count = int((df['Risk_Label'] == 'AT RISK').sum())
        safe_count = total_students - at_risk_count
        at_risk_percentage = (at_risk_count / total_students) * 100 if total_students > 0 else 0
        average_risk_score = float(df['Risk_Score'].mean())
        
        # Risk distribution
        bins = [0, 20, 40, 60, 80, 100]
        labels = ['0-20', '20-40', '40-60', '60-80', '80-100']
        df['Risk_Bucket'] = pd.cut(df['Risk_Score'], bins=bins, labels=labels, include_lowest=True)
        risk_distribution = df['Risk_Bucket'].value_counts().to_dict()
        
        highest_risk_students = df.sort_values('Risk_Score', ascending=False).head(10)[['Name', 'UID', 'Risk_Score', 'Risk_Label']].fillna('N/A').to_dict('records')
        
        column_summary = {}
        for col in expected_cols:
            if df[col].dtype == 'object':
                column_summary[col] = {'type': 'categorical', 'mode': str(df[col].mode()[0])}
            else:
                column_summary[col] = {'type': 'numeric', 'mean': float(df[col].mean()), 'min': float(df[col].min()), 'max': float(df[col].max())}
                
        # Format top 5 features
        top_risk_factors = [{'feature': name, 'importance': imp} for name, imp in zip(feature_names[:5], feature_importances[:5])]
        
        return jsonify({
            'total_students': total_students,
            'at_risk_count': at_risk_count,
            'safe_count': safe_count,
            'at_risk_percentage': at_risk_percentage,
            'average_risk_score': average_risk_score,
            'top_risk_factors': top_risk_factors,
            'risk_distribution': risk_distribution,
            'highest_risk_students': highest_risk_students,
            'column_summary': column_summary,
            'full_results': df.fillna('').to_dict('records')
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
