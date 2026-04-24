from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class StudentProfile(db.Model):
    __tablename__ = 'student_profiles'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    uid = db.Column(db.String(50), unique=True, nullable=False)
    contact = db.Column(db.String(120), nullable=True)
    risk_score = db.Column(db.Float, nullable=False)
    prediction_label = db.Column(db.String(20), nullable=False) # "AT RISK" or "SAFE"
    bulk_upload = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    teacher_actions = db.relationship('TeacherAction', backref='student', lazy=True)

class TeacherAction(db.Model):
    __tablename__ = 'teacher_actions'
    
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('student_profiles.id'), nullable=True)
    remark_text = db.Column(db.Text, nullable=False)
    is_bulk = db.Column(db.Boolean, default=False)
    action_date = db.Column(db.DateTime, default=datetime.utcnow)

class WellbeingResponse(db.Model):
    __tablename__ = 'wellbeing_responses'
    
    id = db.Column(db.Integer, primary_key=True)
    student_name = db.Column(db.String(120), nullable=False)
    student_uid = db.Column(db.String(50), nullable=False)
    responses = db.Column(db.Text, nullable=False) # JSON string of all answers
    diagnostic_flags = db.Column(db.Text, nullable=False) # JSON string of flagged conditions
    recommendations = db.Column(db.Text, nullable=False) # JSON string of full recommendation text
    share_with_teacher = db.Column(db.Boolean, default=False)
    submitted_at = db.Column(db.DateTime, default=datetime.utcnow)

class DiagnosticSession(db.Model):
    __tablename__ = 'diagnostic_sessions'
    
    id = db.Column(db.Integer, primary_key=True)
    session_token = db.Column(db.String(100), unique=True, nullable=False)
    academic_data = db.Column(db.Text, nullable=True) # JSON string
    wellbeing_data = db.Column(db.Text, nullable=True) # JSON string
    combined_risk_score = db.Column(db.Float, nullable=True)
    gemini_advice = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
