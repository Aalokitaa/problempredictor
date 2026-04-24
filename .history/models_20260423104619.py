"""
KACHUA Database Models
SQLAlchemy ORM models for student profiles, teacher actions, and diagnostic data
"""

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json

db = SQLAlchemy()

class StudentProfile(db.Model):
    """
    Student academic and risk profile.
    Stores basic student info, risk scores, and bulk upload tracking.
    """
    __tablename__ = 'student_profiles'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    uid = db.Column(db.String(50), unique=True, nullable=False)
    contact = db.Column(db.String(120))
    risk_score = db.Column(db.Float, default=0.0)  # 0-1 normalized
    prediction_label = db.Column(db.String(20), default='SAFE')  # SAFE or AT RISK
    bulk_upload = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    teacher_actions = db.relationship('TeacherAction', backref='student', lazy=True, cascade='all, delete-orphan')
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'uid': self.uid,
            'contact': self.contact,
            'risk_score': round(self.risk_score, 4),
            'prediction_label': self.prediction_label,
            'bulk_upload': self.bulk_upload,
            'created_at': self.created_at.isoformat()
        }

class TeacherAction(db.Model):
    """
    Teacher intervention notes and actions.
    Tracks individual remarks, group interventions, and action dates.
    """
    __tablename__ = 'teacher_actions'
    
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('student_profiles.id'), nullable=True)
    remark_text = db.Column(db.Text, nullable=False)
    is_bulk = db.Column(db.Boolean, default=False)
    action_date = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'student_id': self.student_id,
            'remark_text': self.remark_text,
            'is_bulk': self.is_bulk,
            'action_date': self.action_date.isoformat()
        }

class WellbeingResponse(db.Model):
    """
    Comprehensive wellbeing assessment responses.
    Stores all 83 wellbeing questions, diagnostic flags, and recommendations.
    """
    __tablename__ = 'wellbeing_responses'
    
    id = db.Column(db.Integer, primary_key=True)
    student_name = db.Column(db.String(120), nullable=False)
    student_uid = db.Column(db.String(50), nullable=False)
    responses = db.Column(db.Text)  # JSON string of all Q13-Q83 answers
    diagnostic_flags = db.Column(db.Text)  # JSON string of flagged conditions with confidence
    recommendations = db.Column(db.Text)  # JSON string of full recommendation text per condition
    share_with_teacher = db.Column(db.Boolean, default=False)
    submitted_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def set_responses(self, responses_dict):
        """Store responses as JSON string."""
        self.responses = json.dumps(responses_dict)
    
    def get_responses(self):
        """Retrieve responses from JSON string."""
        return json.loads(self.responses) if self.responses else {}
    
    def set_diagnostic_flags(self, flags_dict):
        """Store diagnostic flags as JSON string."""
        self.diagnostic_flags = json.dumps(flags_dict)
    
    def get_diagnostic_flags(self):
        """Retrieve diagnostic flags from JSON string."""
        return json.loads(self.diagnostic_flags) if self.diagnostic_flags else {}
    
    def set_recommendations(self, recs_dict):
        """Store recommendations as JSON string."""
        self.recommendations = json.dumps(recs_dict)
    
    def get_recommendations(self):
        """Retrieve recommendations from JSON string."""
        return json.loads(self.recommendations) if self.recommendations else {}
    
    def to_dict(self):
        return {
            'id': self.id,
            'student_name': self.student_name,
            'student_uid': self.student_uid,
            'responses': self.get_responses(),
            'diagnostic_flags': self.get_diagnostic_flags(),
            'recommendations': self.get_recommendations(),
            'share_with_teacher': self.share_with_teacher,
            'submitted_at': self.submitted_at.isoformat()
        }

class DiagnosticSession(db.Model):
    """
    Complete diagnostic session combining academic and wellbeing data.
    Stores the Gemini AI advice and combined risk assessment.
    """
    __tablename__ = 'diagnostic_sessions'
    
    id = db.Column(db.Integer, primary_key=True)
    session_token = db.Column(db.String(100), unique=True, nullable=False)
    academic_data = db.Column(db.Text)  # JSON string of academic assessment data
    wellbeing_data = db.Column(db.Text)  # JSON string of wellbeing data
    combined_risk_score = db.Column(db.Float, default=0.0)
    gemini_advice = db.Column(db.Text)  # Full Gemini response
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def set_academic_data(self, data_dict):
        """Store academic data as JSON string."""
        self.academic_data = json.dumps(data_dict)
    
    def get_academic_data(self):
        """Retrieve academic data from JSON string."""
        return json.loads(self.academic_data) if self.academic_data else {}
    
    def set_wellbeing_data(self, data_dict):
        """Store wellbeing data as JSON string."""
        self.wellbeing_data = json.dumps(data_dict)
    
    def get_wellbeing_data(self):
        """Retrieve wellbeing data from JSON string."""
        return json.loads(self.wellbeing_data) if self.wellbeing_data else {}
    
    def to_dict(self):
        return {
            'id': self.id,
            'session_token': self.session_token,
            'academic_data': self.get_academic_data(),
            'wellbeing_data': self.get_wellbeing_data(),
            'combined_risk_score': round(self.combined_risk_score, 4),
            'gemini_advice': self.gemini_advice,
            'created_at': self.created_at.isoformat()
        }
