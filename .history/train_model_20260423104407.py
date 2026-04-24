"""
KACHUA Model Training Module
Comprehensive student risk prediction model trainer
Author: Aalokita Chibb

This module generates synthetic student data, applies intelligent risk scoring logic,
and trains a RandomForest classifier to predict academic risk.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

def generate_synthetic_data(n_samples=5000, random_state=42):
    """
    Generate realistic synthetic student data for training.
    
    Features:
    - Age: 16-25 years
    - Gender: Male, Female, Other
    - AddressType: Urban, Rural
    - G1_Score: 0-100
    - PastFailures: 0-5
    - StudyTime: 0-10 hours/week
    - Absences: 0-30 per semester
    - GoOut: 1-5 frequency scale
    - Dalc: 1-5 daily alcohol scale
    - Walc: 1-5 weekend alcohol scale
    - FreeTime: 1-5 hours/day
    - SchoolSup: Yes/No
    - FamSup: Yes/No
    - Internet: Yes/No
    - HigherEd: Yes/No
    - Risk_Status: 0=Safe, 1=At Risk
    """
    
    np.random.seed(random_state)
    
    data = {
        'Age': np.random.randint(16, 26, n_samples),
        'Gender': np.random.choice(['Male', 'Female', 'Other'], n_samples),
        'AddressType': np.random.choice(['Urban', 'Rural'], n_samples),
        'G1_Score': np.random.randint(20, 101, n_samples),
        'PastFailures': np.random.randint(0, 6, n_samples),
        'StudyTime': np.random.randint(0, 11, n_samples),
        'Absences': np.random.randint(0, 31, n_samples),
        'GoOut': np.random.randint(1, 6, n_samples),
        'Dalc': np.random.randint(1, 6, n_samples),
        'Walc': np.random.randint(1, 6, n_samples),
        'FreeTime': np.random.randint(1, 6, n_samples),
        'SchoolSup': np.random.choice(['Yes', 'No'], n_samples),
        'FamSup': np.random.choice(['Yes', 'No'], n_samples),
        'Internet': np.random.choice(['Yes', 'No'], n_samples),
        'HigherEd': np.random.choice(['Yes', 'No'], n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Apply sophisticated risk scoring logic
    risk_scores = np.zeros(n_samples)
    
    # High absence + low grades = strong risk indicator
    high_absence_mask = df['Absences'] > 15
    low_grade_mask = df['G1_Score'] < 50
    risk_scores[high_absence_mask & low_grade_mask] += 0.6
    
    # Past failures compound risk
    risk_scores[df['PastFailures'] > 2] += 0.4
    
    # Low study time + high social life + no support = risk
    low_study_mask = df['StudyTime'] < 2
    high_goout_mask = df['GoOut'] > 4
    no_famsup_mask = df['FamSup'] == 'No'
    no_schoolsup_mask = df['SchoolSup'] == 'No'
    
    risk_scores[low_study_mask & high_goout_mask & no_famsup_mask & no_schoolsup_mask] += 0.5
    
    # Substance use patterns indicate risk
    high_dalc_mask = df['Dalc'] > 3
    high_walc_mask = df['Walc'] > 3
    risk_scores[high_dalc_mask | high_walc_mask] += 0.3
    
    # Poor attendance (any level) + low study time
    risk_scores[(df['Absences'] > 8) & (df['StudyTime'] < 3)] += 0.25
    
    # Protective factors reduce risk
    has_support_mask = (df['FamSup'] == 'Yes') | (df['SchoolSup'] == 'Yes')
    risk_scores[has_support_mask] = np.maximum(0, risk_scores[has_support_mask] - 0.2)
    
    # Higher education goals + internet access protect
    higher_ed_mask = df['HigherEd'] == 'Yes'
    internet_mask = df['Internet'] == 'Yes'
    risk_scores[higher_ed_mask & internet_mask] = np.maximum(0, risk_scores[higher_ed_mask & internet_mask] - 0.15)
    
    # Good grades protect significantly
    good_grade_mask = df['G1_Score'] > 70
    risk_scores[good_grade_mask] = np.maximum(0, risk_scores[good_grade_mask] - 0.3)
    
    # Normalize scores to 0-1 range for better model training
    risk_scores = np.clip(risk_scores, 0, 1)
    
    # Convert to binary classification: At Risk (1) if score > 0.4
    df['Risk_Status'] = (risk_scores > 0.4).astype(int)
    
    return df

def prepare_data(df):
    """Prepare data for model training: encode categorical variables."""
    
    categorical_cols = ['Gender', 'AddressType', 'SchoolSup', 'FamSup', 'Internet', 'HigherEd']
    encoders = {}
    
    df_encoded = df.copy()
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col])
        encoders[col] = le
    
    return df_encoded, encoders

def train_model(df, encoders):
    """Train RandomForest classifier with balanced class weights."""
    
    # Separate features and target
    X = df.drop('Risk_Status', axis=1)
    y = df['Risk_Status']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Initialize and train RandomForest
    model = RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        random_state=42,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Generate predictions for reporting
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    return model, X_train, X_test, y_train, y_test, y_pred, y_pred_proba

def save_artifacts(model, encoders, models_dir='models'):
    """Save trained model and encoders to disk."""
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    # Save model
    model_path = os.path.join(models_dir, 'rf_model.pkl')
    joblib.dump(model, model_path)
    print(f"✓ Model saved: {model_path}")
    
    # Save encoders
    for col, encoder in encoders.items():
        encoder_path = os.path.join(models_dir, f'encoder_{col}.pkl')
        joblib.dump(encoder, encoder_path)
        print(f"✓ Encoder saved: {encoder_path}")

def print_feature_importance(model, feature_names):
    """Print feature importance rankings."""
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE RANKINGS")
    print("="*60)
    
    for i, idx in enumerate(indices):
        print(f"{i+1:2d}. {feature_names[idx]:20s} → {importances[idx]:7.4f}")
    
    print("="*60)

def main():
    """Main training pipeline."""
    
    print("\n" + "="*70)
    print("KACHUA MODEL TRAINING PIPELINE")
    print("Building Academic Risk Prediction System")
    print("="*70)
    
    # Step 1: Generate synthetic data
    print("\n[1/5] Generating synthetic student data...")
    df = generate_synthetic_data(n_samples=5000, random_state=42)
    print(f"✓ Generated {len(df)} student records")
    print(f"  Features: {len(df.columns)-1}")
    print(f"  Risk distribution: {df['Risk_Status'].value_counts().to_dict()}")
    
    # Step 2: Prepare data
    print("\n[2/5] Preparing data (encoding categorical variables)...")
    df_encoded, encoders = prepare_data(df)
    print(f"✓ Encoded {len(encoders)} categorical features")
    print(f"  Encoded features: {list(encoders.keys())}")
    
    # Step 3: Train model
    print("\n[3/5] Training RandomForest model (200 estimators)...")
    model, X_train, X_test, y_train, y_test, y_pred, y_pred_proba = train_model(df_encoded, encoders)
    print(f"✓ Model trained on {len(X_train)} samples")
    print(f"  Training accuracy: {model.score(X_train, y_train):.4f}")
    print(f"  Testing accuracy: {model.score(X_test, y_test):.4f}")
    
    # Step 4: Print detailed metrics
    print("\n[4/5] Generating detailed classification report...")
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_test, y_pred, target_names=['Safe', 'At Risk']))
    
    # Print confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(f"  True Negatives:  {cm[0,0]:6d}  |  False Positives: {cm[0,1]:6d}")
    print(f"  False Negatives: {cm[1,0]:6d}  |  True Positives:  {cm[1,1]:6d}")
    print("="*60)
    
    # Print feature importance
    feature_names = list(df_encoded.drop('Risk_Status', axis=1).columns)
    print_feature_importance(model, feature_names)
    
    # Step 5: Save artifacts
    print("\n[5/5] Saving model and encoders...")
    save_artifacts(model, encoders)
    
    # Final summary
    print("\n" + "="*70)
    print("✓ KACHUA MODEL TRAINING COMPLETE")
    print("="*70)
    print(f"Model Status: Ready for production")
    print(f"Total Accuracy: {model.score(X_test, y_test):.2%}")
    print(f"Risk Prediction Accuracy: {cm[1,1]/(cm[1,1]+cm[1,0]):.2%}" if (cm[1,1]+cm[1,0]) > 0 else "N/A")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
