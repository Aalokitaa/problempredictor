"""
KACHUA - Model Training Script
Author: Aalokita Chibb
Description: Generates simulated student data and trains a RandomForest model to predict academic risk.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

def generate_data(num_students=5000):
    np.random.seed(42)
    
    # Generate random features
    age = np.random.randint(16, 26, num_students)
    gender = np.random.choice(['Male', 'Female', 'Prefer not to say'], num_students, p=[0.48, 0.48, 0.04])
    address_type = np.random.choice(['Urban area', 'Rural area'], num_students, p=[0.7, 0.3])
    g1_score = np.random.randint(0, 101, num_students)
    past_failures = np.random.randint(0, 6, num_students)
    study_time = np.random.randint(1, 11, num_students)
    absences = np.random.randint(0, 31, num_students)
    go_out = np.random.randint(1, 6, num_students)
    dalc = np.random.randint(1, 6, num_students)
    walc = np.random.randint(1, 6, num_students)
    free_time = np.random.randint(1, 6, num_students)
    school_sup = np.random.choice(['Yes', 'No'], num_students, p=[0.3, 0.7])
    fam_sup = np.random.choice(['Yes', 'No'], num_students, p=[0.6, 0.4])
    internet = np.random.choice(['Yes', 'No'], num_students, p=[0.85, 0.15])
    higher_ed = np.random.choice(['Yes', 'No'], num_students, p=[0.9, 0.1])
    
    df = pd.DataFrame({
        'Age': age,
        'Gender': gender,
        'AddressType': address_type,
        'G1_Score': g1_score,
        'PastFailures': past_failures,
        'StudyTime': study_time,
        'Absences': absences,
        'GoOut': go_out,
        'Dalc': dalc,
        'Walc': walc,
        'FreeTime': free_time,
        'SchoolSup': school_sup,
        'FamSup': fam_sup,
        'Internet': internet,
        'HigherEd': higher_ed
    })
    
    # Determine Risk_Status
    # Base risk starts at 0
    risk_score = np.zeros(num_students)
    
    # High absences + low score + high alcohol -> strong risk
    cond1 = (df['Absences'] > 15) & (df['G1_Score'] < 50) & (df['Dalc'] > 3)
    risk_score[cond1] += 5
    
    # Low study time + high go out + no support -> risk
    cond2 = (df['StudyTime'] <= 3) & (df['GoOut'] >= 4) & (df['FamSup'] == 'No') & (df['SchoolSup'] == 'No')
    risk_score[cond2] += 3
    
    # Past failures
    risk_score += df['PastFailures'] * 1.5
    
    # Modifiers
    risk_score[df['HigherEd'] == 'Yes'] -= 1
    risk_score[df['Internet'] == 'Yes'] -= 0.5
    
    # Final label assignment: if risk_score > some threshold, at risk
    # Let's say threshold is 3
    df['Risk_Status'] = np.where(risk_score >= 3, 1, 0)
    
    # Add some noise to make it realistic
    noise_idx = np.random.choice(num_students, int(num_students * 0.05), replace=False)
    df.loc[noise_idx, 'Risk_Status'] = 1 - df.loc[noise_idx, 'Risk_Status']
    
    return df

def train():
    df = generate_data(5000)
    
    categorical_cols = ['Gender', 'AddressType', 'SchoolSup', 'FamSup', 'Internet', 'HigherEd']
    
    os.makedirs('models', exist_ok=True)
    
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        joblib.dump(le, f'models/encoder_{col}.pkl')
        
    X = df.drop('Risk_Status', axis=1)
    y = df['Risk_Status']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nFeature Importances (sorted descending):")
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    features = X.columns
    for i in range(X.shape[1]):
        print(f"{i + 1}. {features[indices[i]]}: {importances[indices[i]]:.4f}")
        
    joblib.dump(clf, 'models/rf_model.pkl')
    print("\n[SUCCESS] Model and encoders saved to models/ directory.")

if __name__ == "__main__":
    train()
