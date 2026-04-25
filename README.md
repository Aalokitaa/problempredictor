<div align="center">
  <img src="static/images/logo.png" alt="KACHUA Logo" width="150" height="150" onerror="this.style.display='none'">
  <h1>🐢 KACHUA - AI-Powered Student Wellbeing & Diagnostic Platform</h1>
  <p><em>Predicting academic risks and fostering student wellbeing with Machine Learning and Generative AI.</em></p>

  [![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
  [![Flask](https://img.shields.io/badge/Flask-2.x-lightgrey.svg)](https://flask.palletsprojects.com/)
  [![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange.svg)](https://scikit-learn.org/)
  [![Gemini AI](https://img.shields.io/badge/Gemini%20AI-Generative%20Insights-blueviolet.svg)](https://deepmind.google/technologies/gemini/)
  [![Vercel Ready](https://img.shields.io/badge/Vercel-Deployment%20Ready-black.svg)](https://vercel.com/)
</div>

<br>

## 🌟 Overview

**KACHUA** is a comprehensive educational technology platform designed to help educators identify students at risk of academic struggles and provide actionable, personalized support. By combining a predictive machine learning model (Random Forest) with Google's powerful Gemini AI, KACHUA offers deep insights into both academic performance and emotional wellbeing.

The platform features a modern, premium **Light Organic** design with minimalist frosted-glass UI elements, ensuring a delightful user experience for educators.

## ✨ Key Features

- **🎯 Predictive Academic Risk Analytics:** Uses a trained Random Forest model to predict if a student is "SAFE" or "AT RISK" based on 15 socio-demographic and academic factors.
- **🧠 Generative AI Advice:** Integrates Google's Gemini AI to generate highly personalized, empathetic, and actionable advice tailored to individual student profiles and wellbeing responses.
- **📊 Bulk Analysis Capabilities:** Allows educators to upload CSV/Excel files to instantly analyze entire classrooms, complete with risk distributions and factor summaries.
- **❤️ Wellbeing Diagnostics:** Interactive questionnaires designed to gauge student mental health, stress levels, and learning environments.
- **📱 Responsive, Premium UI:** A stunning frosted-glass aesthetic with interactive 3D elements, optimized for all devices.
- **☁️ Cloud-Ready:** Pre-configured for seamless serverless deployment on Vercel.

## 🛠️ Technology Stack

- **Backend:** Python, Flask, SQLAlchemy
- **Machine Learning:** Scikit-Learn, Pandas, Numpy, Joblib
- **AI Integration:** `google-generativeai` (Gemini API)
- **Frontend:** HTML5, CSS3 (Vanilla, Glassmorphism design), JavaScript
- **Database:** SQLite (local) / Ephemeral SQLite via `/tmp` (Vercel)

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Google Gemini API Key

### Local Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/kachua.git
   cd kachua
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   Copy the example environment file and fill in your keys:
   ```bash
   cp .env.example .env
   ```
   *Make sure to add your `GEMINI_API_KEY` and a secure `FLASK_SECRET_KEY` inside the `.env` file.*

5. **Train the Machine Learning Model:**
   Before running the app, you need to generate the model and encoders:
   ```bash
   python train_model.py
   ```
   *(This will create `.pkl` files in the `models/` directory)*

6. **Run the Application:**
   ```bash
   python app.py
   ```
   The app will be available at `http://localhost:5000`.

## 📂 Project Structure

```text
KACHUA/
├── app.py                 # Main Flask application entry point
├── train_model.py         # Script to train the Random Forest model
├── check_models.py        # Utility to verify model generation
├── gemini_utils.py        # Google Gemini AI API integration
├── models.py              # SQLAlchemy database models
├── requirements.txt       # Python dependencies
├── vercel.json            # Vercel deployment configuration
├── .env                   # Environment variables (API keys)
├── models/                # Directory containing trained .pkl model files
├── static/                # CSS, JS, and image assets
└── templates/             # HTML Jinja2 templates
```

## 🌐 Deployment (Vercel)

KACHUA is fully optimized for Vercel. 
1. Connect your GitHub repository to Vercel.
2. In the Vercel dashboard, add the required Environment Variables (`GEMINI_API_KEY`, `FLASK_SECRET_KEY`).
3. Deploy! The `vercel.json` and internal `/tmp` database routing handles the serverless environment seamlessly.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! 
Feel free to check [issues page](https://github.com/yourusername/kachua/issues).

## 📝 License

This project is licensed under the MIT License.
