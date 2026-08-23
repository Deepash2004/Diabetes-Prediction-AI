@echo off

call .venv\Scripts\activate

echo Starting Diabetes Risk Prediction App...

streamlit run app\app.py

pause
