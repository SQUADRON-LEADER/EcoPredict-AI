@echo off
echo 🌱 GHG Emissions Predictor - Starting App...
echo.
echo Activating virtual environment...
cd /d "c:\Users\aayus\OneDrive\Desktop\GHG"

echo Starting Streamlit app...
echo App will open in your browser at: http://localhost:8501
echo.
echo Press Ctrl+C to stop the app
echo.
c:/Users/aayus/OneDrive/Desktop/GHG/.venv/Scripts/streamlit.exe run app.py

pause
