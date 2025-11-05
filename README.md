🚗 Indian Vehicle Price Predictor (IVPP) – Full Stack ML Web Application

This project is a Full Stack Machine Learning Web Application that predicts the fair market price of Indian cars and bikes based on features such as brand, age, kilometers driven, fuel type, seller type, transmission, and owner type. It provides real-time predictions along with confidence range, feature importance visualization, and future depreciation forecasting.

🌐 Live Project Links

Frontend (Streamlit UI): https://full-stack-car-price-prediction-model-6v7sr9gkzqfg2rps2mfqug.streamlit.app/

Backend (FastAPI API): https://full-stack-car-price-prediction-model-1.onrender.com

API Swagger Docs: https://full-stack-car-price-prediction-model-1.onrender.com/docs

GitHub Repo: https://github.com/dishambha/Full-Stack-Car-Price-prediction-model

🧱 Tech Stack

Frontend: Streamlit
Backend: FastAPI (Python)
Machine Learning: Scikit-Learn, Pandas, NumPy
Deployment: Render (Backend), Streamlit Cloud (Frontend)
Version Control: Git & GitHub

✨ Features

Predicts both Car and Bike prices

Displays:

Predicted Mean Price

95% Confidence Price Range

Feature Importance Chart

Depreciation Forecast Curve

Clean and interactive UI built using Streamlit

Fully deployed and publicly accessible

📁 Project Structure

backend/ → FastAPI backend + ML model
frontend/ → Streamlit UI
requirements.txt → Dependencies
README.md → Project documentation

🚀 How to Run Locally

Step 1: Clone Repository
git clone https://github.com/dishambha/Full-Stack-Car-Price-prediction-model

cd Full-Stack-Car-Price-prediction-model

Step 2: Create & Activate Virtual Environment
python -m venv venv
./venv/Scripts/activate (Windows)

Step 3: Install Dependencies
pip install -r requirements.txt

Step 4: Run Backend
cd backend
uvicorn main:app --reload
Backend will run at: http://127.0.0.1:8000/docs

Step 5: Run Frontend
cd ../frontend
streamlit run app.py
Frontend will run at: http://localhost:8501

🧠 What I Learned

Deploying ML models with FastAPI

Connecting frontend and backend via REST API

Building UI interfaces with Streamlit

Deploying production apps on Render & Streamlit Cloud

Using GitHub for version control and collaboration

🪪 Author

Dishambha Awasthi
B.Tech – Computer Science & Engineering
Babu Banarasi Das University, Lucknow

GitHub: https://github.com/dishambha

LinkedIn: (optional — send if you want me to add)

⭐ Support

If you found this project helpful, please ⭐ Star the repository. It motivates me to build more projects!
