# Kidney_Disease_Classif
data from : https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data/data

Heart Disease Prediction Project
📋 Project Overview

This project aims to predict heart disease risk using a machine learning model. The application is built using Python, with data exploration performed in a Jupyter notebook and a prediction interface created with Streamlit. The project is fully Dockerized for easy deployment.

The goal is to optimize the net benefit 

🚀 Features

    Data Exploration and Analysis: In-depth exploration of the dataset using a Jupyter notebook.
    Machine Learning Model: Trained models using scikit-learn, XGBoost, and LightGBM.
    Interactive Web App: Streamlit app for doctors to input patient data and get heart disease predictions.
    Docker Support: Easily deploy the application using Docker.

📁 Folder Structure

my_project/
├── app.py                       # Streamlit app code
├── Dockerfile                   # Dockerfile for containerization
├── requirements.txt             # Python dependencies
├── exploration.ipynb            # Jupyter notebook for EDA
├── best_model__binaryRandomForest.joblib  # Trained model file
├── standarded_data_featureseng.csv        # Preprocessed dataset
└── README.md                    # Project documentation (this file)

📊 Exploratory Data Analysis (EDA)

The EDA was performed in exploration.ipynb. Here are the key insights:

    Dataset Overview: The dataset contains patient information such as age, cholesterol level, blood pressure, etc.
    Correlation Analysis: Identified key features correlated with heart disease risk.
    Data Cleaning: Removed missing values and handled outliers.
    Feature Engineering: Created new features (e.g., chol_trestbps_ratio, age_sex_interaction) for improved model performance.
    Visualization: Plotted various graphs to understand the distribution and relationships between features.

🖥️ Streamlit App

The app.py file contains the code for the Streamlit web application. It allows users to input patient information and get a prediction on whether the patient is at risk of heart disease. The app handles input scaling using a RobustScaler to ensure compatibility with the trained model.
🐳 Dockerization

The project is Dockerized for easy deployment. The Dockerfile installs all dependencies and sets up the Streamlit app.
Dockerfile

# Utiliser l'image officielle Python
FROM python:3.11-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier le fichier requirements.txt et installer les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste du code de l'application
COPY . .

# Exposer le port 8501 pour Streamlit
EXPOSE 8501

# Commande pour lancer l'application Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

🛠️ Setup Instructions
Prerequisites

    Python 3.11 or higher
    Docker (if using Docker)
    Poetry (if using pyproject.toml for dependencies)

1. Local Setup

If you want to run the app locally without Docker:

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py

2. Docker Setup

To run the app using Docker:

# Build the Docker image
docker build -t streamlit-heart-disease-app .

# Run the Docker container
docker run -p 8501:8501 streamlit-heart-disease-app

Access the app at http://localhost:8501.
3. Using Docker Compose (Optional)

If you prefer using Docker Compose, create a docker-compose.yml file:

version: '3.8'

services:
  streamlit-app:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8501:8501"

Then, run:

docker-compose up

🧪 Example Input and Output

Example Input:

    Age: 55
    Sex: 1 (Male)
    Chest Pain Type (cp): -1.5
    Resting Blood Pressure (trestbps): 130
    Cholesterol (chol): 250
    Fasting Blood Sugar (fbs): 0
    Resting ECG (restecg): 1
    Max Heart Rate (thalach): 150
    Exercise Induced Angina (exang): 0
    ST Depression (oldpeak): 1.5

Example Output:

Prediction: Sick
Probability of being sick: 85.0%

🤝 Contributing

Contributions are welcome! Please fork the repository and create a pull request.
📝 License

This project is licensed under the MIT License - see the LICENSE file for details.