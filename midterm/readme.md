
# CTR Prediction: Click-Through Rate Prediction Project

This project predicts whether a user will click on an advertisement using demographic, session, and product data. It combines data preprocessing, advanced modeling, and a Dockerized deployment pipeline.
The dataset is taken from here: https://www.kaggle.com/datasets/arashnic/ctr-in-advertisement/data

---

## Features
- **Data Preprocessing:** Handles missing values, time-based feature extraction, and encoding.
- **Model Training:** Evaluates Logistic Regression, Decision Trees, Random Forest, and XGBoost.
- **Hyperparameter Tuning:** Optimizes XGBoost for the best F1 score.
- **Model Deployment:** Includes Dockerized service and a Python prediction client.

---

## Usage

### 1. Train the Model
Use `ad-click-prediction.py` to train the model with the best hyperparameters:
```bash
python ad-click-prediction.py
```
This script saves the model and preprocessors as:
- `dict_vectorizer.pkl`
- `standard_scaler.pkl`
- `xgb_model.pkl`

### 2. Test the Model
Use `test-predict.py` to load the saved models and make prediction for the single data point in the script:
```bash
python test-predict.py
```

### 3. Run the Dockerized Service

#### Build the Docker Image
```bash
docker build -t adpred .
```

#### Run the Docker Container
```bash
docker run -d -p 9696:9696 adpred
```

#### Use the Client
Run the client to send prediction requests to the service:
```bash
python predict_client.py
```

Check [Documentation](Documentation/documentation.pdf) for more information.