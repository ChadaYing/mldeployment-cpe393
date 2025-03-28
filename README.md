# Housing Price Prediction API (CPE393 Lab)

This project builds and deploys a machine learning model to predict house prices using a dataset called `Housing.csv`.  
The model is served using a Flask API and packaged inside a Docker container.

---

## Setup Steps

### Step 1: Train the Model

Run the training script to train a regression model and save it:

```bash
python train.py
```

This will create a file called `model.pkl` inside the `app/` folder.

---

### Step 2: Build Docker Image

Build the Docker image from your project folder:

```bash
docker build -t ml-model .
```

---

### Step 3: Run the Flask App in Docker

```bash
docker run -p 9000:9000 ml-model
```

Your API will be available at:  
[http://127.0.0.1:9000](http://127.0.0.1:9000)

---

## Sample API Request and Response

### Endpoint: `/predict`  
**Method:** `POST`  
**URL:** `http://127.0.0.1:9000/predict`

#### Request Body

```json
{
  "features": [
    [7420, 4, 2, 3, 1, 0, 1, 0, 1, 2, 1, 1, 0]
  ]
}
```

Each array contains 13 input features in this order:  
`[area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hotwaterheating, airconditioning, parking, prefarea, furnishingstatus_semi-furnished, furnishingstatus_unfurnished]`

---

#### Response Example

```json
[
  {
    "prediction": 9010628.2,
    "confidence": 0.0
  }
]
```

- `prediction` = predicted house price
- `confidence` = simulated confidence score (0 = low, 1 = high)

---

## Health Check

To make sure the API is running:

Go to:  
[http://127.0.0.1:9000/health](http://127.0.0.1:9000/health)

Expected response:

```json
{
  "status": "ok"
}
```

---

## Tech Stack

- Python
- scikit-learn
- Flask
- Docker
