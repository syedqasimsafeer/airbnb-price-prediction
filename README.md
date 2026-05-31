# Airbnb Price Prediction Using Machine Learning

A complete, deployable, beginner-friendly machine learning project that predicts Airbnb listing prices in New York City.

This repository includes:

- Public dataset loading
- Data cleaning
- Exploratory data analysis
- Regression model training
- Model comparison
- Feature importance analysis
- Interactive Streamlit web app
- GitHub + Streamlit Cloud deployment support

---

## Live App

After deployment, add your Streamlit link here:

```text
https://your-app-name.streamlit.app
```

---

## Project Type

This is a **regression** machine learning project.

The target variable is:

```text
price
```

---

## Dataset

The app uses a public New York City Airbnb dataset.

Main dataset URL used in the app:

```text
https://raw.githubusercontent.com/ManarOmar/New-York-Airbnb-2019/master/AB_NYC_2019.csv
```

The app also includes a backup URL and a small demo-data fallback so the interface can still open if the public dataset source is temporarily unavailable.

---

## Features Used

The model uses these features:

```text
neighbourhood_group
neighbourhood
latitude
longitude
room_type
minimum_nights
number_of_reviews
reviews_per_month
calculated_host_listings_count
availability_365
```

---

## Models Trained

The app trains and compares:

1. Linear Regression
2. Ridge Regression
3. Random Forest Regressor

The best model is selected using RMSE.

---

## Evaluation Metrics

The models are evaluated using:

- MAE
- RMSE
- R² Score

---

## App Features

The Streamlit app includes:

- Price prediction form
- Interactive prediction gauge
- Model comparison chart
- Actual vs predicted plot
- Price distribution chart
- Room type price comparison
- Neighbourhood group price comparison
- Feature importance chart
- Dataset preview
- Beginner-friendly project explanation

---

## Repository Structure

```text
airbnb-price-prediction/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── .streamlit/
│   └── config.toml
│
├── notebooks/
│   └── 01_airbnb_price_prediction.ipynb
│
├── data/
│   └── README.md
│
├── models/
│   └── README.md
│
└── artifacts/
    └── README.md
```

---

## How to Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

---

## How to Deploy on Streamlit Community Cloud

1. Create a public GitHub repository.
2. Upload all extracted project files to the repository.
3. Go to Streamlit Community Cloud.
4. Click **Create app** or **New app**.
5. Select your GitHub repository.
6. Set the main file path to:

```text
app.py
```

7. Click **Deploy**.

---

## Important GitHub Upload Note

Do **not** upload only the `.zip` file to GitHub.

First extract the zip file on your computer, then upload the extracted files and folders to your GitHub repository.

GitHub repository should show `app.py` directly on the main page. If GitHub only shows a `.zip` file, Streamlit cannot run the app.

---

## Educational Note

This project is for learning and portfolio demonstration. Real Airbnb prices also depend on features not available in this dataset, such as amenities, photos, cleaning fee, season, live demand, property condition, and host quality.
