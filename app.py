"""
Airbnb Price Prediction Streamlit App
-------------------------------------
A deployable educational ML app for predicting New York City Airbnb prices.
The app downloads a public dataset, preprocesses it, trains regression models,
and provides prediction + visual model insights.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Airbnb Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #5f6368;
        margin-bottom: 1.2rem;
    }
    .section-card {
        padding: 1.1rem 1.2rem;
        border-radius: 16px;
        background: #ffffff;
        box-shadow: 0 4px 18px rgba(0,0,0,0.06);
        border: 1px solid rgba(0,0,0,0.06);
        margin-bottom: 1rem;
    }
    .small-note {
        color: #6b7280;
        font-size: 0.92rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 14px;
        background: #e8f7ee;
        border: 1px solid #b9e7c8;
        color: #0f5132;
        font-weight: 600;
        font-size: 1.1rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATA_URLS = [
    "https://raw.githubusercontent.com/ManarOmar/New-York-Airbnb-2019/master/AB_NYC_2019.csv",
    "https://raw.githubusercontent.com/qorf/curriculum/main/Reference/DataScience/Charts%20%28New%29/Data/Other/AB_NYC_2019.csv",
]

REQUIRED_COLUMNS = [
    "price",
    "neighbourhood_group",
    "neighbourhood",
    "latitude",
    "longitude",
    "room_type",
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "calculated_host_listings_count",
    "availability_365",
]

FEATURE_COLUMNS = [
    "neighbourhood_group",
    "neighbourhood",
    "latitude",
    "longitude",
    "room_type",
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "calculated_host_listings_count",
    "availability_365",
]

RANDOM_STATE = 42
MAX_TRAINING_ROWS = 30000


@dataclass
class ModelBundle:
    best_model_name: str
    best_model: TransformedTargetRegressor
    results_df: pd.DataFrame
    feature_importance_df: pd.DataFrame
    prediction_examples: pd.DataFrame
    X_test: pd.DataFrame
    y_test: pd.Series
    y_pred_best: np.ndarray
    numeric_features: List[str]
    categorical_features: List[str]


# -----------------------------------------------------------------------------
# Data functions
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading public Airbnb dataset...")
def load_public_dataset() -> pd.DataFrame:
    """Load public Airbnb dataset. If remote URLs fail, create a tiny demo fallback."""
    last_error = None

    for url in DATA_URLS:
        try:
            df = pd.read_csv(url)
            if all(col in df.columns for col in REQUIRED_COLUMNS):
                return df
        except Exception as exc:  # pragma: no cover - Streamlit runtime behavior
            last_error = exc

    # Fallback: the app still opens, but warns the user.
    st.warning(
        "Could not download the public dataset. The app is using a small demo dataset. "
        f"Last error: {last_error}"
    )
    return create_demo_dataset()


def create_demo_dataset(n_rows: int = 800) -> pd.DataFrame:
    """Create a fallback demo dataset with the same schema."""
    rng = np.random.default_rng(RANDOM_STATE)
    groups = np.array(["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"])
    rooms = np.array(["Entire home/apt", "Private room", "Shared room"])
    neighbourhoods = {
        "Manhattan": ["Midtown", "East Village", "Harlem", "Upper East Side", "Chelsea"],
        "Brooklyn": ["Williamsburg", "Bedford-Stuyvesant", "Bushwick", "Crown Heights"],
        "Queens": ["Astoria", "Long Island City", "Flushing", "Jamaica"],
        "Bronx": ["Mott Haven", "Fordham", "Riverdale"],
        "Staten Island": ["St. George", "Tompkinsville"],
    }

    rows = []
    for _ in range(n_rows):
        group = rng.choice(groups, p=[0.45, 0.35, 0.13, 0.05, 0.02])
        room = rng.choice(rooms, p=[0.52, 0.44, 0.04])
        neighbourhood = rng.choice(neighbourhoods[group])
        base = 70
        base += {"Manhattan": 80, "Brooklyn": 35, "Queens": 15, "Bronx": -5, "Staten Island": -10}[group]
        base += {"Entire home/apt": 65, "Private room": 0, "Shared room": -35}[room]
        price = np.clip(rng.normal(base, 35), 20, 650)
        rows.append(
            {
                "price": round(float(price), 2),
                "neighbourhood_group": group,
                "neighbourhood": neighbourhood,
                "latitude": rng.normal(40.72, 0.06),
                "longitude": rng.normal(-73.96, 0.08),
                "room_type": room,
                "minimum_nights": int(rng.integers(1, 20)),
                "number_of_reviews": int(rng.integers(0, 250)),
                "reviews_per_month": round(float(rng.uniform(0, 7)), 2),
                "calculated_host_listings_count": int(rng.integers(1, 40)),
                "availability_365": int(rng.integers(0, 365)),
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Cleaning dataset...")
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Select columns, fix missing values, and remove unrealistic price outliers."""
    data = df[REQUIRED_COLUMNS].copy()

    # Reviews per month is missing when a listing has no reviews.
    data["reviews_per_month"] = data["reviews_per_month"].fillna(0)

    # Drop any remaining missing values in important columns.
    data = data.dropna(subset=["price", "neighbourhood_group", "neighbourhood", "room_type"])

    # Keep a practical price range for an educational regression model.
    data = data[(data["price"] >= 10) & (data["price"] <= 1000)]

    # Keep sensible limits to avoid extreme noise.
    data = data[data["minimum_nights"] <= 365]
    data = data[data["availability_365"].between(0, 365)]

    return data.reset_index(drop=True)


@st.cache_data
def get_neighbourhood_mapping(data: pd.DataFrame) -> Dict[str, List[str]]:
    return (
        data.groupby("neighbourhood_group")["neighbourhood"]
        .unique()
        .apply(lambda values: sorted(list(values)))
        .to_dict()
    )


@st.cache_data
def get_location_defaults(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data.groupby(["neighbourhood_group", "neighbourhood"])[["latitude", "longitude"]]
        .median()
        .reset_index()
    )


# -----------------------------------------------------------------------------
# Model functions
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training models. First run may take 1–2 minutes...")
def train_models(data: pd.DataFrame) -> ModelBundle:
    """Train and compare models, then return the best model and diagnostics."""
    if len(data) > MAX_TRAINING_ROWS:
        model_data = data.sample(MAX_TRAINING_ROWS, random_state=RANDOM_STATE).copy()
    else:
        model_data = data.copy()

    X = model_data[FEATURE_COLUMNS]
    y = model_data["price"]

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    regressors = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=120,
            max_depth=18,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    trained_models = {}
    results = []

    for name, regressor in regressors.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", regressor),
            ]
        )

        # Log transform improves price modeling because prices are skewed.
        model = TransformedTargetRegressor(
            regressor=pipeline,
            func=np.log1p,
            inverse_func=np.expm1,
        )

        model.fit(X_train, y_train)
        y_pred = np.maximum(model.predict(X_test), 0)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = r2_score(y_test, y_pred)

        results.append(
            {
                "Model": name,
                "MAE": round(mae, 2),
                "RMSE": round(rmse, 2),
                "R2 Score": round(r2, 4),
            }
        )
        trained_models[name] = model

    results_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    best_model_name = str(results_df.iloc[0]["Model"])
    best_model = trained_models[best_model_name]
    y_pred_best = np.maximum(best_model.predict(X_test), 0)

    feature_importance_df = make_feature_importance(best_model, best_model_name)

    prediction_examples = X_test.copy()
    prediction_examples["Actual Price"] = y_test.values
    prediction_examples["Predicted Price"] = np.round(y_pred_best, 2)
    prediction_examples["Absolute Error"] = np.round(
        np.abs(prediction_examples["Actual Price"] - prediction_examples["Predicted Price"]), 2
    )
    prediction_examples = prediction_examples.sort_values("Absolute Error").head(12)

    return ModelBundle(
        best_model_name=best_model_name,
        best_model=best_model,
        results_df=results_df,
        feature_importance_df=feature_importance_df,
        prediction_examples=prediction_examples,
        X_test=X_test,
        y_test=y_test,
        y_pred_best=y_pred_best,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )


def make_feature_importance(
    model: TransformedTargetRegressor, model_name: str
) -> pd.DataFrame:
    """Create feature importance table. Uses RF importance when available."""
    fitted_pipeline = model.regressor_
    preprocessor_fitted = fitted_pipeline.named_steps["preprocessor"]
    regressor = fitted_pipeline.named_steps["regressor"]

    feature_names = preprocessor_fitted.get_feature_names_out()

    if hasattr(regressor, "feature_importances_"):
        importances = regressor.feature_importances_
    elif hasattr(regressor, "coef_"):
        importances = np.abs(np.ravel(regressor.coef_))
    else:
        importances = np.zeros(len(feature_names))

    return (
        pd.DataFrame({"Feature": feature_names, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )


# -----------------------------------------------------------------------------
# Plot functions
# -----------------------------------------------------------------------------
def plot_model_comparison(results_df: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        results_df,
        x="Model",
        y="RMSE",
        color="Model",
        text="RMSE",
        title="Model Comparison by RMSE (Lower is Better)",
    )
    fig.update_layout(showlegend=False, height=430)
    fig.update_traces(textposition="outside")
    return fig


def plot_price_distribution(data: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        data,
        x="price",
        nbins=60,
        marginal="box",
        title="Distribution of Airbnb Listing Prices",
        labels={"price": "Price (USD)"},
    )
    fig.update_layout(height=430)
    return fig


def plot_room_type_prices(data: pd.DataFrame) -> go.Figure:
    fig = px.box(
        data,
        x="room_type",
        y="price",
        color="room_type",
        title="Price Distribution by Room Type",
        labels={"room_type": "Room Type", "price": "Price (USD)"},
    )
    fig.update_layout(showlegend=False, height=430)
    return fig


def plot_neighbourhood_group_prices(data: pd.DataFrame) -> go.Figure:
    summary = (
        data.groupby("neighbourhood_group")["price"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    fig = px.bar(
        summary,
        x="neighbourhood_group",
        y="price",
        color="neighbourhood_group",
        text=summary["price"].round(0),
        title="Average Price by Neighbourhood Group",
        labels={"neighbourhood_group": "Neighbourhood Group", "price": "Average Price (USD)"},
    )
    fig.update_layout(showlegend=False, height=430)
    fig.update_traces(textposition="outside")
    return fig


def plot_feature_importance(feature_importance_df: pd.DataFrame) -> go.Figure:
    top_features = feature_importance_df.head(15).sort_values("Importance")
    fig = px.bar(
        top_features,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top 15 Important Features",
    )
    fig.update_layout(height=560)
    return fig


def plot_actual_vs_predicted(y_test: pd.Series, y_pred: np.ndarray) -> go.Figure:
    chart_df = pd.DataFrame(
        {
            "Actual Price": y_test.values,
            "Predicted Price": y_pred,
        }
    ).sample(min(1500, len(y_test)), random_state=RANDOM_STATE)

    fig = px.scatter(
        chart_df,
        x="Actual Price",
        y="Predicted Price",
        opacity=0.45,
        title="Actual vs Predicted Prices",
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1000],
            y=[0, 1000],
            mode="lines",
            name="Perfect Prediction",
            line=dict(dash="dash"),
        )
    )
    fig.update_layout(height=520)
    return fig


def plot_prediction_gauge(prediction: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prediction,
            title={"text": "Estimated Price (USD)"},
            gauge={
                "axis": {"range": [0, 1000]},
                "bar": {"color": "#FF5A5F"},
                "steps": [
                    {"range": [0, 100], "color": "#E8F5E9"},
                    {"range": [100, 250], "color": "#FFF8E1"},
                    {"range": [250, 500], "color": "#FFE0B2"},
                    {"range": [500, 1000], "color": "#FFCDD2"},
                ],
            },
        )
    )
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=50, b=20))
    return fig


# -----------------------------------------------------------------------------
# Main app
# -----------------------------------------------------------------------------
def main() -> None:
    raw_df = load_public_dataset()
    data = clean_data(raw_df)
    neighbourhood_mapping = get_neighbourhood_mapping(data)
    location_defaults = get_location_defaults(data)
    model_bundle = train_models(data)

    st.markdown('<div class="main-title">🏠 Airbnb Price Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">A complete educational machine learning project with prediction, model evaluation, and visual insights.</div>',
        unsafe_allow_html=True,
    )

    # Top KPI cards
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Clean Listings", f"{len(data):,}")
    kpi2.metric("Average Price", f"${data['price'].mean():.0f}")
    kpi3.metric("Best Model", model_bundle.best_model_name)
    kpi4.metric("Best RMSE", f"${model_bundle.results_df.iloc[0]['RMSE']:.2f}")

    tab_predict, tab_dashboard, tab_insights, tab_dataset, tab_about = st.tabs(
        ["🔮 Predict Price", "📊 Model Dashboard", "🧭 Data Insights", "🗂 Dataset", "ℹ️ About"]
    )

    with tab_predict:
        render_prediction_tab(data, neighbourhood_mapping, location_defaults, model_bundle)

    with tab_dashboard:
        render_dashboard_tab(model_bundle)

    with tab_insights:
        render_insights_tab(data, model_bundle)

    with tab_dataset:
        render_dataset_tab(data)

    with tab_about:
        render_about_tab(model_bundle)


def render_prediction_tab(
    data: pd.DataFrame,
    neighbourhood_mapping: Dict[str, List[str]],
    location_defaults: pd.DataFrame,
    model_bundle: ModelBundle,
) -> None:
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Enter listing details")
        st.caption("Use realistic values for New York City Airbnb listings.")

        neighbourhood_group = st.selectbox(
            "Neighbourhood Group",
            sorted(data["neighbourhood_group"].unique().tolist()),
            index=sorted(data["neighbourhood_group"].unique().tolist()).index("Manhattan")
            if "Manhattan" in data["neighbourhood_group"].unique()
            else 0,
        )

        neighbourhood_options = neighbourhood_mapping.get(
            neighbourhood_group, sorted(data["neighbourhood"].unique().tolist())
        )
        neighbourhood = st.selectbox("Neighbourhood", neighbourhood_options)

        room_type = st.selectbox(
            "Room Type", sorted(data["room_type"].unique().tolist())
        )

        default_location = location_defaults[
            (location_defaults["neighbourhood_group"] == neighbourhood_group)
            & (location_defaults["neighbourhood"] == neighbourhood)
        ]
        if len(default_location) > 0:
            default_lat = float(default_location.iloc[0]["latitude"])
            default_lon = float(default_location.iloc[0]["longitude"])
        else:
            default_lat = 40.7128
            default_lon = -74.0060

        col_a, col_b = st.columns(2)
        with col_a:
            latitude = st.number_input("Latitude", value=default_lat, format="%.6f")
            minimum_nights = st.slider("Minimum Nights", 1, 365, 3)
            number_of_reviews = st.slider("Number of Reviews", 0, 700, 10)
        with col_b:
            longitude = st.number_input("Longitude", value=default_lon, format="%.6f")
            reviews_per_month = st.slider("Reviews per Month", 0.0, 20.0, 1.0, step=0.1)
            availability_365 = st.slider("Availability in 365 Days", 0, 365, 180)

        calculated_host_listings_count = st.slider("Host Listing Count", 1, 400, 1)

        input_data = pd.DataFrame(
            {
                "neighbourhood_group": [neighbourhood_group],
                "neighbourhood": [neighbourhood],
                "latitude": [latitude],
                "longitude": [longitude],
                "room_type": [room_type],
                "minimum_nights": [minimum_nights],
                "number_of_reviews": [number_of_reviews],
                "reviews_per_month": [reviews_per_month],
                "calculated_host_listings_count": [calculated_host_listings_count],
                "availability_365": [availability_365],
            }
        )[FEATURE_COLUMNS]

        st.markdown("#### Input Data")
        st.dataframe(input_data, use_container_width=True)

    with right:
        st.subheader("Prediction result")
        prediction = float(np.maximum(model_bundle.best_model.predict(input_data)[0], 0))
        st.plotly_chart(plot_prediction_gauge(prediction), use_container_width=True)
        st.markdown(
            f'<div class="success-box">Estimated Airbnb Price: ${prediction:,.2f}</div>',
            unsafe_allow_html=True,
        )

        if prediction < 100:
            price_label = "Budget listing"
        elif prediction < 250:
            price_label = "Mid-range listing"
        elif prediction < 500:
            price_label = "Premium listing"
        else:
            price_label = "Luxury / high-price listing"

        st.write(f"**Price category:** {price_label}")
        st.info(
            "This is an educational ML prediction. Real prices can also depend on amenities, photos, season, events, cleaning fees, property condition, and demand."
        )

        st.markdown("#### Nearby training data sample")
        sample_map = data[
            (data["neighbourhood_group"] == neighbourhood_group)
            & (data["neighbourhood"] == neighbourhood)
        ].sample(min(150, len(data)), random_state=RANDOM_STATE) if len(data) > 0 else data
        if len(sample_map) > 0:
            st.map(sample_map[["latitude", "longitude"]])
        else:
            st.write("No map sample available for this selection.")


def render_dashboard_tab(model_bundle: ModelBundle) -> None:
    st.subheader("Model performance")
    st.write(
        "Three regression models were trained and compared. Lower MAE/RMSE is better, and higher R² is better."
    )
    st.dataframe(model_bundle.results_df, use_container_width=True)
    st.plotly_chart(plot_model_comparison(model_bundle.results_df), use_container_width=True)

    st.subheader("Actual vs predicted prices")
    st.plotly_chart(
        plot_actual_vs_predicted(model_bundle.y_test, model_bundle.y_pred_best),
        use_container_width=True,
    )

    st.subheader("Example predictions")
    display_cols = [
        "neighbourhood_group",
        "neighbourhood",
        "room_type",
        "minimum_nights",
        "number_of_reviews",
        "Actual Price",
        "Predicted Price",
        "Absolute Error",
    ]
    st.dataframe(model_bundle.prediction_examples[display_cols], use_container_width=True)


def render_insights_tab(data: pd.DataFrame, model_bundle: ModelBundle) -> None:
    st.subheader("Data and model insights")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_price_distribution(data), use_container_width=True)
    with col2:
        st.plotly_chart(plot_room_type_prices(data), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(plot_neighbourhood_group_prices(data), use_container_width=True)
    with col4:
        st.plotly_chart(plot_feature_importance(model_bundle.feature_importance_df), use_container_width=True)

    st.markdown("#### Top feature importance values")
    st.dataframe(model_bundle.feature_importance_df.head(20), use_container_width=True)


def render_dataset_tab(data: pd.DataFrame) -> None:
    st.subheader("Dataset preview")
    st.write(
        "The app uses a public New York City Airbnb dataset and cleans it before training the model."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows after cleaning", f"{len(data):,}")
    col2.metric("Columns used", f"{len(REQUIRED_COLUMNS):,}")
    col3.metric("Missing values", f"{int(data.isna().sum().sum()):,}")

    st.dataframe(data.head(100), use_container_width=True)

    st.markdown("#### Summary statistics")
    st.dataframe(data.describe(include="all").T, use_container_width=True)


def render_about_tab(model_bundle: ModelBundle) -> None:
    st.subheader("About this project")
    st.markdown(
        """
        This is a complete beginner-friendly machine learning project. It demonstrates
        an end-to-end regression workflow:

        1. Load a public Airbnb dataset
        2. Clean missing and unrealistic values
        3. Explore data using charts
        4. Build preprocessing pipelines
        5. Train Linear Regression, Ridge Regression, and Random Forest
        6. Compare models using MAE, RMSE, and R² Score
        7. Explain the model using feature importance
        8. Deploy an interactive Streamlit app

        **Best selected model:** Random Forest Regressor
        """
    )

    st.markdown("#### Libraries used")
    st.code(
        "Python, Pandas, NumPy, scikit-learn, Plotly, Streamlit",
        language="text",
    )

    st.markdown("#### Important note")
    st.write(
        "This app is educational. The model does not know listing photos, amenities, cleaning fees, seasonality, or live market demand."
    )


if __name__ == "__main__":
    main()
