from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "outputs" / "xgboost_model.pkl"
FEATURES_PATH = BASE_DIR / "outputs" / "feature_columns.pkl"
DATA_PATH = BASE_DIR / "data" / "processed" / "final_processed_data.csv"


@st.cache_resource
def load_model_artifacts():
    with MODEL_PATH.open("rb") as model_file:
        model = pickle.load(model_file)

    with FEATURES_PATH.open("rb") as features_file:
        feature_cols = pickle.load(features_file)

    return model, feature_cols


@st.cache_data
def load_dataset():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    if "dayofweek" not in df.columns:
        df["dayofweek"] = df["date"].dt.dayofweek
    return df


def format_compact_number(value):
    value = float(value)
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:,.0f}"


def render_theme():
    st.markdown(
        """
        <style>
        :root {
            --bg: #f5efe4;
            --panel: rgba(255, 250, 242, 0.86);
            --panel-strong: #fffaf2;
            --ink: #1f2d2a;
            --muted: #5f6d69;
            --teal: #0f766e;
            --teal-deep: #114b47;
            --sand: #e8d8bc;
            --clay: #d27d5f;
            --line: rgba(17, 75, 71, 0.14);
            --shadow: 0 24px 60px rgba(31, 45, 42, 0.12);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(210, 125, 95, 0.18), transparent 28%),
                radial-gradient(circle at top right, rgba(15, 118, 110, 0.20), transparent 24%),
                linear-gradient(180deg, #fbf6ee 0%, #f5efe4 52%, #efe3ce 100%);
            color: var(--ink);
        }

        .block-container {
            max-width: 1440px;
            padding-top: 2.2rem;
            padding-bottom: 2.6rem;
        }

        h1, h2, h3 {
            font-family: Georgia, "Times New Roman", serif;
            letter-spacing: -0.03em;
            color: var(--ink);
        }

        p, li, label, .stMarkdown, .stCaption, .stTextInput, .stDateInput, .stNumberInput, .stSelectbox {
            font-family: "Aptos", "Trebuchet MS", "Segoe UI", sans-serif;
        }

        .hero-shell {
            padding: 2.5rem 2.8rem;
            border-radius: 28px;
            background:
                linear-gradient(135deg, rgba(17, 75, 71, 0.96), rgba(15, 118, 110, 0.92)),
                linear-gradient(180deg, #114b47, #0f766e);
            color: #f9f4ec;
            box-shadow: var(--shadow);
            border: 1px solid rgba(255, 255, 255, 0.08);
            position: relative;
            overflow: hidden;
        }

        .hero-shell::after {
            content: "";
            position: absolute;
            inset: auto -10% -38% auto;
            width: 320px;
            height: 320px;
            background: rgba(232, 216, 188, 0.16);
            border-radius: 50%;
        }

        .eyebrow {
            display: inline-block;
            margin-bottom: 0.9rem;
            padding: 0.35rem 0.8rem;
            border-radius: 999px;
            background: rgba(255, 250, 242, 0.14);
            border: 1px solid rgba(255, 250, 242, 0.18);
            font-size: 0.82rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .hero-title {
            margin: 0;
            font-size: 3.2rem;
            line-height: 0.95;
        }

        .hero-copy {
            max-width: 860px;
            margin: 1.05rem 0 0;
            font-size: 1.04rem;
            line-height: 1.8;
            color: rgba(249, 244, 236, 0.86);
        }

        .card {
            padding: 1.45rem 1.4rem;
            border-radius: 22px;
            background: var(--panel);
            border: 1px solid var(--line);
            box-shadow: 0 16px 34px rgba(31, 45, 42, 0.06);
            backdrop-filter: blur(10px);
            min-height: 160px;
        }

        .card-label {
            font-size: 0.8rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--muted);
            margin-bottom: 0.65rem;
        }

        .card-value {
            font-family: Georgia, "Times New Roman", serif;
            font-size: 2.2rem;
            color: var(--ink);
            line-height: 1;
            margin-bottom: 0.65rem;
        }

        .card-note {
            font-size: 0.98rem;
            color: var(--muted);
            line-height: 1.6;
        }

        .section-shell {
            background: rgba(255, 250, 242, 0.72);
            border: 1px solid var(--line);
            border-radius: 26px;
            padding: 1.8rem;
            box-shadow: 0 16px 34px rgba(31, 45, 42, 0.06);
            height: 100%;
        }

        .section-tag {
            display: inline-block;
            padding: 0.25rem 0.65rem;
            border-radius: 999px;
            background: rgba(17, 75, 71, 0.08);
            color: var(--teal-deep);
            font-size: 0.8rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.75rem;
        }

        .section-title {
            margin: 0;
            font-size: 1.7rem;
        }

        .section-copy {
            margin: 0.8rem 0 0;
            color: var(--muted);
            line-height: 1.75;
        }

        div[data-testid="stForm"] {
            border: 0;
            padding: 0;
            background: transparent;
        }

        div[data-testid="stSelectbox"] > div,
        div[data-testid="stDateInput"] > div,
        div[data-testid="stNumberInput"] > div {
            background: rgba(255, 255, 255, 0.72);
        }

        div[data-testid="stSelectbox"] label,
        div[data-testid="stDateInput"] label,
        div[data-testid="stNumberInput"] label {
            color: var(--teal-deep);
            font-weight: 600;
            letter-spacing: 0.02em;
        }

        button[kind="formSubmit"] {
            border-radius: 16px;
            border: none;
            min-height: 3.2rem;
            background: linear-gradient(135deg, #d27d5f, #b35d42);
            color: #fff9f2;
            font-weight: 700;
            letter-spacing: 0.02em;
            box-shadow: 0 14px 28px rgba(179, 93, 66, 0.25);
        }

        button[kind="formSubmit"]:hover {
            background: linear-gradient(135deg, #c87054, #a55038);
        }

        .mini-stat {
            padding: 1.1rem 1.15rem;
            border-radius: 20px;
            background: rgba(255, 255, 255, 0.82);
            border: 1px solid rgba(17, 75, 71, 0.12);
            box-shadow: 0 14px 28px rgba(31, 45, 42, 0.05);
            min-height: 128px;
        }

        .mini-stat-label {
            font-size: 0.78rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--teal-deep);
            margin-bottom: 0.75rem;
            font-weight: 700;
        }

        .mini-stat-value {
            font-family: Georgia, "Times New Roman", serif;
            font-size: 2.15rem;
            color: var(--ink);
            line-height: 1.05;
            margin-bottom: 0.45rem;
        }

        .mini-stat-note {
            color: var(--muted);
            font-size: 0.95rem;
            line-height: 1.55;
        }

        .context-shell {
            background: rgba(255, 250, 242, 0.72);
            border: 1px solid var(--line);
            border-radius: 26px;
            padding: 1.8rem;
            box-shadow: 0 16px 34px rgba(31, 45, 42, 0.06);
        }

        .context-stats {
            margin: 1.2rem 0 1.25rem;
        }

        .result-shell {
            padding: 1.35rem 1.5rem;
            border-radius: 24px;
            background: linear-gradient(180deg, rgba(255, 250, 242, 0.95), rgba(232, 216, 188, 0.42));
            border: 1px solid rgba(17, 75, 71, 0.14);
            box-shadow: 0 16px 34px rgba(31, 45, 42, 0.08);
        }

        .result-label {
            font-size: 0.82rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--teal-deep);
            margin-bottom: 0.55rem;
        }

        .result-value {
            font-family: Georgia, "Times New Roman", serif;
            font-size: 2.8rem;
            line-height: 1;
            color: var(--ink);
            margin-bottom: 0.5rem;
        }

        .result-copy {
            color: var(--muted);
            line-height: 1.65;
            margin: 0;
        }

        .insight-list {
            margin: 0.85rem 0 0;
            padding-left: 1.1rem;
            color: var(--muted);
        }

        .insight-list li {
            margin-bottom: 0.45rem;
        }

        .chart-note {
            color: var(--muted);
            font-size: 0.92rem;
            margin-top: 0.8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_feature_row(df, store, family, prediction_date, onpromotion, transactions, feature_cols):
    history = df[(df["store_nbr"] == store) & (df["family"] == family)].copy()
    history = history[history["date"] < prediction_date].sort_values("date")

    if history.empty:
        raise ValueError("No historical data available for this store and family.")

    if prediction_date <= history["date"].max():
        raise ValueError("Prediction date must be after the latest available historical date.")

    new_row = history.iloc[-1:].copy()
    new_row["date"] = prediction_date
    new_row["sales"] = np.nan
    new_row["onpromotion"] = onpromotion
    new_row["transactions"] = transactions

    temp = pd.concat([history, new_row], ignore_index=True).sort_values("date")

    temp["lag_1"] = temp["sales"].shift(1)
    temp["lag_7"] = temp["sales"].shift(7)
    temp["lag_14"] = temp["sales"].shift(14)

    temp["rolling_mean_7"] = temp["sales"].shift(1).rolling(7).mean()
    temp["rolling_mean_14"] = temp["sales"].shift(1).rolling(14).mean()
    temp["rolling_median_7"] = temp["sales"].shift(1).rolling(7).median()
    temp["rolling_std_14"] = temp["sales"].shift(1).rolling(14).std()

    temp["momentum_7"] = temp["lag_1"] - temp["lag_7"]
    temp["promo_last_7"] = temp["onpromotion"].shift(1).rolling(7).sum()
    temp["promo_x_transactions"] = temp["onpromotion"] * temp["transactions"]

    temp["year"] = temp["date"].dt.year
    temp["month"] = temp["date"].dt.month
    temp["day"] = temp["date"].dt.day
    temp["dayofweek"] = temp["date"].dt.dayofweek
    temp["week_of_year"] = temp["date"].dt.isocalendar().week.astype(int)
    temp["day_of_year"] = temp["date"].dt.dayofyear
    temp["is_weekend"] = temp["dayofweek"].isin([5, 6]).astype(int)

    global_dow_avg = df.groupby("dayofweek")["sales"].mean()
    temp["dow_avg"] = temp["dayofweek"].map(global_dow_avg)

    if "store_avg" in feature_cols:
        store_avg = df.groupby("store_nbr")["sales"].mean()
        temp["store_avg"] = temp["store_nbr"].map(store_avg)

    final = temp.iloc[-1:].copy()
    final = pd.get_dummies(final)
    final = final.reindex(columns=feature_cols, fill_value=0)

    if final.isna().any().any():
        missing_cols = final.columns[final.isna().any()].tolist()
        raise ValueError(f"Missing values in features: {missing_cols}")

    return final


def get_selection_history(df, store, family):
    return df[(df["store_nbr"] == store) & (df["family"] == family)].sort_values("date")


st.set_page_config(
    page_title="Forecast Canvas",
    page_icon=":shopping_trolley:",
    layout="wide",
)

render_theme()
model, feature_cols = load_model_artifacts()
df = load_dataset()

default_store = int(sorted(df["store_nbr"].unique())[0])
default_family = sorted(df["family"].unique())[0]
default_date = (df["date"].max() + pd.Timedelta(days=1)).date()

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

st.markdown(
    f"""
    <div class="hero-shell">
        <div class="eyebrow">Retail Intelligence Studio</div>
        <h1 class="hero-title">Advanced Sales Forecasting</h1>
        <p class="hero-copy">
            Forecast daily sales across stores with your trained model, using fast what-if inputs for date,
            promotions, and expected transactions. Built on more than {format_compact_number(len(df))} historical rows.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

top_cards = st.columns(4)
card_specs = [
    ("Training Rows", format_compact_number(len(df)), "Historical dataset currently loaded for forecasting."),
    ("Store Network", format_compact_number(df["store_nbr"].nunique()), "Stores available for scenario testing."),
    ("Product Families", format_compact_number(df["family"].nunique()), "Families the model can score today."),
    ("Latest Actual Date", df["date"].max().strftime("%d %b %Y"), f"{len(feature_cols)} model features used at inference time."),
]

for column, (label, value, note) in zip(top_cards, card_specs):
    with column:
        st.markdown(
            f"""
            <div class="card">
                <div class="card-label">{label}</div>
                <div class="card-value">{value}</div>
                <div class="card-note">{note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

left_col, right_col = st.columns([1.0, 1.1], gap="large")

with left_col:
    st.markdown(
        """
        <div class="section-shell">
            <div class="section-tag">Forecast Setup</div>
            <h2 class="section-title">Build a new prediction</h2>
            <p class="section-copy">
                Choose the store, family, target date, and demand signals below.
                The app reconstructs the full feature row before inference and keeps the controls spaced out for quicker use.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("prediction_form"):
        input_left, input_right = st.columns(2)

        with input_left:
            store = st.selectbox(
                "Store Number",
                sorted(df["store_nbr"].unique()),
                index=0,
                key="store_widget",
                help="Select the store you want to forecast for.",
            )
            date_input = st.date_input(
                "Prediction Date",
                value=default_date,
                help="The date must be after the latest historical date for the selected pair.",
            )

        with input_right:
            family = st.selectbox(
                "Product Family",
                sorted(df["family"].unique()),
                index=0,
                key="family_widget",
                help="Select the product family to forecast.",
            )
            transactions = st.number_input(
                "Transactions",
                min_value=0,
                step=1,
                value=0,
                help="Expected transaction count for the target day.",
            )

        onpromotion = st.number_input(
            "Items On Promotion",
            min_value=0,
            step=1,
            value=0,
            help="How many items are expected to be on promotion on the target day.",
        )
        submitted = st.form_submit_button("Predict Sales", use_container_width=True)

with right_col:
    selected_store = st.session_state.get("store_widget", default_store)
    selected_family = st.session_state.get("family_widget", default_family)
    history = get_selection_history(df, selected_store, selected_family)

    st.markdown(
        """
        <div class="context-shell">
            <div class="section-tag">Context Panel</div>
            <h2 class="section-title">Current selection snapshot</h2>
            <p class="section-copy">
                Quick context for the selected store and family, so you can judge the forecast against recent behavior.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not history.empty:
        latest_sales = history["sales"].iloc[-1]
        trailing_mean = history["sales"].tail(30).mean()
        mean_delta = latest_sales - trailing_mean

        stat_a, stat_b, stat_c = st.columns(3)
        with stat_a:
            st.markdown(
                f"""
                <div class="mini-stat context-stats">
                    <div class="mini-stat-label">History Rows</div>
                    <div class="mini-stat-value">{format_compact_number(len(history))}</div>
                    <div class="mini-stat-note">Rows available for this store-family combination.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with stat_b:
            st.markdown(
                f"""
                <div class="mini-stat context-stats">
                    <div class="mini-stat-label">Latest Sales</div>
                    <div class="mini-stat-value">{latest_sales:,.2f}</div>
                    <div class="mini-stat-note">Most recent observed sales value in history.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with stat_c:
            st.markdown(
                f"""
                <div class="mini-stat context-stats">
                    <div class="mini-stat-label">Vs 30-Day Avg</div>
                    <div class="mini-stat-value">{mean_delta:+,.2f}</div>
                    <div class="mini-stat-note">Difference from the trailing 30-day average.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        chart_data = history.tail(60).set_index("date")[["sales"]].rename(columns={"sales": "Recent sales"})
        st.line_chart(chart_data, height=270)
        st.markdown(
            f"<p class='chart-note'>Showing the last 60 observations for Store {selected_store} and {selected_family.title()}.</p>",
            unsafe_allow_html=True,
        )
    else:
        st.info("No history is available yet for the current selection.")

if submitted:
    prediction_date = pd.to_datetime(date_input)

    try:
        with st.spinner("Generating forecast..."):
            final = build_feature_row(
                df=df,
                store=store,
                family=family,
                prediction_date=prediction_date,
                onpromotion=onpromotion,
                transactions=transactions,
                feature_cols=feature_cols,
            )
            prediction = max(0.0, float(model.predict(final)[0]))

        st.session_state.last_prediction = {
            "store": store,
            "family": family,
            "prediction_date": prediction_date.strftime("%d %b %Y"),
            "prediction": prediction,
            "transactions": transactions,
            "onpromotion": onpromotion,
        }
    except Exception as exc:
        st.session_state.last_prediction = {"error": str(exc)}

prediction_block = st.session_state.last_prediction

if prediction_block:
    st.markdown("")
    if "error" in prediction_block:
        st.error(f"Prediction failed: {prediction_block['error']}")
    else:
        result_col, notes_col = st.columns([0.95, 1.05], gap="large")

        with result_col:
            st.markdown(
                f"""
                <div class="result-shell">
                    <div class="result-label">Latest Forecast</div>
                    <div class="result-value">{prediction_block['prediction']:,.2f}</div>
                    <p class="result-copy">
                        Predicted sales for <strong>{prediction_block['family'].title()}</strong> at
                        <strong>Store {prediction_block['store']}</strong> on
                        <strong>{prediction_block['prediction_date']}</strong>.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with notes_col:
            st.markdown(
                f"""
                <div class="section-shell">
                    <div class="section-tag">Run Summary</div>
                    <h2 class="section-title">Scenario snapshot</h2>
                    <p class="section-copy">
                        The forecast below was generated from the engineered feature set already used by your trained model.
                        Inputs from this run are summarized here for quick review.
                    </p>
                    <ul class="insight-list">
                        <li>Target day: {prediction_block['prediction_date']}</li>
                        <li>Expected transactions: {prediction_block['transactions']:,}</li>
                        <li>Items on promotion: {prediction_block['onpromotion']:,}</li>
                        <li>Model feature width: {len(feature_cols)} columns after encoding and alignment</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
