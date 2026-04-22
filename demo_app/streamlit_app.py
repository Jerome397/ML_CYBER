from __future__ import annotations

from typing import Any, Dict

import pandas as pd
import plotly.express as px
import requests
import streamlit as st


try:
    BACKEND_URL = st.secrets["BACKEND_URL"]
except Exception:
    BACKEND_URL = "http://127.0.0.1:8000"


def get_json(path: str) -> Dict[str, Any]:
    r = requests.get(f"{BACKEND_URL}{path}", timeout=20)
    r.raise_for_status()
    return r.json()


def post_json(path: str, payload: dict | None = None) -> Dict[str, Any]:
    r = requests.post(f"{BACKEND_URL}{path}", json=payload or {}, timeout=20)
    r.raise_for_status()
    return r.json()


def make_logs_df(items):
    if not items:
        return pd.DataFrame(columns=[
            "row_index",
            "actual",
            "predicted",
            "correct",
            "confidence",
            "features",
            "raw_record",
        ])

    rows = []
    for item in items:
        rows.append({
            "row_index": item.get("row_index"),
            "actual": item.get("actual"),
            "predicted": item.get("predicted"),
            "correct": item.get("correct"),
            "confidence": item.get("confidence"),
            "features": item.get("features"),
            "raw_record": item.get("raw_record"),
        })

    return pd.DataFrame(rows)


st.set_page_config(page_title="Cyber Demo", layout="wide")
st.title("Cybersecurity Streaming Demo")

page = st.sidebar.radio("Page", ["1. Model Comparison", "2. Manual Prediction Logs"])

tick_response = None

if st.sidebar.button("Process Next Line"):
    try:
        tick_response = post_json("/tick")
        st.sidebar.success("Processed one new row from each dataset.")
        if tick_response.get("errors"):
            st.sidebar.error("Tick errors:")
            for k, v in tick_response["errors"].items():
                st.sidebar.write(f"{k}: {v}")
    except Exception as e:
        st.sidebar.error(str(e))

if st.sidebar.button("Reset Logs"):
    try:
        post_json("/reset")
        st.sidebar.success("Logs reset.")
    except Exception as e:
        st.sidebar.error(str(e))

health = get_json("/health")
state = get_json("/state")

st.sidebar.markdown("### Backend status")
st.sidebar.write(health["status"])

if health.get("errors"):
    st.sidebar.warning("Some resources failed to load:")
    for err in health["errors"]:
        st.sidebar.write(f"- {err}")

if tick_response and tick_response.get("errors"):
    st.error("Prediction errors from backend:")
    st.json(tick_response["errors"])

if page.startswith("1"):
    st.subheader("Model Comparison")

    dashboard = get_json("/dashboard")

    dataset_titles = {
        "binary": "Binary Dataset",
        "multiclass": "7-Class Dataset",
        "class33": "33-Class Dataset",
    }

    for dataset_key, title in dataset_titles.items():
        st.markdown(f"### {title}")
        rows = dashboard.get(dataset_key, [])

        if not rows:
            st.info(f"No metrics found for {title}")
            continue

        df = pd.DataFrame(rows)

        metric_long = df.melt(
            id_vars=["model_name"],
            value_vars=["val_accuracy", "val_macro_f1", "test_accuracy", "test_macro_f1"],
            var_name="metric",
            value_name="value"
        ).dropna()

        if not metric_long.empty:
            fig = px.bar(
                metric_long,
                x="model_name",
                y="value",
                color="metric",
                barmode="group",
                title=f"{title} - Validation/Test Metrics"
            )
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(df, use_container_width=True)

    st.markdown("### Why a specific model was selected")
    st.write(
        "The comparison emphasizes macro-F1 because the datasets are imbalanced. "
        "Accuracy is also shown, but the final model choice is mainly justified by balanced performance across classes."
    )

else:
    st.subheader("Manual Prediction Logs")

    tabs = st.tabs([
        "Binary Model",
        "Multiclass Best Model",
        "Multiclass Neural Network",
        "33-Class Model",
    ])

    mapping = {
        "Binary Model": "binary",
        "Multiclass Best Model": "multiclass_best",
        "Multiclass Neural Network": "multiclass_nn",
        "33-Class Model": "class33",
    }

    for tab, title in zip(tabs, mapping.keys()):
        with tab:
            key = mapping[title]
            df = make_logs_df(state["logs"].get(key, []))

            st.markdown(f"### {title}")

            if df.empty:
                st.info("No predictions yet. Press 'Process Next Line'.")
                continue

            summary_cols = st.columns(3)
            summary_cols[0].metric("Rows logged", len(df))

            correctness = df["correct"].dropna()
            if correctness.empty:
                summary_cols[1].metric("Observed accuracy", "-")
            else:
                summary_cols[1].metric("Observed accuracy", f"{float(correctness.mean()):.2%}")

            conf = df["confidence"].dropna()
            if conf.empty:
                summary_cols[2].metric("Average confidence", "-")
            else:
                summary_cols[2].metric("Average confidence", f"{float(conf.mean()):.3f}")

            st.dataframe(
                df[["row_index", "actual", "predicted", "correct", "confidence"]],
                use_container_width=True,
                height=320
            )

            st.markdown("#### Latest transformed features")
            st.json(df.iloc[0]["features"])

            st.markdown("#### Latest raw record")
            st.json(df.iloc[0]["raw_record"])