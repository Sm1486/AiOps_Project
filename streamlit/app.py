import os
import plotly.graph_objects as go
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
MLFLOW_URL = os.getenv("MLFLOW_URL", "http://localhost:5000")
GRAFANA_URL = os.getenv("GRAFANA_URL", "http://localhost:3000")

st.set_page_config(page_title="CIFAR-10 Image Classifier", layout="wide")

with st.sidebar:
    st.header("Monitoring Links")
    st.markdown(f"[MLflow Experiment Tracking]({MLFLOW_URL})")
    st.markdown(f"[Grafana Dashboard]({GRAFANA_URL})")
    st.markdown("---")
    st.caption("Backend: " + BACKEND_URL)

st.title("CIFAR-10 Image Classifier")

CLASSES = ['Airplane','Automobile','Bird','Cat','Deer', 'Dog','Frog','Horse','Ship','Truck']

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(uploaded_file, caption=uploaded_file.name, use_container_width=True)

    with col2:
        st.subheader("Classification Result")
        with st.spinner("Classifying"):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/predict",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), "image/jpeg")},
                    timeout=10,
                )
            except requests.exceptions.ConnectionError:
                st.error("Could not reach the backend.")
                st.stop()
            except requests.exceptions.Timeout:
                st.error("Request timed out. The backend may be overloaded.")
                st.stop()

        if response.status_code == 200:
            result = response.json()
            predicted = result["predicted_class"]
            confidence = result["confidence"]
            latency_ms = result.get("latency_ms", "N/A")
            all_probs: dict = result["all_probabilities"]
            st.success(f"**Prediction: {predicted.upper()}**")
            c1, c2 = st.columns(2)
            c1.metric("Confidence", f"{confidence:.1%}")
            c2.metric("Latency", f"{latency_ms} ms")
            sorted_items = sorted(all_probs.items(), key=lambda kv: kv[1])
            fig = go.Figure(
                go.Bar(
                    x=[v for _, v in sorted_items],
                    y=[k for k, _ in sorted_items],
                    orientation="h",
                    marker_color=[
                        "#EF4444" if k == predicted else "#3B82F6"
                        for k, _ in sorted_items
                    ],
                    text=[f"{v:.1%}" for _, v in sorted_items],
                    textposition="outside",
                )
            )
            fig.update_layout(
                title="Class Probability Distribution",
                xaxis_title="Probability",
                yaxis_title="Class",
                xaxis=dict(range=[0, 1]),
                height=420,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

        elif response.status_code == 422:
            st.error("Invalid input: the backend could not process the uploaded file.")
        else:
            st.error("Classification failed, try again.")
