import pickle
import numpy as np
import pandas as pd
import gradio as gr

FEATURES = ["radius", "texture", "perimeter", "area"]
CLASS_LABELS = {1: "Normal", -1: "Anomaly"}


# 1. Load trained Isolation Forest pipeline
with open("iforest_model.pkl", "rb") as f:
    iforest_pipeline = pickle.load(f)


def _predict_iforest(X: np.ndarray):
    """
    Helper: returns predictions (+1 normal / -1 anomaly) and scores.
    """
    preds = iforest_pipeline.predict(X)          # +1 or -1
    scores = iforest_pipeline.decision_function(X)
    return preds, scores


# 2A. Single-row prediction
def predict_single(radius, texture, perimeter, area):
    X = np.array([[radius, texture, perimeter, area]])
    preds, scores = _predict_iforest(X)
    pred = int(preds[0])
    score = float(scores[0])

    return {
        "Prediction": CLASS_LABELS[pred],
        "Raw label": int(pred),  # +1 or -1
        "Anomaly score (higher = more normal)": score,
    }


# 2B. Batch prediction from uploaded CSV
def predict_batch(file):
    if file is None:
        return pd.DataFrame({"error": ["Please upload a CSV file."]})

    try:
        df = pd.read_csv(file.name)
    except Exception as e:
        return pd.DataFrame({"error": [f"Could not read CSV: {e}"]})

    # Ensure required columns
    missing = [col for col in FEATURES if col not in df.columns]
    if missing:
        return pd.DataFrame({
            "error": [f"Missing columns in CSV: {', '.join(missing)}"],
            "required_columns": [", ".join(FEATURES)],
        })

    X = df[FEATURES].astype(float).to_numpy()
    preds, scores = _predict_iforest(X)
    labels = [CLASS_LABELS[int(p)] for p in preds]

    result = df.copy()
    result["prediction"] = labels
    result["raw_label"] = preds
    result["anomaly_score"] = scores

    # Move prediction columns near front
    cols = ["prediction", "raw_label", "anomaly_score"] + \
        [c for c in result.columns if c not in ["prediction", "raw_label", "anomaly_score"]]
    result = result[cols]

    return result


def predict_batch_and_save(file):
    """
    Same as predict_batch, but returns a CSV path for DownloadButton.
    """
    result = predict_batch(file)
    if not isinstance(result, pd.DataFrame):
        result = pd.DataFrame({"error": ["Unknown error in prediction."]})

    csv_path = "iforest_batch_predictions.csv"
    result.to_csv(csv_path, index=False)
    return csv_path


# 3. Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# Isolation Forest Anomaly Detection")
    gr.Markdown(
        "Trained on 4 features from the breast cancer dataset. "
        "Use single prediction for one record or CSV upload for batch scoring.\n\n"
        f"**Required feature columns for CSV:** {', '.join(FEATURES)}"
    )

    # ---- Tab 1: Single prediction ----
    with gr.Tab("Single prediction"):
        with gr.Row():
            radius = gr.Number(label="radius", value=14.0)
            texture = gr.Number(label="texture", value=20.0)
        with gr.Row():
            perimeter = gr.Number(label="perimeter", value=90.0)
            area = gr.Number(label="area", value=600.0)

        btn_single = gr.Button("Predict")
        out_single = gr.JSON(label="Prediction details")

        btn_single.click(
            fn=predict_single,
            inputs=[radius, texture, perimeter, area],
            outputs=out_single,
        )

    # ---- Tab 2: Batch prediction ----
    with gr.Tab("Batch prediction (CSV upload)"):
        gr.Markdown(
            "Upload a CSV file with columns: "
            f"`{', '.join(FEATURES)}`.\n\n"
            "You will get anomaly/normal labels and scores for each row. "
            "You can also download the results as a CSV."
        )

        file_input = gr.File(label="Upload CSV file", file_types=[".csv"])
        btn_batch = gr.Button("Run batch prediction")

        out_batch = gr.Dataframe(
            label="Predictions (input + outputs)",
            interactive=False,
        )

        download_btn = gr.DownloadButton(
            label="Download results as CSV"
        )

        # Show table
        btn_batch.click(
            fn=predict_batch,
            inputs=[file_input],
            outputs=out_batch,
        )

        # Download CSV
        download_btn.click(
            fn=predict_batch_and_save,
            inputs=[file_input],
            outputs=download_btn,
        )


if __name__ == "__main__":
    demo.launch()
