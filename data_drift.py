from PIL import ImageStat
import numpy as np
import anyio
from pathlib import Path
import pandas as pd
from PIL import Image
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from google.cloud import storage
import io

# Global variables to hold baseline data and model metadata
baseline_data = None
BUCKET_NAME = "mlops_project25_group72"


def preprocess_image(image: Path) -> dict:
    """
    Extract meaningful features from an image for drift analysis.

    Features:
    - Mean pixel intensity (RGB channels).
    - Brightness (overall intensity).
    - Contrast (standard deviation of pixel intensities).
    """

    # Compute per-channel mean and standard deviation
    stat = ImageStat.Stat(image)
    mean_r, mean_g, mean_b = stat.mean
    std_r, std_g, std_b = stat.stddev

    # Overall brightness (average of RGB means)
    brightness = np.mean([mean_r, mean_g, mean_b])

    contrast = np.std(np.array(image))

    # Return extracted features as a dictionary
    return {
        "mean_r": mean_r,
        "mean_g": mean_g,
        "mean_b": mean_b,
        "std_r": std_r,
        "std_g": std_g,
        "std_b": std_b,
        "brightness": brightness,
        "contrast": contrast,
    }


def load_baseline_data() -> pd.DataFrame:
    """
    Load and preprocess baseline images from Cloud Storage, extracting features.
    """
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    # check if data drift csv exists
    baseline_file_name = "baseline_data.csv"
    baseline_blob = bucket.blob(baseline_file_name)

    if baseline_blob.exists():
        print(f"Loading baseline data from {baseline_file_name}")
        baseline_df = pd.read_csv(io.BytesIO(baseline_blob.download_as_bytes()))
    else:
        print("creating baseline csv")
        # Load labels CSV from Cloud Storage
        labels_blob = bucket.blob("data/p/translated_image_labels.csv")
        labels_data = labels_blob.download_as_string()
        labels_df = pd.read_csv(io.StringIO(labels_data.decode("utf-8")))

        # List to store features and labels
        features = []

        # Iterate over the rows in the labels DataFrame
        for i, row in labels_df.iterrows():
            image_name = row["image_name"]
            label = row["label"]

            # Fetch image from Cloud Storage
            image_blob = bucket.blob(f"data/p/images/{image_name}")
            if image_blob.exists():
                image_data = image_blob.download_as_bytes()
                image = Image.open(io.BytesIO(image_data)).convert("RGB")

                # Preprocess the image
                image_features = preprocess_image(image)
                image_features["label"] = label  # Add the label
                features.append(image_features)
            else:
                print(f"Warning: Image {image_name} not found in Cloud Storage")

        # Convert the list of features to a DataFrame
        baseline_df = pd.DataFrame(features)

        # save the csv to bucket
        csv_data = baseline_df.to_csv(index=False)
        blob = bucket.blob("baseline_data.csv")
        blob.upload_from_string(csv_data, content_type="text/csv")

    return baseline_df


def run_drift_analysis(reference_data: pd.DataFrame, current_data: pd.DataFrame):
    """
    Run drift analysis and save the report.
    """
    # Ensure only numeric features are passed
    numeric_columns = reference_data.select_dtypes(include=[np.number]).columns.tolist()
    report = Report(metrics=[DataDriftPreset()])  # Check drift for all numeric features
    report.run(
        reference_data=reference_data[numeric_columns],
        current_data=current_data[numeric_columns],
    )
    report.save_html("drift_report.html")


def load_latest_predictions(directory: Path, n: int) -> pd.DataFrame:
    """
    Load the N most recent prediction files and preprocess.
    """
    print(f"Loading {n} most recent predictions from {directory}")
    new_data = []
    client = storage.Client()
    bucket = client.get_bucket(BUCKET_NAME)

    folder_name = directory / "preds/"
    folder_name = "data/preds/"
    print(f"Loading predictions from {folder_name}")
    # blobs = bucket.blob(folder_name)

    # Download files
    blobs = bucket.list_blobs(prefix=folder_name)  # List all objects in the bucket

    for i, blob in enumerate(blobs):
        if blob.name.endswith((".jpg", ".jpeg", ".png")):
            print(blob.name)
            image_blob = bucket.blob(f"{blob.name}")
            image_data = image_blob.download_as_bytes()
            image = Image.open(io.BytesIO(image_data)).convert("RGB")

            process = preprocess_image(image)
            # process = preprocess_image(data_path / im)
            process["label"] = int(blob.name.split("/")[-1].split("_")[0])
            new_data.append(process)
            if i >= n:
                break

    return pd.DataFrame(new_data)


def lifespan(app: FastAPI):
    """Prepare baseline data and class names on app startup."""
    global baseline_data

    baseline_data = load_baseline_data()

    yield

    del baseline_data


# Initialize FastAPI app
app = FastAPI(lifespan=lifespan)


@app.get("/report")
async def get_report(n: int = 5):
    """Generate and return a data drift report."""
    # Load the latest predictions (assumes features and predictions are pre-extracted)
    dir = BUCKET_NAME + "/data"
    prediction_data = load_latest_predictions(Path(dir), n)

    print(baseline_data.head())
    print(prediction_data.head())

    # Run drift analysis
    run_drift_analysis(baseline_data, prediction_data)

    # Read the generated report
    async with await anyio.open_file("drift_report.html", encoding="utf-8") as f:
        html_content = await f.read()

    return HTMLResponse(content=html_content, status_code=200)


@app.get("/")
def read_root():
    return {"message": "Welcome to the Data Drift API"}
