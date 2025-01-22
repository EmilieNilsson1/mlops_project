from PIL import ImageStat
import numpy as np
import anyio
from pathlib import Path
from torchvision import transforms
import pandas as pd
from PIL import Image
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from google.cloud import storage
import io
import os

# Global variables to hold baseline data and model metadata
baseline_data = None

def preprocess_image(image_path: Path) -> dict:
    """
    Extract meaningful features from an image for drift analysis.

    Features:
    - Mean pixel intensity (RGB channels).
    - Brightness (overall intensity).
    - Contrast (standard deviation of pixel intensities).
    """
    image = Image.open(image_path).convert("RGB")  # Ensure image is RGB
    
    # Compute per-channel mean and standard deviation
    stat = ImageStat.Stat(image)
    mean_r, mean_g, mean_b = stat.mean
    std_r, std_g, std_b = stat.stddev
    
    # Overall brightness (average of RGB means)
    brightness = np.mean([mean_r, mean_g, mean_b])
    
    # Return extracted features as a dictionary
    return {
        "mean_r": mean_r,
        "mean_g": mean_g,
        "mean_b": mean_b,
        "std_r": std_r,
        "std_g": std_g,
        "std_b": std_b,
        "brightness": brightness,
    }

def load_baseline_data() -> pd.DataFrame:
    """
    Load and preprocess baseline images from Cloud Storage, extracting features.
    """

    labels_df = pd.read_csv("data/processed/translated_image_labels.csv")
    # List to store features and labels
    features = []
    print("Processing baseline data...")

    # Iterate over the rows in the labels DataFrame
    for i, row in labels_df.iterrows():
        image_name = row["image_name"]
        label = row["label"]

        # Fetch image from Cloud Storage
        #image= Image.open(f"data/processed/images/{image_name}").convert("RGB")

        # Preprocess the image
        image_features = preprocess_image(f"data/processed/images/{image_name}")
        image_features["label"] = label  # Add the label
        features.append(image_features)

        if i % 100 == 0:
            print(f"Processed {i} images")

    # Convert the list of features to a DataFrame
    baseline_df = pd.DataFrame(features)

    # save the baseline data to a csv file
    baseline_df.to_csv("baseline_data.csv", index=False)
    return baseline_df

def run_drift_analysis(reference_data: pd.DataFrame, current_data: pd.DataFrame):
    """
    Run drift analysis and save the report.
    """
    # Ensure only numeric features are passed
    numeric_columns = reference_data.select_dtypes(include=[np.number]).columns.tolist()
    report = Report(metrics=[DataDriftPreset()])  # Check drift for all numeric features
    report.run(reference_data=reference_data[numeric_columns], current_data=current_data[numeric_columns])
    report.save_html("drift_report.html")

def load_latest_predictions(n: int) -> pd.DataFrame:
    """
    Load the N most recent prediction files and preprocess.
    """
    new_data = []
    data_path = Path("data/predictions")
    # loop through the predictions directory and load the images
    for im in os.listdir(data_path):
        # image = Image.open(data_path / im)
        # image = image.convert("RGB")
        process = preprocess_image(data_path / im)
        process["label"] = int(im.split("_")[0])
        new_data.append(process)

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
    prediction_data = load_latest_predictions(n)
    
    # Run drift analysis
    run_drift_analysis(baseline_data, prediction_data)
    print(baseline_data.head())
    print(prediction_data.head())
    # # Read the generated report
    async with await anyio.open_file("drift_report.html", encoding="utf-8") as f:
        html_content = await f.read()

    return HTMLResponse(content=html_content, status_code=200)
