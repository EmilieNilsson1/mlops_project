from google.cloud import storage
import os

def upload_directory_contents(bucket_name, source_directory):
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    for root, _, files in os.walk(source_directory):
        for file in files:
            file_path = os.path.join(root, file)
            blob = bucket.blob(os.path.relpath(file_path, source_directory))
            blob.upload_from_filename(file_path)
            print(f"Upload for {file_path} completed")

# Usage
upload_directory_contents("mlops_project25_group72", "C:/Users/Vikto/OneDrive/Dokumenter/Studie/5.Sem/02476_ML_ops/archive")