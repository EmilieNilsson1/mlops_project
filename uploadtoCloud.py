from google.cloud import storage
import os


def upload_directory_contents(
    bucket_name: str, source_directory: str, destination_folder: str
):
    # Initialize the storage client
    client = storage.Client("endless-galaxy-447815-e4")  # Project ID
    bucket = client.bucket(bucket_name)  # Get the target bucket

    # Walk through the local source directory
    for root, _, files in os.walk(source_directory):
        for file in files:
            file_path = os.path.join(root, file)  # Full local file path

            # Compute the relative path of the file within the source directory
            relative_path = os.path.relpath(file_path, source_directory)

            # Create the corresponding blob in the bucket, appending the destination folder to the path
            # The destination_folder specifies the desired folder structure in the bucket
            destination_path = os.path.join(destination_folder, relative_path).replace(
                "\\", "/"
            )

            # Create a blob in the desired folder and upload the file
            blob = bucket.blob(destination_path)
            blob.upload_from_filename(file_path)
            print(f"Upload for {file_path} completed to {destination_path}")


# Usage
upload_directory_contents(
    "mlops_project25_group72",
    "C:/Users/nunni/OneDrive/Skrivebord/MLOps/project/mlops_project/data/raw/cane",
    "data/raw/cane",
)
