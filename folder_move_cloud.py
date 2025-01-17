from google.cloud import storage

def move_folder_contents(bucket_name, source_folder, destination_folder):
    client = storage.Client('endless-galaxy-447815-e4')
    bucket = client.bucket(bucket_name)

    # List all objects in the source folder
    blobs = bucket.list_blobs(prefix=source_folder)

    for blob in blobs:
        # Define new blob path in the destination folder
        new_blob_name = blob.name.replace(source_folder, destination_folder, 1)
        new_blob = bucket.rename_blob(blob, new_blob_name)
        print(f"Moved {blob.name} to {new_blob.name}")

# Example usage
bucket_name = "mlops_project25_group72"
source_folder = "data/processed/"  # Ensure this ends with a '/'
destination_folder = "data/p/images/"  # Ensure this ends with a '/'

move_folder_contents(bucket_name, source_folder, destination_folder)