from fastapi import FastAPI
import inference

app = FastAPI()

app.include_router(inference.router)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Image Classifier API"}