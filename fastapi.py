from fastapi import FastAPI, Request, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from src.RiceImgClassification.pipeline.prediction import PredictionPipeline
import os
import shutil

# Initialize FastAPI app
app = FastAPI()

# Mount static files (CSS and uploaded images)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Define a function to handle image prediction
def predict_image(filename):
    pipeline = PredictionPipeline(filename)
    predicted_class = pipeline.predict()
    return predicted_class, f"/uploads/{filename}"

# Route for the homepage
@app.get("/")
async def homepage(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Route to handle image upload and prediction
@app.post("/predict/")
async def predict(request: Request, file: UploadFile = File(...)):
    # Save the uploaded file
    with open(f"uploads/{file.filename}", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Predict using the uploaded file
    predicted_class, img_path = predict_image(file.filename)

    # Render the result page with prediction
    return templates.TemplateResponse("result.html", {"request": request, "predicted_class": predicted_class, "image_path": img_path})
