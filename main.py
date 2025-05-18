import json, os
import matplotlib.pyplot as plt
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.config import Configuration
from app.forms.classification_form import ClassificationForm
from app.ml.classification_utils import classify_image
from app.utils import list_images
#Feature 2:
from PIL import Image, ImageEnhance
from fastapi.responses import Response
from io import BytesIO
from pydantic import BaseModel
from fastapi import HTTPException


app = FastAPI()
config = Configuration() 

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")


@app.get("/info")
def info() -> dict[str, list[str]]:
    """Returns a dictionary with the list of models and
    the list of available image files."""
    list_of_images = list_images()
    list_of_models = Configuration.models
    data = {"models": list_of_models, "images": list_of_images}
    return data


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """The home page of the service with histogram functionality."""
    return templates.TemplateResponse(
        "home.html",
        {"request": request, "images": list_images()}
    )


@app.get("/classifications")
def create_classify(request: Request):
    """Renders the classification selection page."""
    return templates.TemplateResponse(
        "classification_select.html",
        {"request": request, "images": list_images(), "models": Configuration.models},
    )


@app.post("/classifications")
async def request_classification(request: Request):
    """Handles the classification request."""
    form = ClassificationForm(request)
    await form.load_data()
    image_id = form.image_id
    model_id = form.model_id
    classification_scores = classify_image(model_id=model_id, img_id=image_id)
    return templates.TemplateResponse(
        "classification_output.html",
        {
            "request": request,
            "image_id": image_id,
            "classification_scores": json.dumps(classification_scores),
        },
    )

  
#create gets for the download of the JSON results and the GRAPH
# Issue 3:
@app.get("/download_results")
def download_results(image_id : str, classification_scores : str):
    """Download the results (classification
     scores) as a JSON file"""
    file_name = f"classification_result_{image_id}.json"
    file_path = "downloads/" + file_name
    classification_scores = json.loads(classification_scores)
    with open(file_path, "w") as f:
        json.dump(classification_scores, f)

    return FileResponse(file_path, media_type="application/json", filename=file_name)

@app.get("/download_plot")
def download_plot(image_id : str, classification_scores : str):
    """Download the PNG file showing the
    top 5 scores in a plot (bar chart)."""
    scores = json.loads(classification_scores)
    labels = [score[0] for score in scores]
    values = [score[1] for score in scores]
    file_name = f"classification_plot_{image_id}.png"
    file_path = "downloads/" + file_name
    #create plot
    colors = ["darkgreen", "xkcd:crimson", "goldenrod", "blue", "xkcd:plum"]
    labels.reverse()
    values.reverse()
    colors.reverse()
    plt.figure(figsize=(9, 5))
    plt.barh(labels, values, color=colors)
    plt.suptitle('Output scores')
    plt.grid()
    plt.savefig(file_path)
    plt.close()

    return FileResponse(file_path, media_type="image/png", filename=file_name)

  
  #Feature 2:
class TransformRequest(BaseModel):
    image_id: str
    brightness: float
    contrast: float
    color: float
    sharpness: float

@app.post("/transform")
async def transform_image(request: TransformRequest):
    """Apply image transformations and return the result"""
    try:
        # Open the image
        image_path = os.path.join(config.image_folder_path, request.image_id)
        img = Image.open(image_path)

        # Apply transformations
        if request.brightness != 1.0:
            img = ImageEnhance.Brightness(img).enhance(request.brightness)
        if request.contrast != 1.0:
            img = ImageEnhance.Contrast(img).enhance(request.contrast)
        if request.color != 1.0:
            img = ImageEnhance.Color(img).enhance(request.color)
        if request.sharpness != 1.0:
            img = ImageEnhance.Sharpness(img).enhance(request.sharpness)

        # Save to bytes
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        return Response(content=img_byte_arr.getvalue(), media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
