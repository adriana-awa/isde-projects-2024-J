"""
This is a simple classification service. It accepts an url of an
image and returns the top-5 classification labels and scores.
"""
import importlib
import json
import logging
import os
import torch
from PIL import Image
from torchvision import transforms
from pathlib import Path
from app.config import Configuration


conf = Configuration()


def fetch_image(image_id):
    """Gets the image from the specified ID. It returns only images
    downloaded in the folder specified in the configuration object."""
    image_path = os.path.join(conf.image_folder_path, image_id)
    img = Image.open(image_path)
    return img


def get_labels():
    """Returns the labels of Imagenet dataset as a list, where
    the index of the list corresponds to the output class."""
    labels_path = os.path.join(conf.image_folder_path, "imagenet_labels.json")
    with open(labels_path) as f:
        labels = json.load(f)
    return labels


def get_model(model_id):
    """Imports a pretrained model from the ones that are specified in
    the configuration file. This is needed as we want to pre-download the
    specified model in order to avoid unnecessary waits for the user."""
    if model_id in conf.models:
        try:
            module = importlib.import_module("torchvision.models")
            return module.__getattribute__(model_id)(weights="DEFAULT")
        except ImportError:
            logging.error("Model {} not found".format(model_id))
    else:
        raise ImportError


def classify_image(model_id: str, image_input):
    """
    Classifies an image using the specified model.
    image_input can be either a path (str) or a PIL.Image object
    """
    if isinstance(image_input, (str, Path)):
        image = Image.open(image_input)
    else:
        image = image_input

    model = get_model(model_id)
    model.eval()
    transform = transforms.Compose(
        (
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        )
    )

    # apply transform from torchvision
    image = image.convert("RGB")
    preprocessed = transform(image).unsqueeze(0)

    # gets the output from the model
    out = model(preprocessed)
    _, indices = torch.sort(out, descending=True)

    # transforms scores as percentages
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

    # gets the labels
    labels = get_labels()

    # takes the top-5 classification output and returns it
    # as a list of tuples (label_name, score)
    output = [[labels[idx], percentage[idx].item()] for idx in indices[0][:5]]

    #image.close()
    return output
