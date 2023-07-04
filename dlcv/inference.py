import os
import json
from typing import List, Dict

import torch
from dlcv.model import (
    load_pretrained_faster_rcnn_resnet50_fpn_v2,
    load_pretrained_faster_rcnn_resnet50_fpn,
    load_pretrained_mask_rcnn_resnet50_fpn,
)
from dlcv.texbigDataset import TexBigDataset
from torch.utils.data import DataLoader
import dlcv.detection.transforms as transforms
import dlcv.detection.utils as utils
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict


def select_model(model_name: str, model_path: str) -> torch.nn.Module:
    """
    Selects and loads the specified pre-trained model.

    Args:
        model_name (str): The name of the model to select.
        model_path (str): The path to the pre-trained model weights.

    Returns:
        torch.nn.Module: The loaded pre-trained model.
    """
    available_models = ["FasterRCNN V1", "FasterRCNN V2", "MaskRCNN"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name not in available_models:
        raise Exception(
            "Please double check the input model. "
            "Available Models are only FasterRCNN V1, "
            "FasterRCNN V2, MaskRCNN."
        )

    if model_name == "FasterRCNN V2":
        model = load_pretrained_faster_rcnn_resnet50_fpn_v2(num_classes=20)
    elif model_name == "FasterRCNN V1":
        model = load_pretrained_faster_rcnn_resnet50_fpn(num_classes=20)
    elif model_name == "MaskRCNN":
        model = load_pretrained_mask_rcnn_resnet50_fpn(num_classes=20)

    model.to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


def create_test_dataloader(images_path: str) -> DataLoader:
    """
    Creates a data loader for the test dataset.

    Args:
        images_path (str): The path to the directory containing the test images.

    Returns:
        DataLoader: The data loader for the test dataset.
    """
    transform = transforms.Compose(
        [
            transforms.ImageToTensor(),
            transforms.ConvertImageDtype(torch.float32),
        ]
    )

    test_dataset = TexBigDataset(
        root=images_path,
        mode="test",
        transform=transform,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        collate_fn=utils.collate_fn,
    )

    return test_dataloader


def extract_predictions(
    model: torch.nn.Module, data_loader: DataLoader, device: torch.device
) -> List[Dict[str, any]]:
    """
    Extracts predictions from a trained model.

    Args:
        model (torch.nn.Module): The trained model.
        data_loader (DataLoader): The data loader for the test data.
        device (torch.device): The device to run the inference on (e.g., "cuda", "cpu").

    Returns:
        List[Dict[str, any]]: A list of dictionaries containing the predictions.
            Each dictionary represents a prediction and contains the following keys:
            - "file_name": The name of the image file.
            - "category_id": The predicted category ID.
            - "bbox": The predicted bounding box coordinates [xmin, ymin, width, height].
            - "score": The confidence score of the prediction.
    """
    predictions = []
    model.eval()
    for images, targets in data_loader:
        images = list(img.to(device) for img in images)
        with torch.no_grad():
            outputs = model(images)

        for i, output in enumerate(outputs):
            file_name = targets[i]["img_name"]
            scores = output["scores"]
            boxes = output["boxes"]
            labels = output["labels"]

            for score, box, label in zip(scores, boxes, labels):
                xmin, ymin, xmax, ymax = box.tolist()
                width = xmax - xmin
                height = ymax - ymin
                prediction = {
                    "file_name": file_name,
                    "category_id": label.item(),
                    "bbox": [xmin, ymin, width, height],
                    "score": score.item(),
                }
                predictions.append(prediction)

    output_file = "predictions.json"
    with open(output_file, "w") as f:
        json.dump(predictions, f)

    return predictions




def plot_predictions(predictions, images_path, output_path):
    """
    Plots the predictions on the images and saves them to the output path.

    Args:
        predictions (List[Dict[str, any]]): A list of dictionaries containing the predictions.
            Each dictionary represents a prediction and contains the following keys:
            - "file_name": The name of the image file.
            - "category_id": The predicted category ID.
            - "bbox": The predicted bounding box coordinates [xmin, ymin, width, height].
            - "score": The confidence score of the prediction.
        images_path (str): The path to the directory containing the input images.
        output_path (str): The path to the directory to save the output images.
    """
    os.makedirs(output_path, exist_ok=True)
    label_mapping = {
        1: "paragraph",
        2: "equation",
        3: "logo",
        4: "editorial note",
        5: "sub-heading",
        6: "caption",
        7: "image",
        8: "footnote",
        9: "page number",
        10: "table",
        11: "heading",
        12: "author",
        13: "decoration",
        14: "footer",
        15: "header",
        16: "noise",
        17: "frame",
        18: "column title",
        19: "advertisement",
    }

    image_predictions = defaultdict(list)
    for prediction in predictions:
        file_name = prediction["file_name"]
        image_predictions[file_name].append(prediction)

    for file_name, image_preds in image_predictions.items():
        image_path = os.path.join(images_path, file_name)
        output_image_path = os.path.join(output_path, file_name)

        image = Image.open(image_path)
        image = image.convert("RGB")

        plt.imshow(image)

        for prediction in image_preds:
            category_id = prediction["category_id"]
            bbox = prediction["bbox"]
            score = prediction["score"]

            plt.gca().add_patch(
                plt.Rectangle(
                    (bbox[0], bbox[1]),
                    bbox[2],
                    bbox[3],
                    fill=False,
                    edgecolor="red",
                    linewidth=2,
                )
            )

            plt.text(
                bbox[0],
                bbox[1],
                f"{label_mapping[category_id]} ({score:.2f})",
                color="red",
                fontsize=8,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="white", pad=0),
            )

        plt.axis("off")
        plt.savefig(output_image_path, bbox_inches="tight")
        plt.close()


def inference() -> None:
    """
    Run inference using a selected model on a set of images and plot the predictions.

    This function prompts the user to select a model, input the model path, images path, and output path.
    It then loads the model, creates a test dataloader, performs inference on the images,
    and plots the predictions.

    Args:
        None

    Returns:
        None
    """
    model_name = input("Please select model: \n")

    model_path = input("Please input the path to the model: \n")

    images_path = input("Please input the path to the images: \n")

    output_path = input("Please input the output path: \n")

    model = select_model(model_name, model_path)
    test_dataloader = create_test_dataloader(images_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions = extract_predictions(model, test_dataloader, device)

    predictions_file = "/Users/torky/Documents/final-project-Torky24/predictions.json"
    with open(predictions_file, "r") as f:
        predictions = json.load(f)

    plot_predictions(predictions, images_path, output_path)


if __name__ == "__main__":
    inference()
