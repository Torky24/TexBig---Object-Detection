from PIL import Image
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_image_with_bboxes(
    dataset_path: str, image_path: str, save_dir: str = None
) -> None:
    """
    Load and display an image with bounding boxes from a COCO-formatted dataset.

    Args:
        dataset_path (str): Path to the COCO dataset file (JSON format).
        image_path (str): Path to the image file.
        save_dir (str, optional): Path to the directory to save the plotted image. Defaults to None.

    Returns:
        None
    """
    with open(dataset_path, "r") as file:
        dataset = json.load(file)

    image_info = None
    for img_info in dataset["images"]:
        if os.path.basename(img_info["file_name"]) == os.path.basename(
            image_path
        ) or img_info["file_name"] == os.path.basename(image_path):
            image_info = img_info
            break

    if image_info is None:
        print("Image not found in the dataset.")
        return

    image = Image.open(image_path)
    image = image.convert("RGB")
    plt.figure(figsize=(10, 8))
    plt.imshow(image)

    annotations = [
        ann for ann in dataset["annotations"] if ann["image_id"] == image_info["id"]
    ]

    ax = plt.gca()
    for ann in annotations:
        bbox = ann["bbox"]
        label_id = ann["category_id"]
        for cat in dataset["categories"]:
            if cat["id"] == label_id:
                label_name = cat["name"]
                break

        rect = patches.Rectangle(
            (bbox[0], bbox[1]),
            bbox[2],
            bbox[3],
            linewidth=2,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)

        ax.text(
            bbox[0],
            bbox[1] - 5,
            label_name,
            color="r",
            fontsize=10,
            backgroundcolor="w",
        )

    plt.axis("off")

    if save_dir:
        save_path = os.path.join(save_dir, os.path.basename("annotated" + image_path))
        plt.savefig(save_path)
        print(f"Plotted image saved at: {save_path}")


import json
import matplotlib.pyplot as plt
import numpy as np


def plot_label_distribution(train_file: str, val_file: str) -> None:
    """
    Plot the label distribution for the train and validation datasets.

    Args:
        train_file (str): Path to the train JSON file.
        val_file (str): Path to the validation JSON file.

    Returns:
        None
    """
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

    def count_labels(dataset: dict) -> dict:
        """
        Count the occurrence of each label in the dataset.

        Args:
            dataset (dict): Dataset dictionary containing annotations.

        Returns:
            dict: Dictionary containing label counts.
        """
        label_counts = {}
        for entry in dataset["annotations"]:
            label_id = entry["category_id"]
            label = label_mapping.get(label_id)
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1
        return label_counts

    with open(train_file, "r") as f:
        train_data = json.load(f)

    with open(val_file, "r") as f:
        val_data = json.load(f)

    train_label_counts = count_labels(train_data)
    val_label_counts = count_labels(val_data)

    labels = list(label_mapping.values())
    train_counts = [train_label_counts.get(label, 0) for label in labels]
    val_counts = [val_label_counts.get(label, 0) for label in labels]

    width = 0.35
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, train_counts, width, label="Train")
    ax.bar(x + width / 2, val_counts, width, label="Validation")

    ax.set_xlabel("Label")
    ax.set_ylabel("Count")
    ax.set_title("Label Distribution - Train vs Val")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.legend()

    plt.show()


if __name__ == "__main__":
    image_path = "/Users/torky/Documents/final-project-Torky24/images/14688302_1881_Seite_008.png"
    json_file = "/Users/torky/Downloads/train.json"
    val_file = "/Users/torky/Downloads/val.json"

    # plot_image_with_bboxes(
    #    annotation_file,
    #     image_path,
    #     save_dir="/Users/torky/Documents/final-project-Torky24/images",
    # )

    plot_label_distribution(json_file, val_file)
