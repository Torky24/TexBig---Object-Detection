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


def plot_class_distribution(dataset_path: str, save_dir: str = None) -> None:
    """
    Extract the class distribution from a COCO-formatted dataset and plot it as a bar graph.

    Args:
        dataset_path (str): Path to the COCO dataset file (JSON format).
        save_dir (str, optional): Path to the directory to save the plotted image. Defaults to None.

    Returns:
        None
    """
    with open(dataset_path, "r") as file:
        dataset = json.load(file)

    class_distribution = {}
    for category in dataset["categories"]:
        class_distribution[category["name"]] = 0

    for annotation in dataset["annotations"]:
        class_id = annotation["category_id"]
        class_name = next(
            category["name"]
            for category in dataset["categories"]
            if category["id"] == class_id
        )
        class_distribution[class_name] += 1

    classes = list(class_distribution.keys())
    counts = list(class_distribution.values())

    plt.figure(figsize=(10, 6))
    plt.bar(classes, counts)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("Class Distribution")
    plt.xticks(rotation=90)
    plt.tight_layout()

    if save_dir:
        save_path = os.path.join(save_dir, "class_distribution.png")
        plt.savefig(save_path)
        print(f"Class distribution plot saved at: {save_path}")


if __name__ == "__main__":
    image_path = "/Users/torky/Documents/final-project-Torky24/images/14688302_1881_Seite_008.png"
    annotation_file = "/Users/torky/Documents/final-project-Torky24/images/train.json"

    plot_image_with_bboxes(
        annotation_file,
        image_path,
        save_dir="/Users/torky/Documents/final-project-Torky24/images",
    )
    plot_class_distribution(
        annotation_file, save_dir="/Users/torky/Documents/final-project-Torky24/images"
    )
