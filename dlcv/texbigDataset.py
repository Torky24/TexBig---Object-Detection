import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import numpy as np


class TexBigDataset(Dataset):
    def __init__(self, root: str, mode: str, transform=None, isMaskRCNN: bool = False):
        """
        Custom dataset for the TexBig dataset.

        Args:
            root (str): Root directory of the dataset.
            mode (str): Dataset mode. Supported modes are 'train', 'validate', and 'test'.
            transform (callable): Optional transform to be applied to the image and target.
            isMaskRCNN (bool): Flag indicating whether the dataset is used for Mask R-CNN.
        """
        self.mode = mode
        if self.mode == "train":
            self.img_path = os.path.join(root, "train")
            self.json_path = os.path.join(root, "train.json")
        elif self.mode == "validate":
            self.img_path = os.path.join(root, "val")
            self.json_path = os.path.join(root, "val.json")
        elif self.mode == "test":
            self.img_path = os.path.join(root, "test")
        else:
            raise ValueError(
                "Invalid mode. Supported modes are 'train', 'validate', and 'test'."
            )

        if mode != "test":
            with open(self.json_path, "r") as json_file:
                self.data = json.load(json_file)
            self.img = self.preprocess_data()
            self.annotations = self.data["annotations"]

        self.transform = transform
        self.isMaskRCNN = isMaskRCNN

    def __len__(self):
        if hasattr(self, "img"):
            return len(self.img)
        else:
            img_files = sorted(os.listdir(self.img_path))
            return len(img_files)

    def __getitem__(self, idx):
        if hasattr(self, "img"):
            img_name = self.img[idx]["file_name"]
            img_id = self.img[idx]["id"]
            img_path = os.path.join(self.img_path, img_name)
        else:
            img_files = sorted(os.listdir(self.img_path))
            img_name = img_files[idx]
            img_path = os.path.join(self.img_path, img_name)
            img_id = None

        img = Image.open(img_path).convert("RGB")

        target = {}
        if self.mode == "test":
            target["img_name"] = img_name

        if hasattr(self, "annotations") and img_id is not None:
            annotations = [
                anno for anno in self.annotations if anno["image_id"] == img_id
            ]
            boxes = []
            labels = []
            areas = []
            iscrowds = []

            for annotation in annotations:
                xmin, ymin, width, height = annotation["bbox"]
                xmax = xmin + width
                ymax = ymin + height
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(annotation["category_id"])
                areas.append(annotation["area"])
                iscrowds.append(annotation["iscrowd"])

            target = {
                "boxes": torch.as_tensor(boxes, dtype=torch.float32),
                "labels": torch.as_tensor(labels, dtype=torch.int64),
                "image_id": torch.tensor([idx]),
                "area": torch.as_tensor(areas, dtype=torch.float32),
                "iscrowd": torch.as_tensor(iscrowds, dtype=torch.uint8),
            }

            if self.isMaskRCNN:
                masks = []
                for annotation in annotations:
                    segmentation = annotation["segmentation"]
                    mask = self.create_mask(segmentation, img.size)
                    masks.append(mask)
                masks = torch.stack(masks)
                target["masks"] = masks

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target if target is not None else img_name

    def preprocess_data(self):
        """
        Preprocesses the dataset by removing images without annotations.

        Returns:
            list: Preprocessed image metadata.
        """
        images = []
        for img in self.data["images"]:
            img_id = img["id"]
            annotations = [
                anno for anno in self.data["annotations"] if anno["image_id"] == img_id
            ]
            if len(annotations) > 0:
                images.append(img)
        return images

    def create_mask(self, segmentation, image_size):
        """
        Creates a mask from the given segmentation.

        Args:
            segmentation (list): List of segmentation points.
            image_size (tuple): Size of the image.

        Returns:
            torch.Tensor: Binary mask tensor.
        """
        mask = Image.fromarray(np.zeros(image_size, dtype=np.uint8))
        draw = ImageDraw.Draw(mask)
        for segment in segmentation:
            pts = np.array(segment).reshape((-1, 2))
            pts = [tuple(point) for point in pts]
            draw.polygon(pts, outline=1, fill=1)
        return torch.from_numpy(np.array(mask))
