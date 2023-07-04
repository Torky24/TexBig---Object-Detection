import torchvision
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def load_pretrained_faster_rcnn_resnet50_fpn(num_classes: int) -> torch.nn.Module:
    """
    Loads a pretrained Faster R-CNN model with a ResNet50 backbone and FPN architecture.

    Args:
        num_classes (int): Number of output classes.

    Returns:
        torch.nn.Module: Pretrained Faster R-CNN model.
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_channels=in_features, num_classes=num_classes
    )

    return model


def load_pretrained_mask_rcnn_resnet50_fpn(num_classes: int) -> torch.nn.Module:
    """
    Loads a pretrained Mask R-CNN model with a ResNet50 backbone and FPN architecture.

    Args:
        num_classes (int): Number of output classes.

    Returns:
         torch.nn.Module: Pretrained Mask R-CNN model.
    """
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256

    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    return model


def load_pretrained_faster_rcnn_resnet50_fpn_v2(num_classes: int) -> torch.nn.Module:
    """
    Loads a pretrained Faster R-CNN model with a ResNet50 backbone and FPN architecture.

    Args:
        num_classes (int): Number of output classes.

    Returns:
        torch.nn.Module: Pretrained Faster R-CNN model.
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(pretrained=True)

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_channels=in_features, num_classes=num_classes
    )

    return model
