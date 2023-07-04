import torch
import dlcv.detection.transforms as transforms
import dlcv.detection.utils as utils
from torch.utils.data import DataLoader
from dlcv.texbigDataset import TexBigDataset
from dlcv.inference import inference


from dlcv.model import ( # noqa
    load_pretrained_mask_rcnn_resnet50_fpn, # noqa
    load_pretrained_faster_rcnn_resnet50_fpn,
    load_pretrained_faster_rcnn_resnet50_fpn_v2,
)
from dlcv.train import train_model


if __name__ == "__main__":
    transform = transforms.Compose(
        [
            transforms.ImageToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.RandomPhotometricDistort(),
        ]
    )

    batch_size = 2
    num_classes = 20
    num_epochs = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = TexBigDataset(
        root="/kaggle/input/texbigdataset/archive",
        mode="train",
        transform=transform,
        isMaskRCNN=False,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=utils.collate_fn,
    )

    val_dataset = TexBigDataset(
        root="/kaggle/input/texbigdataset/archive",
        mode="validate",
        transform=transform,
        isMaskRCNN=False,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=utils.collate_fn,
    )

    test_dataset = TexBigDataset(
        root="/kaggle/input/texbigdataset/archive (1)",
        mode="test",
        transform=transform,
        isMaskRCNN=False,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=utils.collate_fn,
    )

    model = load_pretrained_faster_rcnn_resnet50_fpn_v2(
        num_classes=num_classes
        )

    model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.01, weight_decay=0.0001, momentum=0.9
    )

    train_model(
        model=model,
        train_data_loader=train_dataloader,
        validation_dataloader=val_dataloader,
        device=device,
        optimizer=optimizer,
        num_epochs=num_epochs,
    )

    inference()
