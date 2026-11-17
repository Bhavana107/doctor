from __future__ import annotations

from PIL import Image, ImageOps
import torch.nn as nn
from torchvision.models import resnet50


IMAGE_SIZE = (224, 112)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def prepare_image(
    image: Image.Image,
    size: int | tuple[int, int] = IMAGE_SIZE,
) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    output_size = (size, size) if isinstance(size, int) else size
    return ImageOps.pad(
        image,
        output_size,
        method=Image.Resampling.BICUBIC,
        color=(255, 255, 255),
        centering=(0.5, 0.5),
    )


def build_model(num_classes: int, pretrained: bool = False) -> nn.Module:
    from torchvision.models import ResNet50_Weights

    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
