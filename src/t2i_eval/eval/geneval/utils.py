import clip_benchmark.metrics.zeroshot_classification as zsc
import numpy as np
import torch
from PIL import Image

# Patch zsc.tqdm to be silent or a no-op if needed, as in original code
zsc.tqdm = lambda it, *args, **kwargs: it

# Color list for CLIP-based color classification
COLORS = [
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "purple",
    "pink",
    "brown",
    "black",
    "white",
]

COLOR_CLASSIFIERS = {}


class ImageCrops(torch.utils.data.Dataset):
    def __init__(self, image: Image.Image, objects, transform=None, bgcolor="#999"):
        self._image = image.convert("RGB")
        self.transform = transform
        if bgcolor == "original":
            self._blank = self._image.copy()
        else:
            self._blank = Image.new("RGB", image.size, color=bgcolor)
        self._objects = objects

    def __len__(self):
        return len(self._objects)

    def __getitem__(self, index):
        box, mask = self._objects[index]
        if mask is not None:
            # Ensure mask matches image size
            assert tuple(self._image.size[::-1]) == tuple(mask.shape), (
                index,
                self._image.size[::-1],
                mask.shape,
            )
            image = Image.composite(self._image, self._blank, Image.fromarray(mask))
        else:
            image = self._image

        # Match official Geneval behavior: crop(box[:4]) directly.
        crop_box = [int(c) for c in box[:4]]
        image = image.crop(crop_box)

        if self.transform:
            return self.transform(image), 0
        return image, 0


def color_classification(
    image: Image.Image,
    bboxes,
    classname: str,
    clip_model,
    tokenizer,
    device: str | torch.device,
    transform,
):
    """
    Classify color of objects using CLIP.
    """
    device_str = str(device)

    if classname not in COLOR_CLASSIFIERS:
        COLOR_CLASSIFIERS[classname] = zsc.zero_shot_classifier(
            clip_model,
            tokenizer,
            COLORS,
            [
                f"a photo of a {{c}} {classname}",
                f"a photo of a {{c}}-colored {classname}",
                "a photo of a {c} object",
            ],
            device_str,
        )
    clf = COLOR_CLASSIFIERS[classname]

    dataloader = torch.utils.data.DataLoader(
        ImageCrops(image, bboxes, transform=transform),
        batch_size=16,
        num_workers=0,
    )
    with torch.no_grad():
        pred, _ = zsc.run_classification(clip_model, clf, dataloader, device_str)
        return [COLORS[index.item()] for index in pred.argmax(1)]


def compute_iou(box_a, box_b):
    def area_fn(box):
        return max(box[2] - box[0] + 1, 0) * max(box[3] - box[1] + 1, 0)

    i_area = area_fn(
        [
            max(box_a[0], box_b[0]),
            max(box_a[1], box_b[1]),
            min(box_a[2], box_b[2]),
            min(box_a[3], box_b[3]),
        ]
    )
    u_area = area_fn(box_a) + area_fn(box_b) - i_area
    return i_area / u_area if u_area else 0


def relative_position(obj_a, obj_b, position_threshold=0.1):
    """Give position of A relative to B, factoring in object dimensions"""
    boxes = np.array([obj_a[0], obj_b[0]])[:, :4].reshape(2, 2, 2)
    center_a, center_b = boxes.mean(axis=-2)
    dim_a, dim_b = np.abs(np.diff(boxes, axis=-2))[..., 0, :]
    offset = center_a - center_b
    #
    revised_offset = np.maximum(
        np.abs(offset) - position_threshold * (dim_a + dim_b), 0
    ) * np.sign(offset)
    if np.all(np.abs(revised_offset) < 1e-3):
        return set()
    #
    dx, dy = revised_offset / np.linalg.norm(offset)
    relations = set()
    if dx < -0.5:
        relations.add("left of")
    if dx > 0.5:
        relations.add("right of")
    if dy < -0.5:
        relations.add("above")
    if dy > 0.5:
        relations.add("below")
    return relations
