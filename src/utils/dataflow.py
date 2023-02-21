import os
from typing import Dict, Tuple, Union, Sequence

import PIL.Image
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop, get_dimensions, pad

from src.utils import transforms_utils
from src.utils.labels_dict import UNI_UID2UNAME


def ann_to_embedding(x: torch.Tensor, embeddings: torch.Tensor):
    '''
    args:
        x: annotation torch.Size([B, img_size[0], img_size[1]])
        embeddings: label embeddings torch.Size([C, embed_dim])

    returns:
        feature_map: label embeddings for every pixel torch.Size([B, img_size[0], img_size[1], embed_dim])
    '''
    orig_shape = x.shape
    indices = x.view(-1).long()
    feature_map = embeddings[indices]
    feature_map = feature_map.view(*orig_shape, embeddings.shape[-1])
    return feature_map


def ann_to_one_hot(ann: torch.Tensor, num_classes: int):
    base_shape = ann.shape
    ann = ann.view(-1)
    one_hot = F.one_hot(ann.long(), num_classes=num_classes)
    one_hot = one_hot.view(*base_shape, num_classes).permute(0, 3, 1, 2)
    return one_hot


class PairRandomCrop(transforms.RandomCrop):
    def forward(self, img1, img2):
        """
        Args:
            img1 (PIL Image or Tensor): Image to be cropped.
            img2 (PIL Image or Tensor): Image to be cropped.

        Returns:
            Tuple of two PIL Image or Tensor: Cropped image.
        """
        if self.padding is not None:
            img1 = pad(img1, self.padding, self.fill, self.padding_mode)
            img2 = pad(img2, self.padding, self.fill, self.padding_mode)

        _, height, width = get_dimensions(img1)
        _, height2, width2 = get_dimensions(img2)
        assert width2 == width
        assert height2 == height

        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img1 = pad(img1, padding, self.fill, self.padding_mode)
            img2 = pad(img2, padding, self.fill, self.padding_mode)

        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img1 = pad(img1, padding, self.fill, self.padding_mode)
            img2 = pad(img2, padding, self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img1, self.size)

        return crop(img1, i, j, h, w), crop(img2, i, j, h, w)

def get_data_transforms():
    mean, std = transforms_utils.get_imagenet_mean_std()

    img_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    ann_transform = transforms.Compose(
                [
                    transforms.PILToTensor(),
                ]
            )
    return img_transform, ann_transform

class CMPDataset(Dataset):
    def __init__(
        self,
        data_dirs: Sequence[str] = ("data/",),
        img_size: Union[int, Tuple[int, int]] = (512, 512),
    ):
        self.img_size: Tuple[int, int] = (
            (img_size, img_size) if isinstance(img_size, int) else img_size
        )
        
        self.pair_crop = PairRandomCrop(img_size, pad_if_needed=True, fill=1)
        self.transform, self.ann_transform = get_data_transforms()
        self.items = []
        for data_dir in data_dirs:
            files = os.listdir(data_dir)
            assert len(files) > 0, f"{data_dir} is empty"
            for file in files:
                if file.endswith(".jpg"):
                    img_path = os.path.join(data_dir, file)
                    ann_path = os.path.join(data_dir, file.replace(".jpg", ".png"))
                    assert os.path.isfile(
                        ann_path
                    ), f"Corresponding annotation missed {ann_path}"
                    self.items.append((img_path, ann_path))

    def create_cmp_label_map(
        self, labels: Dict[str, Dict[str, Union[str, int]]], start_idx: int = 194
    ):
        ret = {}
        keys = labels.keys()
        name_to_id = {v: k for k, v in UNI_UID2UNAME.items()}
        background_idx = name_to_id["unlabeled"]
        ret[1] = background_idx
        idx = start_idx
        for key in keys:
            if key in name_to_id:
                ret[labels[key]["label"]] = name_to_id[key]
            else:
                ret[labels[key]["label"]] = idx
                idx += 1
        return ret

    def __len__(self):
        return len(self.items)

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, ann_path = self.items[index]
        img = PIL.Image.open(img_path).convert("RGB")
        ann = PIL.Image.open(ann_path)
        img, ann = self.pair_crop(img, ann)
        x = self.transform(img)

        ann = self.ann_transform(ann).squeeze(0)
        ann = ann - 1  # change indexing

        return x, ann
