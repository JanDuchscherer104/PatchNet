import random
import threading
from typing import Dict, List, Tuple

import albumentations as A
import numpy as np
import torch
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.pytorch import ToTensorV2
from torch import Tensor


class RotateAndRecord(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1):
        super(RotateAndRecord, self).__init__(always_apply, p)
        self.local = threading.local()  # Thread-local storage for rotation index

    def apply(self, img: np.ndarray[np.uint8], **params) -> np.ndarray:
        self.local.rotation_idx = np.random.randint(0, 4)
        return np.rot90(m=img, k=self.local.rotation_idx)  # type: ignore

    def get_transform_init_args_names(self):
        return ()

    def get_params_dependent_on_targets(self, params):
        return {}

    def get_applied_rotation(self):
        return self.local.rotation_idx


class AlbumTransforms:
    def __init__(self, resize: Tuple[int, int] = (64, 64)):
        self.train_augmentations = A.Compose(
            [
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.75
                ),
                A.ElasticTransform(alpha=32, sigma=200, alpha_affine=120 * 0.03, p=0.5),
                A.OpticalDistortion(p=1, shift_limit=0.2),
                A.GridDistortion(num_steps=2, distort_limit=0.2, p=0.4),
                A.GaussianBlur(p=0.4),
                A.RGBShift(
                    r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.95
                ),
                A.OneOf(
                    [
                        A.Blur(blur_limit=3, p=0.5),
                        A.ColorJitter(p=0.8),
                    ],
                    p=0.8,
                ),
                A.CoarseDropout(max_holes=8, max_height=4, max_width=4, p=0.3),
            ]
        )
        self.transforms = A.ReplayCompose(
            [
                A.Resize(*resize),
                RotateAndRecord(always_apply=True),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),  # ImageNet stats
                ToTensorV2(),
            ]
        )

    def __call__(
        self,
        puzzle_pieces: Dict[str, np.ndarray[np.uint8]],  # "piece_{id}"
        labels: np.ndarray[np.uint8],  # (num_pieces, 3) id, row, col
        is_train: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        transformed_pieces: List[Tensor] = []
        rotation_labels: List[Tensor] = []

        for label in labels:
            id, *_ = label
            piece = puzzle_pieces[f"piece_{id}"]
            if is_train:
                piece = self.train_augmentations(image=piece)["image"]
            transformed = self.transforms(image=piece)
            transformed_piece = transformed["image"]

            rotation_label = self.transforms.transforms[1].get_applied_rotation()

            transformed_pieces.append(transformed_piece)
            rotation_labels.append(rotation_label)

        # Convert to tensors
        stacked_transformed_pieces = torch.stack(transformed_pieces, dim=0)
        labels = torch.cat(
            [
                torch.from_numpy(labels[:, 1:]).type(torch.int64),
                torch.tensor(rotation_labels, dtype=torch.int64).unsqueeze(1),
            ],
            dim=1,
        )

        # Shuffle both tensors along the pieces dimension
        indices = torch.randperm(stacked_transformed_pieces.size(0))
        stacked_transformed_pieces = stacked_transformed_pieces[indices]
        labels = labels[indices]

        return stacked_transformed_pieces, labels
