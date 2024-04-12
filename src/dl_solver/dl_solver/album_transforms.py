import albumentations as A
from albumentations.pytorch import ToTensorV2


class AlbumTransforms:
    def __init__(self, resize: tuple = (64, 64)):
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
                A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3),
            ]
        )
        self.transforms = A.Compose(
            [
                A.Resize(*resize),
                A.RandomRotate90(always_apply=False, p=0.5),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # ImageNet
                ToTensorV2(),
            ]
        )

    def __call__(self, image, train=True):
        if train:
            image = self.train_augmentations(image=image)["image"]
        transformed = self.transforms(image=image)
        image = transformed["image"]
        # TODO: check if this is correct
        rotation = transformed["replay"]["transforms"][1]["applied"]
        rotation_label = [0, 1, 2, 3][rotation]

        return image, rotation_label
