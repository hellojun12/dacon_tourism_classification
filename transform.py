import albumentations as A # fast image agumentation library
from albumentations.pytorch.transforms import ToTensorV2 # 이미지 형 변환

class TrainTransform:

    def __init__(self, h, w):

        self.transforms = [ 

            A.Resize(h, w),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
            ToTensorV2()
        ]
        

    def __call__(self, image):
        return A.Compose(self.transforms)(image=image)

class TestTransform:

    def __init__(self, h, w):
        self.transforms = [
            A.Resize(h, w),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
            ToTensorV2()
        ]

    def __call__(self, image):
        return A.Compose(self.transforms)(image=image)