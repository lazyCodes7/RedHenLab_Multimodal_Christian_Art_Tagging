import torchvision.transforms as transforms
import torchvision.transforms.functional as FT
import numpy as np
class SquarePad:
  def __call__(self, image):
    w, h = image.size
    max_wh = np.max([w, h])
    hp = int((max_wh - w) / 2)
    vp = int((max_wh - h) / 2)
    padding = (hp, vp, hp, vp)
    return FT.pad(image, padding, 0, 'constant')
  

class Transform:
    def __init__(self):
        self.transform=transforms.Compose([
            SquarePad(),

            # Padding is done to ensure resolution is same after resizing
            transforms.Resize((224,224)),
            transforms.CenterCrop((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.RandomHorizontalFlip(p=0.5)
        ])

        self.val_transform = transforms.Compose([
            SquarePad(),
            transforms.Resize((224,224)),
            transforms.CenterCrop((224,224)),
            transforms.ToTensor()
        ])

        self.inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.255]
        )