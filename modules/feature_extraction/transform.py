import torchvision.transforms as transforms
import numpy as np
import torchvision.transforms.functional as FT 
class Transform:
    def __init__(self):

        # now use it as the replacement of transforms.Pad class
        self.transform=transforms.Compose([

            # Padding is done to ensure resolution is same after resizing
            transforms.Resize((224,224)),
            transforms.CenterCrop((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.RandomHorizontalFlip(p=0.5)
        ])

        self.val_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.CenterCrop((224,224)),
            transforms.ToTensor()
        ])

class SquarePad:
	def __call__(self, image):
		w, h = image.size
		max_wh = np.max([w, h])
		hp = int((max_wh - w) / 2)
		vp = int((max_wh - h) / 2)
		padding = (hp, vp, hp, vp)
		return FT.pad(image, padding, 0, 'constant')