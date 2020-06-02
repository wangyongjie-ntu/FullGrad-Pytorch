#Filename:	test.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Sab 30 Mei 2020 01:56:33  WIB

import torch
from torchvision import datasets, transforms, utils
import os

from fullgrad.fullgrad import *
from fullgrad.fullgrad_simple import *

from models.vgg import *
from misc_functions import *

dataset = './image'

sampler_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(dataset, transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406],
                std = [0.229, 0.224, 0.225])
            ])), 
        batch_size = batch_size, shuffle = False)

unnormalize = NormalizeInverse(mean = [0.485, 0.456, 0.406],
        std = std = [0.229, 0.224, 0.225])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = vgg11_bn(pretrained = True)
model = model.to(device)

fullgrad = FullGrad(model)
simple_fullgrad = SimpleFullGrad(model)

save_path = 'result'
if os.path.isdir(save_path):
    os.mkdir(save_path))


def compute_saliency_and_save():
    for idx, (data, target) in enumerate(sample_loader):
        data, target = data.to(device).requires_grad_(), target.to(device)

        # compute the saliency maps for the input data
        cam = fullgrad.fullgrad(data)
        cam_simple = simple_fullgrad.fullgrad(data)

        for i in range(data.size(0)):
            filename = save_path + str((idx + 1) * (i+1))
            filename_simple = filename + "_simple"

            image = unnormalize(data[i,:,:,:].cpu())
            save_saliency_map(image, cam[i,:,:,:], filename + ".jpg")
            save_saliency_map(image, cam_simple[i,:,:,:], filename_simple + ".jpg")


if __name__ == "__main__":
    compute_saliency_and_save()
