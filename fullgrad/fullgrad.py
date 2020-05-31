#Filename:	fullgrad.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Min 31 Mei 2020 09:42:59  WIB

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import isclose

class FullGrad(object):
    """
    compute the FullGrad saliency and full gradient decomposition
    """

    def __init__(self, model, img_size = (3, 224, 224)):
        self.model = model
        self.img_size = (1,) + img_size
        self.model.eval()
        self.blockwise_biases = self.model.getBiases()
        self.checkCompleteness()

    def checkCompleteness(self):
        """
        check if completeness property is sastified. If not, it ususally means 
        that some bias gradient are not computed. 
        """

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input = torch.randn(self.img_size).to(device)
        self.model.eval()

        output = self.model(input)
        # compute the full-gradient
        input_grad, bias_grad = self._getGradients(input, target_class = None)

        fullgrad_sum = (input_grad * input).sum()
        for i in range(len(bias_grad)):
            fullgrad_sum += bias_grad[i].sum()

        # compare raw output and full grad sum
        err_message = "\nThis is due to incorrect computation of bias-gradients. Please check models/vgg.py for more information."
        err_string = "Completeness test failed! Raw output = " + str(raw_output.max().item()) + " Full-gradient sum = " + str(fullgradient_sum.item())
        assert isclose(raw_output.max().item(), fullgradient_sum.item(), rel_tol=0.00001), err_string + err_message
        print('Completeness test passed for FullGrad.')

    def _getGradients(self, image, target_class = None):
        """
        compute full-gradient decomposition for an image
        """

        image =  image.requiresg_grad()
        out, features = self.model.getFeature(image)

        if target_class is None:
            target_class = out.data.max(1, keepdim = True)[1]

        agg = 0
        for i in range(image.size(0)):
            agg += out[i, target_class[i]]

        self.model.zero_grad()
        # gradients w.r.t. input and features
        gradients = torch.autograd.grad(outputs = agg, inputs = features)
        
        return gradients[0], gradients[1:]

    def _postprecess(self, input, eps = 1e-6):
        input = abs(input)
        # rescale operation to ensure gradients lie between 0 and 1
        # note: the postprocess may contribute to the visual appearance
        input = input - input.min()
        input = input / (input.max() + eps)
        return input

    def fullgrad(self, image, target_class = None):
        self.model.eval()
        input_grad, intermediate_grad = self._getGradients(image, target_class = target_class)
        img_size = image.size()

        # Input grad * image
        grad = input_grad * image
        grad = self._postprecess(grad).sum(1, keepdim = True)
        cam = grad

        # Intermediate gradient
        for i in range(len(intermediate_grad)):
            if len(intermediate_grad[i].size()) == len(img_size):
                temp = self._postprecess(intermediate_grad[i])
                if len(img_size) == 3:
                    gradient = F.interpolate(temp, size=im_size[2], mode = 'bilinear', align_corners=False)
                else:
                    gradient = F.interpolate(temp, size=(im_size[2], im_size[3]), mode = 'bilinear', align_corners=False)
                cam += gradient.sum(1, keepdim = True)

        return cam


