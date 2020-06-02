#Filename:	fullgrad_simple.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Sab 30 Mei 2020 02:02:17  WIB

import torch
import torch.nn as nn
import torch.nn.functional as F

class FullGradSimple(object):
    def __init__(self, model):
        self.model = model

    def _getGradients(self, image, target_class = None):
        '''
        '''
        image = image.requires_grad_()
        out, features = self.model.getFeatures(image)
        
        if target_class is None:
            target_class = out.data.max(1, keepdim = True)[1]

        agg = 0
        for i in range(len(image)):
            agg += out[i, target_class[i]]

        self.model.zero_grad()
        # compute the gradient w.r.t input and intermediate features
        gradient = torch.autograd.grad(outputs = agg, inputs = features)
        
        # input gradient: gradient[0]
        # imtermediate gradient: gradient[1:]
        return gradient[0], gradient[1:]

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
                    gradient = F.interpolate(temp, size=img_size[2], mode = 'bilinear', align_corners=False)
                else:
                    gradient = F.interpolate(temp, size=(img_size[2], img_size[3]), mode = 'bilinear', align_corners=False)
                cam += gradient.sum(1, keepdim = True)

        return cam

