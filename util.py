from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import variable
import numpy as np
import cv2


def predict_transform(prediction, inp_img, anchors, num_classes, CUDA=True):
    batch_size = prediction.size()[0]
    stride = inp_img // prediction.size()[2]
    grid_size = inp_img // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    prediction = prediction.view(
        batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(
        batch_size, grid_size*grid_size*num_anchors, bbox_attrs)

    anchors = [(a[0]/grid_size, a[1]/grid_size) for a in anchors]
    # The dimensions of the anchors are in accordance to the height
    # and width attributes of the net block.

    for i in (0, 1, 4):
        # Sigmoid for (x, y) and objectness score
        prediction[:, :, i] = torch.sigmoid(prediction[:, :, i])

    grid = np.arange(grid_size)
    x_offset, y_offset = np.meshgrid(grid, grid)
    x_offset = torch.FloatTensor(x_offset).view(-1, 1)
    y_offset = torch.FloatTensor(y_offset).view(-1, 1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1)
    x_y_offset = x_y_offset.repeat(1, num_anchors).view(-1, 2).unsqueeze(0)
    # x_y_offset.size(): (169, 2) -> (1, 507, 2)

    prediction[:, :, :2] += x_y_offset
    # add co-ordinate to the (x, y) of bbox

    anchors = torch.FloatTensor(anchors)
    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors

    prediction[:, :, 5:5 +
               num_classes] = torch.sigmoid(prediction[:, :, 5:5+num_classes])

    prediction[:, :, :4] *= stride
    # prediction.size(): batch_size x 13*13*3 x 85

    return prediction
