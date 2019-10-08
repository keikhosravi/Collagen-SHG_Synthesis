import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

class EDGE(torch.nn.Module):
    def __init__(self, use=True):
        super(EDGE, self).__init__()
        self.usecuda = use

    def forward(self, img1, img2):
        usecuda = self.usecuda
        x_filter = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        y_filter = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        weights_x = torch.from_numpy(x_filter).float().unsqueeze(0).unsqueeze(0)
        weights_y = torch.from_numpy(y_filter).float().unsqueeze(0).unsqueeze(0)
        if usecuda:
            weights_x = weights_x.cuda()
            weights_y = weights_y.cuda()
        
        g1_x = F.conv2d(img1, weights_x, bias=None, stride=1, padding=1, dilation=1, groups=1)
        g2_x = F.conv2d(img2, weights_x, bias=None, stride=1, padding=1, dilation=1, groups=1)
        g1_y = F.conv2d(img1, weights_y, bias=None, stride=1, padding=1, dilation=1, groups=1)
        g2_y = F.conv2d(img2, weights_y, bias=None, stride=1, padding=1, dilation=1, groups=1)
        
        
        g_diff_x = torch.abs(g1_x - g2_x)
        g_dif_y = torch.abs(g1_y - g2_y)
        
#         g_1 = torch.sqrt(torch.pow(g1_x, 2) + torch.pow(g1_y, 2))
#         g_2 = torch.sqrt(torch.pow(g2_x, 2) + torch.pow(g2_y, 2))
        
        return torch.mean(g_diff_x + g_dif_y)
