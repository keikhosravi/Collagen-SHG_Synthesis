import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

# add DenseNet structure
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
#         self.x = x
        self.block0 = nn.Sequential(
            # input image 96x96
            nn.ReLU(),
            nn.Conv2d(3, 64, (5, 5), (1, 1), (2, 2)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),

        )
    
        self.block1 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),       
        )
        
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 4, (1, 1), (1, 1)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(4),
        )
        
        self.side0_3 = nn.Sequential(
            nn.Conv2d(64, 4, (1, 1), (1, 1)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(4),
        )
        
        self.side1_3 = nn.Sequential(
            nn.Conv2d(64, 4, (1, 1), (1, 1)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(4),
        )
        
        self.side2_3 = nn.Sequential(
            nn.Conv2d(64, 4, (1, 1), (1, 1)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(4),
        )
        
        self.fc = nn.Sequential(
            nn.Conv2d(4, 1, (1, 1), (1, 1)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
            
        
    def forward(self, x):
        x=x.float()
        out = self.block0(x)  # 64x96x96
        res0_1 = out
        res0_2 = out
        res0_3 = self.side0_3(out)
        
        out = self.block1(out) # 64x96x96
        res1_2 = out
        res1_3 = self.side1_3(out)
        
        out = out + res0_1
        out = self.block2(out) # 64x96x96
        res2_3 = self.side2_3(out)
        
        out = out + res0_2 + res1_2
        out = self.block3(out) # 4x96x96
        
        out = out + res0_3 + res1_3 + res2_3
        out = self.fc(out)

        return out
    
    
    def _initialize_weights(self):
        pass