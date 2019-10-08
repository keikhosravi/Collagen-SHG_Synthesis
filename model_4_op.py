import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
# 4 original

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
#         self.x = x
        self.block0 = nn.Sequential(
            # input image 96x96
            nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1), # parameters
            # image 48x48
        )
        
        self.block1 = nn.Sequential(           
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1), # parameters
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1), # parameters
            # image 48x48
        )
        
        self.info1 = nn.Sequential(
            # 48x48
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1), # parameters
        )
        
        self.block2 = nn.Sequential(
            # image 48x48
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1), # parameters
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1), # parameters
            # image 48x48
        )
        
        self.info2 = nn.Sequential(
            # 48x48
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1), # parameters
            # 48x48
        )
        
        self.block3 = nn.Sequential(
            # image 48x48
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1), # parameters
            # image 24x24
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1), # parameters
            # image 24x24
        )
        
        self.info3 = nn.Sequential(
            # 24x24
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1), # parameters
            # 24x24
        )
        
        self.side3 = nn.Sequential(
            # image 48x48
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=2, stride=2
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1), # parameters
            # image 24x24
        )
        
        self.block4 = nn.Sequential(
            # image 24x24
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1), # parameters
            # image 12x12            
            nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1), # parameters
        )
        
        self.side4 = nn.Sequential(
            # image 24x24
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=2, stride=2
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1), # parameters
            # image 12x12
            # padding till here is all correct
        
        )
        
        self.block5 = nn.Sequential(
            # image 12x12
            nn.Conv2d(
                in_channels=256, out_channels=2*2*256, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1), # parameters
            nn.PixelShuffle(2),
            # image 24x24
        )
        
        self.side5 = nn.Sequential(
            # image 12x12
            nn.ConvTranspose2d(
                in_channels=256, out_channels=256, kernel_size=2, stride=2
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1), # parameters
            # image 24x24
        )
        
        self.block6 = nn.Sequential(
            # image 24x24
            nn.Conv2d(
                in_channels=256, out_channels=2*2*256, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1), # parameters
            nn.PixelShuffle(2),
            # image 48x48
            nn.Conv2d(
                in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1), # parameters
            # image 48x48
        )
        
        self.side6 = nn.Sequential(
            # image 24x24
            nn.ConvTranspose2d(
                in_channels=256, out_channels=128, kernel_size=2, stride=2,
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1), # parameters
            # image 48x48
        )
        
        self.block7 = nn.Sequential(
            # image 48x48
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1), # parameters
            # image 48x48           
        )
        
        self.block8 = nn.Sequential(
            # image 48x48
            nn.Conv2d(
                in_channels=128, out_channels=2*2*128, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1), # parameters
            nn.PixelShuffle(2),
            # image 96x96
        )
        
        self.side8 = nn.Sequential(
            # image 96x96
            nn.Conv2d(
                in_channels=128, out_channels=1, kernel_size=1, stride=1
            ),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.1), # parameters
            # image 96x96           
        )
        
        self.block9 = nn.Sequential(
            # image 96x96
            nn.Conv2d(
                in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.1), # parameters
            # image 96x96           
        )
        
        
        self.fc = nn.Sequential(
            nn.Sigmoid()
        )
            
        
    def forward(self, x):
        x=x.float()
        out = self.block0(x) 
        residual1 = out
        
        out = self.block1(out)
        info_1 = self.info1(out) 
        out = out + residual1       
        residual2 = out
        
        out = self.block2(out)
        info_2 = self.info2(out)
        out = out + residual2     
        residual3 = out
        
        residual3 = self.side3(residual3) 
        out = self.block3(out) 
        info_3 = self.info3(out)
        out = out + residual3    
        residual4 = out
        
        residual4 = self.side4(residual4)
        out = self.block4(out)
        out = out + residual4        
        residual5 = out
        
        residual5 = self.side5(residual5)
        out = self.block5(out)
        out = out + residual5 + info_3 # recieve info_2       
        residual6 = out
        
        residual6 = self.side6(residual6)
        out = self.block6(out)
        out = out + residual6 + info_2 # recieve info_1
        residual7 = out
                    
        out = self.block7(out)
        out = out + residual7 + info_1
        residual8 = out
        
        residual8 = self.side8(residual8)
        out = self.block8(out)

        out = self.block9(out) 

        out = self.fc(out)
        
#         out = nn.functional.interpolate(out, [96, 96])
        
#         out = F.relu(out) # fully connect sigmoid, image 96x96      
        
#         print(out.size())
        out = out.view(-1, 1, 96, 96)
        return out
    
    
    def _initialize_weights(self):
        pass