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
            nn.ConvTranspose2d(
                in_channels=64, out_channels=32, kernel_size=2, stride=2
            ),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1), # parameters
            # 96x96
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
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1), # parameters
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
                in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1), # parameters
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
        
        self.info4 = nn.Sequential(
            # 12x12
            nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1), # parameters
            # 12x12
        )        
        
        self.block4_1 = nn.Sequential(
            # image 12x12
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1
            ),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1), # parameters
            # image 12x12            
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1), # parameters
            # image 6x6 
        )
        
        self.side4_1 = nn.Sequential(
            # image 24x24
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=2, stride=2
            ),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1), # parameters
            # image 12x12
            # padding till here is all correct
        
        )
        
        self.block4_2 = nn.Sequential(
            # image 6x6
            nn.ConvTranspose2d(
                in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1 , output_padding=1
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1), # parameters
            # image 12x12            
            nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1), # parameters
            # image 12x12 
        )
        
        self.side4_2 = nn.Sequential(
            # image 6x6
            nn.ConvTranspose2d(
                in_channels=512, out_channels=256, kernel_size=2, stride=2
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1), # parameters
            # image 12x12
        )
        
        self.block5 = nn.Sequential(
            # image 12x12
            nn.ConvTranspose2d(
                in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1), # parameters
            # image 24x24
            nn.ConvTranspose2d(
                in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1), # parameters  
            # image 24x24
        )
        
        self.side5 = nn.Sequential(
            # image 12x12
            nn.ConvTranspose2d(
                in_channels=256, out_channels=128, kernel_size=2, stride=2
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1), # parameters
            # image 24x24
        )
        
        self.block6 = nn.Sequential(
            # image 24x24
            nn.ConvTranspose2d(
                in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1), # parameters
            # image 48x48
            nn.ConvTranspose2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1), # parameters
            # image 48x48
        )
        
        self.side6 = nn.Sequential(
            # image 24x24
            nn.ConvTranspose2d(
                in_channels=128, out_channels=64, kernel_size=2, stride=2
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1), # parameters
            # image 48x48
        )
        
        self.block7 = nn.Sequential(
            # image 48x48
            nn.ConvTranspose2d(
                in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1), # parameters
            # image 96x96
        )
        
        self.side7 = nn.Sequential(
            # image 48x48
            nn.ConvTranspose2d(
                in_channels=64, out_channels=32, kernel_size=2, stride=2
            ),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1), # parameters
            # image 96x96
        )
        
        self.block8 = nn.Sequential(
            # image 96x96
            nn.Conv2d(
                in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.1), # parameters
            # image 96x96
        )
        
#         self.block9 = nn.Sequential(
#             # image 48x48
#             nn.PixelShuffle(2)
#             # image 96x96           
#         )
        
        
        self.fc = nn.Sequential(
#             nn.PixelShuffle(96/89)
            nn.Sigmoid()
        )
            
        
    def forward(self, x):
        x=x.float()
        out = self.block0(x) # 48x48
        residual1 = out 
        out = self.block1(out) # 48x48 
        out = out + residual1 
        info_1 = self.info1(out) # 96x96
        
        residual2 = out
        out = self.block2(out) # 48x48 
        out = out + residual2
        info_2 = self.info2(out) # 48x48
        
        residual3 = out
        residual3 = self.side3(residual3) # 24x24
        out = self.block3(out) # 24x24
        out = out + residual3
        info_3 = self.info3(out) # 24x24
        
        residual4 = out
        residual4 = self.side4(residual4)
        out = self.block4(out) # 12x12
        out = out + residual4
        info_4 = self.info4(out) # 12x12
        
        residual4_1 = out
        residual4_1 = self.side4_1(residual4_1)
        out = self.block4_1(out) # 6x6        
        out = out + residual4_1
        
        residual4_2 = out
        residual4_2 = self.side4_2(residual4_2)
        out = self.block4_2(out) # 12x12
        out = out + residual4_2 + info_4
        
        residual5 = out
        residual5 = self.side5(residual5) # 24x24
        out = self.block5(out)
        out = out + residual5 + info_3     
        
        residual6 = out
        residual6 = self.side6(residual6)
        out = self.block6(out) # 48x48       
        out = out + residual6 + info_2
        
        residual7 = out
        residual7 = self.side7(residual7) # 96
        out = self.block7(out)
        out = out + residual7 + info_1
        
        out = self.block8(out) # image 96x96, 4D tensor [batch, channel, h, w]
#        batchsize = out.size()[0]
#         print(batchsize)
#        out = out.view(batchsize, -1)
#        out = self.block9(out) # image 96x96, 2D tensor [batch, features]

        out = self.fc(out)
        
#         out = nn.functional.interpolate(out, [96, 96])
        
#         out = F.relu(out) # fully connect sigmoid, image 96x96      
        
#         print(out.size())
#        out = out.view(-1, 1, 96, 96)
        return out
    
    
    def _initialize_weights(self):
        pass