import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
# 

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
#         self.x = x
        self.block0 = nn.Sequential(
            # input image 96x96
            nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=11, stride=1, padding=5
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1), # parameters
            # image 96x96
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1), # parameters
            nn.MaxPool2d(2, stride=2, padding=0)
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
            # 48x48
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1), # parameters
            # 48x48
            nn.Conv2d(
                in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(64),
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
        
        self.block3 = nn.Sequential(
            # image 48x48
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1), # parameters
            nn.MaxPool2d(2, stride=2, padding=0),
            # image 24x24
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1), # parameters
            # image 24x24
        )
        
        self.info2 = nn.Sequential(
            # 24x24
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1), # parameters
            nn.Conv2d(
                in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1), # parameters
        )
        
        self.side3 = nn.Sequential(
            # image 48x48
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=1, stride=2
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1), # parameters
            # image 24x24
        )
        
        self.block4 = nn.Sequential(
            # image 24x24
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1), # parameters
            nn.MaxPool2d(2, stride=2, padding=0),
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
                in_channels=128, out_channels=256, kernel_size=1, stride=2
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1), # parameters
            # image 12x12
            # padding till here is all correct
        
        )
        
        self.block5 = nn.Sequential(
            # image 12x12
            nn.ConvTranspose2d(
                in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1
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
        
        self.side5 = nn.Sequential(
            # image 12x12
            nn.ConvTranspose2d(
                in_channels=256, out_channels=128, kernel_size=1, stride=2, output_padding=1
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
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1), # parameters
            # image 48x48
        )
        
        self.side6 = nn.Sequential(
            # image 24x24
            nn.ConvTranspose2d(
                in_channels=128, out_channels=64, kernel_size=1, stride=2, output_padding=1
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
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1), # parameters
        )
        
        self.block8 = nn.Sequential(
            # image 96x96
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=11, stride=1, padding=5
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1), # parameters
            # image 48x48
        )
        
        self.block9 = nn.Sequential(
            # image 48x48
#             nn.PixelShuffle(2)
            # image 96x96
            nn.Conv2d(
                in_channels=64, out_channels=4, kernel_size=1, stride=1
            ),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.1), # parameters
            nn.PixelShuffle(2),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.1), # parameters
        )
        
        
        self.fc = nn.Sequential(
#             nn.PixelShuffle(96/89)
            nn.Sigmoid()
        )
            
        
    def forward(self, x):
        x=x.float()
        out = self.block0(x) # save block0 output, 64 channels, 48x48 image
        residual1 = out # save block0 output as residual, 64 channels, 48x48 image
        out = self.block1(out) # run block1, 64 channels, 48x48 image
        info_1 = self.info1(out) # transfer information to save level decoder
        out = out + residual1 # add residuak to output
        
        residual2 = out # 
        out = self.block2(out) # 64 channels, 48x48 image
        out = out + residual2
        
        residual3 = out
        residual3 = self.side3(residual3) # residual need to be conv in order to add
        out = self.block3(out) # 128 channels, 24x24 image
        info_2 = self.info2(out)
        out = out + residual3
        
        residual4 = out
        residual4 = self.side4(residual4)
        out = self.block4(out)
        out = out + residual4
        
        residual5 = out
        residual5 = self.side5(residual5)
        out = self.block5(out)
        out = out + residual5 + info_2 # recieve info_2
        
        residual6 = out
        residual6 = self.side6(residual6)
        out = self.block6(out)
        out = out + residual6 + info_1 # recieve info_1
        
        out = self.block7(out)
        
        out = self.block8(out) # image 96x96, 4D tensor [batch, channel, h, w]
#        batchsize = out.size()[0]
#         print(batchsize)
#        out = out.view(batchsize, -1)
        out = self.block9(out) # image 96x96, 2D tensor [batch, features]

        out = self.fc(out)
        out = F.interpolate(out, mode='nearest', scale_factor=0.5)
        
#         out = nn.functional.interpolate(out, [96, 96])
        
#         out = F.relu(out) # fully connect sigmoid, image 96x96      
        
#         print(out.size())
#         out = out.view(-1, 1, 96, 96)
        return out
    
    
    def _initialize_weights(self):
        pass