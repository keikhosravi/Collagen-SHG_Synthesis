import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
# ResU

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
#         self.x = x
        self.block = nn.Sequential(
            # input image 96x96
            nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=3
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1), # parameters
        ) # 96x96
        
        self.block0 = ResBlock(64, 64) # image 96
        self.block1 = ResBlock(64, 64) # image 96
        self.block2 = ResBlock(64, 128) # image 48
        self.block3 = ResBlock(128, 128) # image 48
        self.block4 = ResBlock(128, 128) # image 48
        self.skip0 = SkipBlock(128, 128) # image 48
        self.block5 = ResBlock(128, 256) # image 24
        self.block6 = ResBlock(256, 256) # image 24
        self.block7 = ResBlock(256, 256) # image 24
        self.skip1 = SkipBlock(256, 256) # image 24
        self.block8 = ResBlock(256, 512) # image 12
        self.block9 = ResBlock(512, 512) # image 12
        self.block10 = ResBlock(512, 512) # image 12
        self.block11 = UpResBlock(512, 256) # image 24
        self.block12 = ResBlock(256, 256) # image 24
        self.block13 = ResBlock(256, 256) # image 24
        self.block14 = UpResBlock(256, 128) # image 48
        self.block15 = ResBlock(128, 128) # image 48
        self.block16 = ResBlock(128, 128) # image 48
        self.block17 = UpResBlock(128, 64) # image 96
        self.block18 = ResBlock(64, 64) # image 96
        self.block19 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=7, stride=1, padding=3
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1), # parameters
            nn.Conv2d(
                in_channels=128, out_channels=1, kernel_size=1, stride=1
            ),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.1), # parameters
        ) 
        
        self.fc = nn.Sequential(
#             nn.PixelShuffle(96/89)
            nn.Sigmoid()
        )
            
        
    def forward(self, x):
        x=x.float()
        out = self.block(x) # 64 channels
        out = self.block1(out) # 64 channels
#         print(out.size()[1])
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        info0 = self.skip0(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.block7(out)
        info1 = self.skip1(out)
        out = self.block8(out)
        out = self.block9(out)
        out = self.block10(out)
        out = self.block11(out)
        out = out + info1
        out = self.block12(out)
        out = self.block13(out)
        out = self.block14(out)
        out = out + info0
        out = self.block15(out)
        out = self.block16(out)
        out = self.block17(out)
        out = self.block18(out)
        out = self.block19(out)
        out = self.fc(out)
        return out
    
    
    def _initialize_weights(self):
        pass
     

class ResBlock(nn.Module):
    def __init__(self, in_features, out_features): # out = in, same features; out = in/2, double features
        super(ResBlock, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.side = nn.Sequential(
            nn.Conv2d(
                in_channels=in_features, out_channels=out_features, kernel_size=1, stride=2
            ),
            nn.BatchNorm2d(out_features),
            nn.LeakyReLU(0.1) # parameters
        )
        self.conv_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_features, out_channels=out_features, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(out_features),
            nn.LeakyReLU(0.1), # parameters
            nn.Conv2d(
                in_channels=out_features, out_channels=out_features, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(out_features),
            nn.LeakyReLU(0.1) # parameters
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_features, out_channels=out_features, kernel_size=1, stride=1
            ),
            nn.BatchNorm2d(out_features),
            nn.LeakyReLU(0.1), # parameters
            nn.Conv2d(
                in_channels=out_features, out_channels=out_features, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(out_features),
            nn.LeakyReLU(0.1), # parameters
            nn.Conv2d(
                in_channels=in_features, out_channels=out_features, kernel_size=1, stride=1
            ),
            nn.BatchNorm2d(out_features),
            nn.LeakyReLU(0.1), # parameters
            nn.MaxPool2d(2, stride=2)
        )

    def forward(self, x):
        if self.in_features == self.out_features:
            return x + self.conv_1(x)
        else:
            x1 = self.side(x)
#             print(x1.size()[1])
            x2 = self.side(x)
#             print(x2.size()[1])
            return x1 + x2


class UpResBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(UpResBlock, self).__init__()
        self.side = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_features, out_channels=out_features, kernel_size=2, stride=2
            ),
            nn.BatchNorm2d(out_features),
            nn.LeakyReLU(0.1) # parameters
        )
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_features, out_channels=out_features, kernel_size=2, stride=2,
            ),
            nn.BatchNorm2d(out_features),
            nn.LeakyReLU(0.1), # parameters
            nn.Conv2d(
                in_channels=out_features, out_channels=out_features, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(out_features),
            nn.LeakyReLU(0.1), # parameters
            nn.Conv2d(
                in_channels=out_features, out_channels=out_features, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(out_features),
            nn.LeakyReLU(0.1) # parameters
        )

    def forward(self, x):
        x1 = self.side(x)
        x2 = self.conv(x)
        return x1 + x2


class SkipBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(SkipBlock, self).__init__()
        self.skip = nn.Sequential(
            nn.Conv2d(
                in_channels=out_features, out_channels=out_features, kernel_size=1, stride=1
            ),
            nn.BatchNorm2d(out_features),
            nn.LeakyReLU(0.1), # parameters
            nn.Conv2d(
                in_channels=in_features, out_channels=out_features, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(out_features),
            nn.LeakyReLU(0.1), # parameters
            nn.Conv2d(
                in_channels=out_features, out_channels=out_features, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(out_features),
            nn.LeakyReLU(0.1), # parameters
            nn.Conv2d(
                in_channels=out_features, out_channels=out_features, kernel_size=1, stride=1
            ),
            nn.BatchNorm2d(out_features),
            nn.LeakyReLU(0.1) # parameters
        )

    def forward(self, x):
        return self.skip(x)
    