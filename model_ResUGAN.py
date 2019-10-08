import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchvision.models import vgg19
import math
# ResU-GAN

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:18])

    def forward(self, img):
        return self.feature_extractor(img)  

class GeneratorResUNet(nn.Module):
    def __init__(self):
        super(GeneratorResUNet, self).__init__()
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
        self.skip0 = SkipBlock(128, 128) # image 48
        self.block4 = ResBlock(128, 256) # image 24
        self.block5 = ResBlock(256, 256) # image 24
        self.skip1 = SkipBlock(256, 256) # image 24
        self.block6 = ResBlock(256, 512) # image 12
        self.block7 = ResBlock(512, 512) # image 12
        self.block8 = ResBlock(512, 512) # image 12
        self.block9 = UpResBlock(512, 256) # image 24
        self.block10 = ResBlock(256, 256) # image 24
        self.block11 = ResBlock(256, 256) # image 24
        self.block12 = UpResBlock(256, 128) # image 48
        self.block13 = ResBlock(128, 128) # image 48
        self.block14 = ResBlock(128, 128) # image 48
        self.block15 = UpResBlock(128, 64) # image 96
        self.block16 = ResBlock(64, 64) # image 96
        self.block17 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1), # parameters
            nn.Conv2d(
                in_channels=64, out_channels=4, kernel_size=3, stride=2, padding=1
            ),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.1), # parameters
            nn.PixelShuffle(2),
            nn.Sigmoid()
        ) 
        
            
        
    def forward(self, x):
        x=x.float()
        out = self.block(x)
        out = self.block0(out) # 64 channels
        out = self.block1(out) # 64 channels
#         print(out.size()[1])
        out = self.block2(out)
        out = self.block3(out)
        info0 = self.skip0(out)
        out = self.block4(out)
        out = self.block5(out)
        info1 = self.skip1(out)
        out = self.block6(out)
        out = self.block7(out)
        out = self.block8(out)
        out = self.block9(out)
        out = out + info1
        out = self.block10(out)
        out = self.block11(out)
        out = self.block12(out)
        out = self.block13(out)
        out = out + info0
        out = self.block14(out)
        out = self.block15(out)
        out = self.block16(out)
        out = self.block17(out)
        return out
    
    
    def _initialize_weights(self):
        pass

class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)
    

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
#             nn.Conv2d(
#                 in_channels=out_features, out_channels=out_features, kernel_size=1, stride=1
#             ),
#             nn.BatchNorm2d(out_features),
#             nn.LeakyReLU(0.1), # parameters
            nn.Conv2d(
                in_channels=in_features, out_channels=out_features, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(out_features),
            nn.LeakyReLU(0.1), # parameters
#             nn.Conv2d(
#                 in_channels=out_features, out_channels=out_features, kernel_size=3, stride=1, padding=1
#             ),
#             nn.BatchNorm2d(out_features),
#             nn.LeakyReLU(0.1), # parameters
            nn.Conv2d(
                in_channels=out_features, out_channels=out_features, kernel_size=1, stride=1
            ),
            nn.BatchNorm2d(out_features),
            nn.LeakyReLU(0.1) # parameters
        )

    def forward(self, x):
        return self.skip(x)
    


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           U-NET
##############################


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [
            nn.Conv2d(in_size, out_size, 4, 2, 1),
            nn.BatchNorm2d(out_size),
            nn.LeakyReLU(0.1),
#             nn.Conv2d(out_size, out_size, 3, 1, 1, bias=False),
#             nn.BatchNorm2d(out_size),
#             nn.LeakyReLU(0.1),
            nn.Conv2d(out_size, out_size, 1, 1, bias=False),
        ]
        if normalize:
            layers.append(nn.BatchNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.1))
        if dropout:
            layers.append(nn.Dropout(dropout))
        side = [
            nn.Conv2d(in_size, out_size, 2, 2, bias=False),
        ]
        self.model = nn.Sequential(*layers)
        self.side = nn.Sequential(*side)

    def forward(self, x):
        x = self.model(x) + self.side(x)
        return x


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.Conv2d(in_size, out_size, 3, 1, 1),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_size, out_size, 1, 1, bias=False),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True),
#             nn.Conv2d(out_size, out_size, 3, 1, 1, bias=False),
#             nn.BatchNorm2d(out_size),
#             nn.ReLU(inplace=True),
        ]
        side = [
            nn.Conv2d(in_size, out_size, 1, 1, bias=False),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)
        self.side = nn.Sequential(*side)

    def forward(self, x, skip_input):
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = self.model(x) + self.side(x)
        x = torch.cat((x, skip_input), 1)

        return x


class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(GeneratorUNet, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.skip1 = SkipBlock(64, 64)
        self.down2 = UNetDown(64, 128)
        self.skip2 = SkipBlock(128, 128)
#         self.Res1 = ResBlock(128, 128)
        self.down3 = UNetDown(128, 256)
        self.skip3 = SkipBlock(256, 256)
        self.down4 = UNetDown(256, 512)
        self.skip4 = SkipBlock(512, 512)
#         self.Res2 = ResBlock(512, 512)
        self.down5 = UNetDown(512, 512)
        self.skip5 = SkipBlock(512, 512)
        self.down6 = UNetDown(512, 512)

        self.up1 = UNetUp(512, 512)
        self.up2 = UNetUp(1024, 512)
#         self.Res3 = ResBlock(1024, 1024)
        self.up3 = UNetUp(1024, 256)
        self.up4 = UNetUp(512, 128)
#         self.Res4 = ResBlock(256, 256)
        self.up5 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Conv2d(128, 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.Sigmoid()
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
#         d2 = self.Res1(d2)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
#         d4 = self.Res2(d4)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        u1 = self.up1(d6, self.skip5(d5))
        u2 = self.up2(u1, self.skip4(d4))
#         u2 = self.Res3(u2)
        u3 = self.up3(u2, self.skip3(d3))
        u4 = self.up4(u3, self.skip2(d2))
#         u4 = self.Res4(u4)
        u5 = self.up5(u4, self.skip1(d1))


        return self.final(u5)
    
##############################
#        Discriminator
##############################


class DiscriminatorPix(nn.Module):
    def __init__(self, in_channels=3):
        super(DiscriminatorPix, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [
                nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1),
                nn.BatchNorm2d(out_filters),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(out_filters, out_filters, 3, stride=1, padding=1),
            ]
            if normalization:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.1, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)
    