import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torchvision.models import vgg19

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Simple model
class SimpleModel(nn.Module):
    '''
    Simple model for pipeline testing which takes tensor images
    Tensor images are expected as: (B x C x H x W)
    '''
    def __init__(self):       
        super(SimpleModel, self).__init__()
        self.up = nn.Upsample(scale_factor=10, mode='nearest')
        self.conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=(1, 1))

    def forward(self, x):
        x = self.up(x)
        x = F.relu(self.conv(x))
        return x


# SRCNN
class SRCNN(nn.Module):
    '''
    SRCNN model for pipeline testing which takes tensor images and upsamples by x times
    Tensor images are expected as: (B x C x H x W)
    '''
    def __init__(self):       
        super(SRCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, stride=(1, 1), padding=(2, 2))
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, stride=(1, 1), padding=(2, 2))
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, stride=(1, 1), padding=(2, 2))
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)

        return x
    


#  ------ Code from LapSRNN -----------------------
def get_upsample_filter(size: int) -> torch.Tensor:
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    bilinear_filter = torch.from_numpy((1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)).float()

    return bilinear_filter


class ConvLayer(nn.Module):
    def __init__(self, channels: int) -> None:
        super(ConvLayer, self).__init__()
        self.cl = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.cl(x)

        return out


class LapSRN(nn.Module):
    def __init__(self) -> None:
        super(LapSRN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True),
        )

        # Scale 2
        laplacian_pyramid_conv1 = []
        for _ in range(10):
            laplacian_pyramid_conv1.append(ConvLayer(64))
        laplacian_pyramid_conv1.append(nn.ConvTranspose2d(64, 64, (4, 4), (2, 2), (1, 1)))
        laplacian_pyramid_conv1.append(nn.LeakyReLU(0.2, True))

        self.laplacian_pyramid_conv1 = nn.Sequential(*laplacian_pyramid_conv1)
        self.laplacian_pyramid_conv2 = nn.ConvTranspose2d(3, 3, (4, 4), (2, 2), (1, 1))
        self.laplacian_pyramid_conv3 = nn.Conv2d(64, 3, (3, 3), (1, 1), (1, 1))

        # Scale 4
        laplacian_pyramid_conv4 = []
        for _ in range(10):
            laplacian_pyramid_conv4.append(ConvLayer(64))
        laplacian_pyramid_conv4.append(nn.ConvTranspose2d(64, 64, (4, 4), (2, 2), (1, 1)))
        laplacian_pyramid_conv4.append(nn.LeakyReLU(0.2, True))

        self.laplacian_pyramid_conv4 = nn.Sequential(*laplacian_pyramid_conv4)
        self.laplacian_pyramid_conv5 = nn.ConvTranspose2d(3, 3, (4, 4), (2, 2), (1, 1))
        self.laplacian_pyramid_conv6 = nn.Conv2d(64, 3, (3, 3), (1, 1), (1, 1))

        # Scale 8
        laplacian_pyramid_conv7 = []
        for _ in range(10):
            laplacian_pyramid_conv7.append(ConvLayer(64))
        laplacian_pyramid_conv7.append(nn.ConvTranspose2d(64, 64, (4, 4), (2, 2), (1, 1)))
        laplacian_pyramid_conv7.append(nn.LeakyReLU(0.2, True))

        self.laplacian_pyramid_conv7 = nn.Sequential(*laplacian_pyramid_conv7)
        self.laplacian_pyramid_conv8 = nn.ConvTranspose2d(3, 3, (4, 4), (2, 2), (1, 1))
        self.laplacian_pyramid_conv9 = nn.Conv2d(64, 3, (3, 3), (1, 1), (1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)

        # X2
        lpc1 = self.laplacian_pyramid_conv1(out)
        lpc2 = self.laplacian_pyramid_conv2(x)
        lpc3 = self.laplacian_pyramid_conv3(lpc1)
        out1 = lpc2 + lpc3
        # X4
        lpc4 = self.laplacian_pyramid_conv4(lpc1)
        lpc5 = self.laplacian_pyramid_conv5(out1)
        lpc6 = self.laplacian_pyramid_conv6(lpc4)
        out2 = lpc5 + lpc6
        # X8
        lpc7 = self.laplacian_pyramid_conv7(lpc4)
        lpc8 = self.laplacian_pyramid_conv8(out2)
        lpc9 = self.laplacian_pyramid_conv9(lpc7)
        out3 = lpc8 + lpc9

        return out1, out2, out3

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            if isinstance(module, nn.ConvTranspose2d):
                c1, c2, h, w = module.weight.data.size()
                weight = get_upsample_filter(h)
                module.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if module.bias is not None:
                    module.bias.data.zero_()


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.mean(torch.sqrt(torch.pow((target - inputs), 2) + self.eps))

        return loss


# SRGAN 

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        discriminator=False,
        use_act=True,
        use_bn=True,
        **kwargs,
    ):
        super().__init__()
        self.use_act = use_act
        self.cnn = nn.Conv2d(in_channels, out_channels, **kwargs, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act = (
            nn.LeakyReLU(0.2, inplace=True)
            if discriminator
            else nn.PReLU(num_parameters=out_channels)
        )

    def forward(self, x):
        return self.act(self.bn(self.cnn(x))) if self.use_act else self.bn(self.cnn(x))


class UpsampleBlock(nn.Module):
    def __init__(self, in_c, scale_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_c, in_c * scale_factor ** 2, 3, 1, 1)
        self.ps = nn.PixelShuffle(scale_factor)  # in_c * 4, H, W --> in_c, H*2, W*2
        self.act = nn.PReLU(num_parameters=in_c)

    def forward(self, x):
        return self.act(self.ps(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.block1 = ConvBlock(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.block2 = ConvBlock(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            use_act=False,
        )

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        return out + x


class Generator(nn.Module):
    def __init__(self, in_channels=3, num_channels=64, num_blocks=16):
        super().__init__()
        self.initial = ConvBlock(in_channels, num_channels, kernel_size=9, stride=1, padding=4, use_bn=False)
        self.residuals = nn.Sequential(*[ResidualBlock(num_channels) for _ in range(num_blocks)])
        self.convblock = ConvBlock(num_channels, num_channels, kernel_size=3, stride=1, padding=1, use_act=False)
        self.upsamples = nn.Sequential(UpsampleBlock(num_channels, 2), UpsampleBlock(num_channels, 2))
        self.final = nn.Conv2d(num_channels, in_channels, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        initial = self.initial(x)
        x = self.residuals(initial)
        x = self.convblock(x) + initial
        x = self.upsamples(x)
        return torch.tanh(self.final(x))


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 64, 128, 128, 256, 256, 512, 512]):
        super().__init__()
        blocks = []
        for idx, feature in enumerate(features):
            blocks.append(
                ConvBlock(
                    in_channels,
                    feature,
                    kernel_size=3,
                    stride=1 + idx % 2,
                    padding=1,
                    discriminator=True,
                    use_act=True,
                    use_bn=False if idx == 0 else True,
                )
            )
            in_channels = feature

        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(512*6*6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        x = self.blocks(x)
        return self.classifier(x)


class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = vgg19(pretrained=True).features[:36].eval().to(DEVICE)
        self.loss = nn.MSELoss()

        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        vgg_input_features = self.vgg(input)
        vgg_target_features = self.vgg(target)
        return self.loss(vgg_input_features, vgg_target_features)