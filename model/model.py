import torch
import torch.nn as nn


def default_conv(in_channels, out_channels, kernel_size, bias):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias
    )


class UpConv(nn.Module):
    def __init__(self):
        super(UpConv, self).__init__()
        self.body = nn.Sequential(
            default_conv(3, 12, 3, True),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.body(x)


class ResidualBlock(nn.Module):
    def __init__(self, n_feats):
        super(ResidualBlock, self).__init__()
        modules_body = [
            default_conv(n_feats, n_feats, 3, bias=True),
            nn.ReLU(inplace=True),
            default_conv(n_feats, n_feats, 3, bias=True)
        ]
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class SingleScaleNet(nn.Module):
    def __init__(self, n_feats, n_resblocks, is_skip, n_channels=3):
        super(SingleScaleNet, self).__init__()
        self.is_skip = is_skip

        modules_head = [
            default_conv(n_channels, n_feats, 5, bias=True),
            nn.ReLU(inplace=True)]

        modules_body = [
            ResidualBlock(n_feats)
            for _ in range(n_resblocks)
        ]

        modules_tail = [default_conv(n_feats, 3, 5, bias=True)]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.head(x)

        res = self.body(x)
        if self.is_skip:
            res += x

        res = self.tail(res)

        return res


class MultiScaleNet(nn.Module):
    def __init__(self, n_feats, n_resblocks, is_skip):
        super(MultiScaleNet, self).__init__()

        self.scale3_net = SingleScaleNet(n_feats, n_resblocks, is_skip, n_channels=3)
        self.upconv3 = UpConv()

        self.scale2_net = SingleScaleNet(n_feats, n_resblocks, is_skip, n_channels=6)
        self.upconv2 = UpConv()

        self.scale1_net = SingleScaleNet(n_feats, n_resblocks, is_skip, n_channels=6)

    def forward(self, mulscale_input):
        input_b1, input_b2, input_b3 = mulscale_input

        output_l3 = self.scale3_net(input_b3)
        output_l3_up = self.upconv3(output_l3)

        output_l2 = self.scale2_net(torch.cat((input_b2, output_l3_up), 1))
        output_l2_up = self.upconv2(output_l2)

        output_l1 = self.scale2_net(torch.cat((input_b1, output_l2_up), 1))

        return output_l1, output_l2, output_l3