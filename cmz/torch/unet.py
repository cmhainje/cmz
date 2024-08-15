"""
unet.py
author: Connor Hainje

Implementation of the U-Net Convolutional Neural Net.

References:
    - https://arxiv.org/abs/1505.04597
    - https://github.com/milesial/Pytorch-UNet
    - https://github.com/eelregit/map2map
"""


import torch
import torch.nn as nn

from typing import Optional


class DoubleConv(nn.Module):
    """
    Two convolutional layers with optional batch normalization and ReLU activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        mid_channels: Optional[int] = None,
        kernel_size: int = 3,
        skip_norm=False,
        **conv_kw
    ):
        """
        Parameters:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels. Default: number of input channels.
            mid_channels (int): Number of channels in the intermediate layer. Default: number of output channels.
            kernel_size (int): Size of the convolutional kernel. Default: 3.
            skip_norm (bool): Skip normalization layers. Default: False.
            **conv_kw: Additional keyword arguments for the convolutional layers.
        """
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        if mid_channels is None:
            mid_channels = out_channels

        c1 = nn.Conv2d(in_channels, mid_channels, kernel_size, **conv_kw)
        b1 = nn.BatchNorm2d(mid_channels) if not skip_norm else nn.Identity()
        a1 = nn.ReLU()

        c2 = nn.Conv2d(mid_channels, out_channels, kernel_size, **conv_kw)
        b2 = nn.BatchNorm2d(out_channels) if not skip_norm else nn.Identity()
        a2 = nn.ReLU()

        self.net = nn.Sequential(c1, b1, a1, c2, b2, a2)

    def forward(self, x):
        """
        Parameters:
            x (torch.Tensor<N, C_in, H, W>): Input.

        Returns:
            torch.Tensor<N, C_out, H, W>: Output.
        """
        return self.net(x)


class DownBlock(nn.Module):
    """
    Downsampling block of the U-Net.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        Parameters:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super().__init__()

        d = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)
        b = nn.BatchNorm2d(out_channels)
        a = nn.ReLU()

        c = DoubleConv(out_channels, out_channels, padding='same')

        self.net = nn.Sequential(d, b, a, c)

    def forward(self, x):
        """
        Parameters:
            x (torch.Tensor<N, C_in, H, W>): Input.

        Returns:
            torch.Tensor<N, C_out, H/2, W/2>: Downsampled and convolved output.
        """
        return self.net(x)


class UpBlock(nn.Module):
    """
    Upsampling block of the U-Net.

    Given an input of shape (N, C_in, H, W), this layer
    - upsamples the input to shape (N, C_out, 2H, 2W)
    - concatenates with a skip connection of shape (N, C_out, 2H, 2W),
      producing a tensor of shape (N, 2 * C_out, 2H, 2W)
    - convolves the concatenated input to shape (N, C_out, 2H, 2W).
    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        Parameters:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        u = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        b = nn.BatchNorm2d(out_channels)
        a = nn.ReLU()
        self.upsample = nn.Sequential(u, b, a)

        self.c = DoubleConv(2 * out_channels, out_channels, padding='same')

    def forward(self, y, x):
        """
        Parameters:
            y (torch.Tensor<N, C_in, H, W>): Input.
            x (torch.Tensor<N, C_out, 2H, 2W>): Skip connection.

        Returns:
            torch.Tensor<N, C_out, 2H, 2W>: Upsampled and convolved output.
        """
        y = self.upsample(y)
        y = torch.cat((y, x), dim=1)
        return self.c(y)


class UBlock(nn.Module):
    """
    Combined downsampling and upsampling blocks of the U-Net.

    Given an image of shape (N, C_in, H, W), this layer
     - downsamples to (N, C_out, H / 2, W / 2)
     - optionally performs a slotted operation
     - upsamples to (N, C_in, H, W)
     - concatenates with the original input (this is the skip connection)
     - convolves back to shape (N, C_in, H, W).
    """

    def __init__(self, in_channels: int, mid_channels: int):
        """
        Parameters:
            in_channels (int): Number of input channels.
            mid_channels (int): Number of channels for interior operations.
        """
        super().__init__()

        self.in_channels = in_channels
        self.mid_channels = mid_channels

        self.d = DownBlock(in_channels, mid_channels)
        self.u = UpBlock(mid_channels, in_channels)

    def forward(self, x, slot: Optional[nn.Module], *slot_args):
        """
        Parameters:
            x (torch.Tensor<N, C_in, H, W>): Input.
            slot (nn.Module): Slot operation to apply to the downsampled tensor.
                Pass None if no slot operation is needed.
            slot_args: Additional arguments for the slot operation.

        Returns:
            torch.Tensor<N, C_in, H, W>: Output.
        """
        y = self.d(x)
        if slot is not None:
            y = slot(y, *slot_args)
        return self.u(y, x)


class UNet(nn.Module):
    """
    U-Net.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: list[int] = [4, 8, 16],
    ):
        """
        Parameters:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            hidden_channels (list[int]): Number of channels in the hidden layers.
                Must have length >= 2.
        """
        super().__init__()

        if len(hidden_channels) < 2:
            raise ValueError("hidden_channels must have at least 2 elements")

        self.c_in = DoubleConv(in_channels, hidden_channels[0], padding='same')
        self.c_out = nn.Conv2d(hidden_channels[0], out_channels, kernel_size=1)
        self.ublocks = nn.ModuleList([
            UBlock(hidden_channels[i-1], hidden_channels[i])
            for i in range(1, len(hidden_channels))
        ])

    def _ublocks_forward(self, x, i: int):
        """
        Recursively apply the U-Net blocks, slotting each into the previous.

        Parameters:
            x (torch.Tensor<N, C, H, W>): Input.
            i (int): Index of the current U-Net block.

        Returns:
            torch.Tensor<N, C, H, W>: Output.
        """
        if i == len(self.ublocks) - 1:
            return self.ublocks[i](x, None)
        return self.ublocks[i](x, self._ublocks_forward, i + 1)

    def forward(self, x):
        """
        Parameters:
            x (torch.Tensor<N, C_in, H, W>): Input.

        Returns:
            torch.Tensor<N, C_out, H, W>: Output.
        """
        x = self.c_in(x)
        x = self._ublocks_forward(x, 0)
        x = self.c_out(x)
        return x
