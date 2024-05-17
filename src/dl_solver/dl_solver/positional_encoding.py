import numpy as np
import torch
import torch.nn as nn
from einops import rearrange


# Learnable Fourier Features for Multi-Dimensional Spatial Positional Encoding
class LearnableFourierFeatures(nn.Module):
    def __init__(
        self,
        pos_dim: int,
        f_dim: int = 768,
        h_dim: int = 32,
        d_dim: int = 768,
        g_dim: int = 1,
        gamma: float = 1.0,
    ) -> None:
        """
        Initializes the Learnable Fourier Features module.

        Args:
            pos_dim (int): Dimensionality of the input position.
            f_dim (int): Dimensionality of the Fourier features. Default is 768.
            h_dim (int): Dimensionality of the hidden layer in the MLP. Default is 32.
            d_dim (int): Dimensionality of the output embedding. Default is 768.
            g_dim (int): Number of positional groups. Default is 1.
            gamma (float): Variance scaling factor for initializing the Fourier feature weight matrix. Default is 1.0.
        """
        super(LearnableFourierFeatures, self).__init__()
        assert (
            f_dim % 2 == 0
        ), "number of fourier feature dimensions must be divisible by 2."
        assert (
            d_dim % g_dim == 0
        ), "number of D dimension must be divisible by the number of G dimension."
        enc_f_dim = int(f_dim / 2)
        dg_dim = int(d_dim / g_dim)
        self.Wr = nn.Parameter(torch.randn([enc_f_dim, pos_dim]) * (gamma**2))
        self.mlp = nn.Sequential(
            nn.Linear(f_dim, h_dim), nn.GELU(), nn.Linear(h_dim, dg_dim)
        )
        self.div_term = np.sqrt(f_dim)

    def forward(self, pos):
        """
        Forward pass for the Learnable Fourier Features module.

        Args:
            pos (Tensor): Input tensor with shape (B, L, G, M), where
                B is the batch size,
                L is the sequence length,
                G is the number of positional groups,
                M is the dimensionality of the position.

        Returns:
            Tensor: Output tensor with shape (B, L, D), where
                D is the dimensionality of the output embedding.
        """
        # input pos dim: (B L G M)
        # output dim: (B L D)
        # L stands for sequence length. all dimensions must be flattened to a single dimension.
        XWr = torch.matmul(pos, self.Wr.T)
        F = torch.cat([torch.cos(XWr), torch.sin(XWr)], dim=-1) / self.div_term
        Y = self.mlp(F)
        pos_enc = rearrange(Y, "b l g d -> b l (g d)")

        return pos_enc


# Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains
class FourierFeatures(nn.Module):
    def __init__(self, pos_dim, f_dim, sigma=10, train=False):
        super(FourierFeatures, self).__init__()
        assert f_dim % 2 == 0, "number of channels must be divisible by 2."
        enc_dim = int(f_dim / 2)
        self.B = torch.randn([pos_dim, enc_dim]) * sigma
        if train:
            self.B = nn.Parameter(self.B)

    def forward(self, pos):
        # pos: (B L C), (B H W C), (B H W T C)
        pos_enc = torch.matmul(pos, self.B.to(pos.device))
        pos_enc = torch.cat([torch.sin(pos_enc), torch.cos(pos_enc)], dim=-1)
        return pos_enc


# Attention is All You Need
class PositionalEncoding(nn.Module):
    def __init__(self, pos_dim, enc_dim):
        super(PositionalEncoding, self).__init__()
        assert (
            enc_dim % (pos_dim * 2) == 0
        ), "dimension of positional encoding must be equal to dim * 2."
        enc_dim = int(enc_dim / 2)
        div_term = torch.exp(
            torch.arange(0.0, enc_dim, 2) * -(np.log(10000.0) / enc_dim)
        )
        freqs = torch.zeros([pos_dim, enc_dim])
        for i in range(pos_dim):
            freqs[i, : enc_dim // 2] = div_term
            freqs[i, enc_dim // 2 :] = div_term
        self.freqs = freqs

    def forward(self, pos):
        # pos: (B L C), (B H W C), (B H W T C)
        pos_enc = torch.matmul(pos, self.freqs.to(pos.device))
        pos_enc = torch.cat([torch.sin(pos_enc), torch.cos(pos_enc)], dim=-1)
        return pos_enc


if __name__ == "__main__":
    """
    example usage of LearnableFourierFeatures

    let
    positional dimension: 2 (2d spatial positions)
    fourier feature dimension: 128
    hidden dimension: 256
    positional encoding dimension: 64
    number of positional groups: 1

    batch size: 4
    sequence length: 1024 (== 32x32 in 2d spatial resolution)
    number of positional groups: 1
    positional dimension: 2
    """
    lff = LearnableFourierFeatures(
        pos_dim=2, f_dim=128, h_dim=256, d_dim=64, g_dim=1
    ).cuda()
    pos = torch.randn([4, 1024, 1, 2]).cuda()
    pe = lff(pos)
    print(pe)
    print(pe.shape)

    """
    example usage of FourierFeatures

    let
    positional dimension: 2 (2d spatial positions)
    fourier feature dimension: 256

    batch size: 4
    sequence length: 32x32
    positional dimension: 2
    """
    ff = FourierFeatures(pos_dim=2, f_dim=256).cuda()
    pos = torch.randn([4, 32, 32, 2]).cuda()
    pe = ff(pos)
    print(pe)
    print(pe.shape)

    """
    example usage of PositionalEncoding

    let
    positional dimension: 2 (2d spatial positions)
    encoding dimension: 256

    batch size: 4
    sequence length: 1024
    positional dimension: 2
    """
    PE = PositionalEncoding(pos_dim=2, enc_dim=256).cuda()
    pos = torch.randn([4, 1024, 2]).cuda()
    pe = PE(pos)
    print(pe)
    print(pe.shape)
