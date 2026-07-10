import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def build_dct_matrix(block_size: int = 8) -> torch.Tensor:
    """构建正交 DCT-II 变换矩阵 (block_size × block_size)"""
    n = block_size
    m = torch.arange(n, dtype=torch.float32)
    k = m.unsqueeze(1)
    dct = torch.cos(math.pi / n * (m + 0.5) * k)
    dct[0] /= math.sqrt(2.0)
    dct *= math.sqrt(2.0 / n)
    return dct  # (n, n), orthogonal: D @ D^T = I


class DCTFrequencyFilter(nn.Module):
    """
    可学习的块状 DCT 频域滤波器。

    将空间残差做 block_size×block_size 的 DCT 分解，
    用可学习的 frequency mask 加权各频带系数，
    再 IDCT 回到空间域。

    初始化时中频带权重较高，DC 和最高频权重较低，
    使水印集中在人眼最不敏感的中频段。
    """

    def __init__(self, block_size: int = 8, init_mid_boost: float = 0.8):
        super().__init__()
        self.block_size = block_size
        dct_mat = build_dct_matrix(block_size)
        self.register_buffer('dct_mat', dct_mat)

        # 初始化频率权重：中频高，DC和高频低
        freq_weight = torch.zeros(block_size, block_size)
        bs = block_size
        for i in range(bs):
            for j in range(bs):
                dist = math.sqrt(i ** 2 + j ** 2)
                max_dist = math.sqrt((bs - 1) ** 2 + (bs - 1) ** 2)
                if dist < 1e-6:
                    freq_weight[i, j] = 0.1  # DC 分量几乎不碰
                else:
                    # 钟形曲线，峰值在中频
                    t = dist / max_dist  # 0 ~ 1
                    freq_weight[i, j] = init_mid_boost * math.sin(math.pi * t)
        self.freq_weight = nn.Parameter(freq_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W), H 和 W 必须被 block_size 整除
        Returns:
            (B, C, H, W) 经过频域滤波的输出
        """
        B, C, H, W = x.shape
        bs = self.block_size
        assert H % bs == 0 and W % bs == 0, \
            f"DCTFilter: H={H}, W={W} 必须被 block_size={bs} 整除"

        # 1. unfold 提取 block_size×block_size 块
        blocks = F.unfold(x, kernel_size=bs, stride=bs)  # (B, C*bs*bs, L)
        L = blocks.shape[2]
        blocks = blocks.view(B, C, bs, bs, L)            # (B, C, bs, bs, L)

        # 2. DCT: D @ block @ D^T，对每个 (B,C,L) 独立
        # 将 B*C*L 合并为 batch 维度
        blocks = blocks.permute(0, 1, 4, 2, 3).contiguous()  # (B, C, L, bs, bs)
        blocks = blocks.view(B * C * L, bs, bs)               # (B*C*L, bs, bs)

        D = self.dct_mat  # (bs, bs)
        blocks_dct = D @ blocks @ D.T                        # (B*C*L, bs, bs)

        # 3. 施加频率遮罩：sigmoid 钳制到 [0,1]
        mask = torch.sigmoid(self.freq_weight)               # (bs, bs)
        blocks_dct = blocks_dct * mask.view(1, bs, bs)

        # 4. IDCT: D^T @ block @ D（D 正交，逆=转置）
        blocks_idct = D.T @ blocks_dct @ D                   # (B*C*L, bs, bs)

        # 5. fold 回原空间
        blocks_idct = blocks_idct.view(B, C, L, bs, bs)
        blocks_idct = blocks_idct.permute(0, 1, 3, 4, 2).contiguous()  # (B, C, bs, bs, L)
        blocks_idct = blocks_idct.reshape(B, C * bs * bs, L)
        x_recon = F.fold(blocks_idct, output_size=(H, W),
                         kernel_size=bs, stride=bs)

        return x_recon


class DCTExtract(nn.Module):
    """
    解码器侧：从水印图中提取 DCT 频域特征。

    对输入做 block-wise DCT，将系数展平后通过一个小型 MLP
    得到频域特征表示，辅助 decoder 主分支解码。
    """

    def __init__(self, block_size: int = 8, in_channels: int = 1,
                 out_features: int = 64):
        super().__init__()
        self.block_size = block_size
        dct_mat = build_dct_matrix(block_size)
        self.register_buffer('dct_mat', dct_mat)
        # 每个 8×8 块有 64 个系数，取有意义的部分
        self.out_features = out_features
        self.fc = nn.Sequential(
            nn.Linear(block_size * block_size, out_features * 2),
            nn.ReLU(inplace=True),
            nn.Linear(out_features * 2, out_features),
        )
        self.in_channels = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, out_features * num_blocks_h * num_blocks_w) DCT 特征
        """
        B, C, H, W = x.shape
        bs = self.block_size
        assert H % bs == 0 and W % bs == 0

        blocks = F.unfold(x, kernel_size=bs, stride=bs)
        L = blocks.shape[2]
        blocks = blocks.view(B, C, bs, bs, L)
        blocks = blocks.permute(0, 1, 4, 2, 3).contiguous()
        blocks = blocks.view(B * C * L, bs, bs)

        D = self.dct_mat
        blocks_dct = D @ blocks @ D.T                         # (B*C*L, bs, bs)
        blocks_dct = blocks_dct.view(B, C * L, bs * bs)       # (B, C*L, 64)

        # 对每个块提取特征
        feat = self.fc(blocks_dct)                            # (B, C*L, out_features)
        feat = feat.view(B, -1)                               # (B, C*L*out_features)

        return feat
