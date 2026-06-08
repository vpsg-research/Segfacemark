import torch
import torch.nn as nn
from encoder_256 import spatial_features_extractor
from encoder_256 import Bottleneck
from network.Attention import ResBlock_CBAM
from network.simAM import Simam_module
from config import training_config as cfg
from network.ConvBlock import ConvBlock
import torch.nn.functional as F
import pickle
import numpy as np
import os


class Decoder(nn.Module):
    def __init__(self, face_segmenter, pca_save_path="/root/watermark/train_256_128/pca_bundle.pkl"):
        super(Decoder, self).__init__()

        self.face_segmenter = face_segmenter
        self.watermark_regions = self.face_segmenter.watermark_regions
        self.num_regions = len(self.watermark_regions)
        self.region_indices = [v["index"] for v in self.watermark_regions.values()]
        self.region_weights = torch.tensor([v["weight"] for v in self.watermark_regions.values()], dtype=torch.float32)

        self.pca_save_path = pca_save_path
        self._loaded_pca_path = None
        self.load_pca_model()

        self.feature_extractor = spatial_features_extractor()
        self.attention = ResBlock_CBAM(64, 16)
        self.simam = Simam_module()

        self.b1 = Bottleneck(64, 64)
        self.b2 = Bottleneck(128, 64)
        self.b3 = Bottleneck(192, 64)

        self.down = nn.Sequential(
            nn.InstanceNorm2d(256),
            nn.Tanh(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            
            nn.InstanceNorm2d(256),
            nn.Tanh(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            
            nn.InstanceNorm2d(128),
            nn.Tanh(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),

            nn.InstanceNorm2d(64),
            nn.Tanh(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),

            nn.InstanceNorm2d(64),
            nn.Tanh(),
            nn.Conv2d(64, cfg.wm_channels, kernel_size=3, padding=1),
        )

        self.conv_wm = ConvBlock(cfg.wm_channels, 1, blocks=2)
        self.fc = nn.Linear(cfg.message_length**2, cfg.message_length)

    def load_pca_model(self):
        if self.pca_save_path == self._loaded_pca_path:
            return
        
        assert os.path.exists(self.pca_save_path), f"PCA文件不存在：{self.pca_save_path}"
        with open(self.pca_save_path, 'rb') as f:
            pca_save_dict = pickle.load(f)
        self.pca = pca_save_dict["pca"]
        self.pca_min = pca_save_dict["min"]
        self.pca_max = pca_save_dict["max"]
        self.pca_range = self.pca_max - self.pca_min
        self.pca_range[self.pca_range < 1e-8] = 1e-8
        
        self._loaded_pca_path = self.pca_save_path
        print(f"成功加载PCA模型：{self.pca_save_path}")

    def forward(self, x, masks_scaled):
        B, C, H, W = x.shape

        if H == 256 and W == 256:
            # 256 -> 128 (通道 1 -> 4)
            x_proc = F.pixel_unshuffle(x, downscale_factor=2)
        else:
            x_proc = x

        fm = self.feature_extractor(x_proc)

        o = self.b1(fm)
        o = o + self.simam(o)
        o = self.b2(o)
        o = o + self.simam(o)
        o = self.b3(o)
        o = o + self.simam(o)
        message = self.down(o)
        message = self.conv_wm(message)
        message = F.interpolate(message, size=(cfg.message_length, cfg.message_length), mode="nearest")
        message = message.squeeze(1).view(message.size(0), -1)
        message = self.fc(message)

        regional_wms_list = []
        region_masks_list = []
        pca_input_dim = 136
        mask_resize_size = (8, 17)

        total_wm_len = message.shape[1]
        checksum_len = 16 if total_wm_len == 128 else 8
        
        valid_regions = [
            region_info for region_info in self.face_segmenter.watermark_regions.values()
            if region_info["index"] != 0
        ]

        for region_info in valid_regions:
            region_class_idx = region_info["index"]

            region_binary_mask = (masks_scaled == region_class_idx).float()

            region_mask_resized = F.interpolate(
                region_binary_mask.unsqueeze(1),
                size=mask_resize_size,
                mode='nearest'
            ).squeeze(1)

            region_mask_flat = region_mask_resized.view(B, pca_input_dim).cpu().numpy()
            region_mask_pca = self.pca.transform(region_mask_flat)
            region_mask_norm = (region_mask_pca - self.pca_min) / self.pca_range
            region_mask_norm = np.clip(region_mask_norm, 0.0, 1.0)

            region_mask_norm = np.round(region_mask_norm).astype(np.float32)
            region_mask_tensor = torch.from_numpy(region_mask_norm).float().to(message.device)
            
            msg_checksum = message[:, :checksum_len]
            msg_payload = message[:, checksum_len:]
            
            masked_payload = msg_payload * region_mask_tensor
            
            pure_region_wm = torch.cat([msg_checksum, masked_payload], dim=1)

            full_region_mask = torch.cat([
                torch.ones((B, checksum_len), device=message.device),
                region_mask_tensor
            ], dim=1)

            region_masks_list.append(full_region_mask)
            regional_wms_list.append(pure_region_wm)

        regional_wms = torch.stack(regional_wms_list, dim=1)

        return message, regional_wms, region_masks_list