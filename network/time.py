import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import torch
import torch.nn as nn
from network import noise
from encoder import Encoder
from decoder import Decoder
from tabulate import tabulate
from config import training_config as cfg
from utils.Quality import psnr, ssim
from datetime import datetime
from utils.DataLoad_highpass import *
from utils.torch_utils import decoded_message_error_rate_batch
from utils.image_wm_dataset import get_loader
from network.noise import stargan_for_test
import json
from tqdm import tqdm
import warnings
from advanced_gnn import GNNModel
import random
import numpy as np
from segfaceAttention import SegFaceWatermarkAdapter
from fvcore.nn import FlopCountAnalysis, parameter_count_table

warnings.filterwarnings("ignore")

history = []
face_segmenter = SegFaceWatermarkAdapter(cfg.device)
val_loader = get_loader(cfg.val_img_dir, cfg.val_wm_dir, cfg.image_size, cfg.batch_size, shuffle=False)

def seed_torch(seed=42):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

def preprocess(images):
    images_Y = images[:, [0], :, :]
    images_U = images[:, [1], :, :]
    images_V = images[:, [2], :, :]
    return images_Y, images_U, images_V

class IWNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(face_segmenter).to(cfg.device)
        self.decoder_t = Decoder(face_segmenter).to(cfg.device)
        in_dim = 1 * 9
        self.gnn = GNNModel(in_channels=in_dim, hidden_dim=64, out_dim=128).to(cfg.device)
        self.TRAIN_PCA_PATH = cfg.TRAIN_PCA_PATH
        self.EVAL_PCA_PATH = cfg.EVAL_PCA_PATH

    def forward(self, U, cover_images, watermark, mask=None):
        # 确保 U 是单通道，cover_images 是 RGB
        u_embedded, masks_scaled = self.encoder(U, cover_images, watermark)
        wm = self.decoder_t(u_embedded, mask if mask is not None else masks_scaled)
        return wm

    def fit(self, log_dir=False, batch_size=cfg.batch_size, lr=float(cfg.lr), epochs=cfg.epochs):
        if not log_dir:
            log_dir = f'exp_segfacemark_test/{datetime.now().strftime("%Y.%m.%d-%H.%M.%S")}'
        os.makedirs(log_dir, exist_ok=True)

        val = val_loader
        with open(os.path.join(log_dir, "config.json"), "wt") as out:
            out.write(json.dumps(cfg, indent=2, default=lambda o: str(o)))

        # 初始化各种噪声对象
        identity = noise.Identity()
        jpeg = noise.jpeg_compression_train
        resize = noise.Resize()
        medianblur = noise.MedianBlur()
        gau_noise = noise.GaussianNoise()
        gau_blur = noise.GaussianBlur()
        dropout_noise = noise.Dropout()
        salt_pepper_noise = noise.SaltPepper()
        stargan = stargan_for_test.stargan_noise
        ganimation = noise.ganimation_noise
        uniface_swap = noise.unifaceswap_noise
        fsrt = noise.fsrt_noise
        cscs = noise.cscs_noise
        hififace = noise.hififace_noise
        stylemask = noise.stylemask_noise
        infoswap = noise.infoswap_noise

        def decode(u_embedded, decoder, masks_scaled):
            return decoder(u_embedded, masks_scaled)

        def validation_attack(input_list, u_embedded, noise_func, decoder, masks_scaled, type="default"):
            noised = noise_func(input_list, type)
            wm, regional_wms, region_masks = decode(u_embedded + noised, decoder, masks_scaled)
            return wm, regional_wms, region_masks

        def switch_pca_mode(eval_mode):
            self.decoder_t.pca_save_path = self.EVAL_PCA_PATH if eval_mode else self.TRAIN_PCA_PATH
            self.decoder_t.load_pca_model()

        metrics = {k: [] for k in [
            "val_psnr", "val_ssim", 
            "val_jpeg_er_all", "val_resize_er_all", "val_medianblur_er_all", "val_gaublur_er_all", "val_gauNoise_er_all",
            "val_dropout_er_all", "val_saltPepper_er_all", "val_identity_er_all", "val_stargan_er_all", "val_ganimation_er_all",
            "val_unifaceswap_er_all", "val_fsrt_er_all", "val_cscs_er_all", "val_hififace_er_all", "val_stylemask_er_all", "val_infoswap_er_all",
            "val_jpeg_er_region", "val_resize_er_region", "val_medianblur_er_region", "val_gaublur_er_region", "val_gauNoise_er_region",
            "val_dropout_er_region", "val_saltPepper_er_region", "val_identity_er_region", "val_stargan_er_region", "val_ganimation_er_region",
            "val_unifaceswap_er_region", "val_fsrt_er_region", "val_cscs_er_region", "val_hififace_er_region", "val_stylemask_er_region", "val_infoswap_er_region"]}

        self.encoder.eval()
        self.decoder_t.eval()
        switch_pca_mode(eval_mode=True)
        print(f"测试阶段，解码器PCA路径：{self.decoder_t.pca_save_path}")

        iterator = tqdm(val)
        with torch.no_grad():
            for step, (images, wms, mask) in enumerate(iterator):
                cover_images = images.to(cfg.device)
                mask = mask.to(cfg.device)
                watermark = wms.to(cfg.device)
                Y, U, V = preprocess(cover_images)
                u_embedded, masks_scaled = self.encoder(U, cover_images, watermark)
                watermarked_images = torch.cat([Y, u_embedded, V], dim=1)
                input_list = [u_embedded.clone(), watermarked_images.clone(), cover_images.clone(), mask.clone()]
                cover_images_cpu = cover_images.detach().cpu() + 1.0
                embedded_images_cpu = watermarked_images.detach().cpu() + 1.0
                metrics["val_psnr"].append(psnr(cover_images_cpu, embedded_images_cpu))
                metrics["val_ssim"].append(ssim(cover_images_cpu, embedded_images_cpu))

                attacks = [(jpeg, "val_jpeg"), (resize, "val_resize"), (medianblur, "val_medianblur"),
                           (gau_blur, "val_gaublur"), (gau_noise, "val_gauNoise"), (dropout_noise, "val_dropout"),
                           (salt_pepper_noise, "val_saltPepper"), (identity, "val_identity"), (stargan, "val_stargan"),
                           (ganimation, "val_ganimation"), (uniface_swap, "val_unifaceswap"), (fsrt, "val_fsrt"),
                           (cscs, "val_cscs"), (hififace, "val_hififace"), (stylemask, "val_stylemask"), (infoswap, "val_infoswap")]

                for noise_func, name in attacks:
                    wm_all, region_wms, region_masks = validation_attack(input_list, u_embedded, noise_func, self.decoder_t, masks_scaled)
                    metrics[name + "_er_all"].append(decoded_message_error_rate_batch(wm_all, watermark).detach().cpu())
                    metrics[name + "_er_region"].append(decoded_message_error_rate_batch(torch.stack([watermark * m for m in region_masks], dim=1), region_wms).detach().cpu())

        metrics_mean = {k: round(np.mean(v), 7) if len(v) > 0 else "NaN" for k, v in metrics.items()}
        history.append(metrics_mean)
        pd.DataFrame(history).to_csv(os.path.join(log_dir, "metrics.tsv"), index=False, sep="\t")
        with open(os.path.join(log_dir, "metrics.json"), "at") as out:
            out.write(json.dumps(metrics_mean, indent=2, default=lambda o: str(o)))
        return history

if __name__ == "__main__":
    seed_torch(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IWNet().to(device)
    model.eval()

    checkpoint_path = "/root/autodl-tmp/code_bk/exp_segfacemark/2026.03.03-13.48.17/model_state_9.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"], strict=True)
    else:
        model.load_state_dict(checkpoint, strict=True)

    print("模型参数量:")
    print(parameter_count_table(model))

    B, H, W = 1, cfg.image_size, cfg.image_size
    dummy_U = torch.randn(B, 1, H, W).to(device)
    dummy_cover = torch.randn(B, 3, H, W).to(device)
    dummy_watermark = torch.randn(B, cfg.message_length).to(device)
    dummy_mask = torch.randn(B, 1, H, W).to(device)

    flops = FlopCountAnalysis(model, (dummy_U, dummy_cover, dummy_watermark, dummy_mask))
    print("FLOPs: {:.2f} G".format(flops.total() / 1e9))

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    starter.record()
    with torch.no_grad():
        output = model(dummy_U, dummy_cover, dummy_watermark, dummy_mask)
    ender.record()
    torch.cuda.synchronize()
    print("单张图像推理时间: {:.3f} ms".format(starter.elapsed_time(ender)))