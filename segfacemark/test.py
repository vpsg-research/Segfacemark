import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import torch
import torch.nn as nn
from network import noise
# from segfacemark.encoder_256 import Encoder
# from segfacemark.decoder_256 import Decoder
from segfacemark.encoder import Encoder
from segfacemark.decoder import Decoder
from tabulate import tabulate
from config import training_config as cfg
from utils.Quality import psnr, ssim
from datetime import datetime
from utils.DataLoad_highpass import *
from utils.torch_utils import decoded_message_error_rate_batch
from utils.image_wm_dataset import get_loader
import json
from tqdm import tqdm
import warnings
from advanced_gnn import GNNModel
import random
import numpy as np
from segfaceAttention import SegFaceWatermarkAdapter
warnings.filterwarnings("ignore")

history = []

face_segmenter = SegFaceWatermarkAdapter(cfg.device)

val_loader = get_loader(
    cfg.val_img_dir, cfg.val_wm_dir,
    cfg.image_size, cfg.batch_size, shuffle=False
)


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
    images_R = images[:, [0], :, :]
    images_G = images[:, [1], :, :]
    images_B = images[:, [2], :, :]
    return images_R, images_G, images_B

class IWNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(face_segmenter).to(cfg.device)
        self.decoder = Decoder(face_segmenter).to(cfg.device)
        in_dim = 1 * 9
        self.gnn = GNNModel(in_channels=in_dim, hidden_dim=64, out_dim=128).to(cfg.device)

        self.TRAIN_PCA_PATH = cfg.TRAIN_PCA_PATH
        self.EVAL_PCA_PATH = cfg.EVAL_PCA_PATH

    def fit(self, log_dir=False, ckpt_path=None):
        if not log_dir:
            log_dir = f'exp_segfacemark_test/{(datetime.now().strftime("%Y.%m.%d-%H.%M.%S"))}'
        os.makedirs(log_dir, exist_ok=True)

        val = val_loader

        with open(os.path.join(log_dir, "config.json"), "wt") as out:
            out.write(json.dumps(cfg, indent=2, default=lambda o: str(o)))

        identity = noise.Identity()
        jpeg = noise.jpeg_compression_train
        resize = noise.Resize()
        medianblur = noise.MedianBlur()
        gau_noise = noise.GaussianNoise()
        gau_blur = noise.GaussianBlur()
        dropout_noise = noise.Dropout()
        salt_pepper_noise = noise.SaltPepper()
        stargan = noise.stargan_noise
        ganimation = noise.ganimation_noise
        cscs = noise.cscs_noise
        hififace = noise.hififace_noise
        RAFSwap = noise.RAFSwap_noise
        infoswap = noise.infoswap_noise
        uniface_swap = noise.unifaceswap_noise

        def validation_attack(input, u_embedded, noise_fn, decoder, masks_scaled, type="default"):
            noised = noise_fn(input, type)
            wm, regional_wms, region_masks = decoder(u_embedded + noised, masks_scaled)
            return wm, regional_wms, region_masks

        def switch_pca_mode(eval_mode):
            if eval_mode:
                self.decoder.pca_save_path = self.EVAL_PCA_PATH
            else:
                self.decoder.pca_save_path = self.TRAIN_PCA_PATH
            self.decoder.load_pca_model()

        metrics = {
            "val_psnr": [],
            "val_ssim": [],
            "val_jpeg_er_all": [],
            "val_resize_er_all": [],
            "val_medianblur_er_all": [],
            "val_gaublur_er_all": [],
            "val_gauNoise_er_all": [],
            "val_dropout_er_all": [],
            "val_saltPepper_er_all": [],
            "val_identity_er_all": [],
            "val_stargan_er_all": [],
            "val_ganimation_er_all": [],
            "val_cscs_er_all": [],
            "val_hififace_er_all": [],
            "val_RAFSwap_er_all": [],
            "val_infoswap_er_all": [],
            "val_unifaceswap_er_all": [],

            "val_jpeg_er_region": [],
            "val_resize_er_region": [],
            "val_medianblur_er_region": [],
            "val_gaublur_er_region": [],
            "val_gauNoise_er_region": [],
            "val_dropout_er_region": [],
            "val_saltPepper_er_region": [],
            "val_identity_er_region": [],
            "val_stargan_er_region": [],
            "val_ganimation_er_region": [],
            "val_cscs_er_region": [],
            "val_hififace_er_region": [],
            "val_RAFSwap_er_region": [],
            "val_infoswap_er_region": [],
            "val_unifaceswap_er_region": [],
        }
        data_robust = []
        data_vis = []

        self.encoder.eval()
        self.decoder.eval()
        switch_pca_mode(eval_mode=True)
        print(f"测试阶段，解码器PCA路径：{self.decoder.pca_save_path}")
        if ckpt_path:
            print(f"加载模型: {ckpt_path}")
        iterator = tqdm(val)
        with torch.no_grad():
            for step, (images, wms, mask) in enumerate(iterator):
                cover_images = images.to(cfg.device)
                mask = mask.to(cfg.device)
                watermark = wms.to(cfg.device)
                R, G, B = preprocess(cover_images)

                b_embedded, masks_scaled = self.encoder(B, cover_images, watermark)
                watermarked_images = torch.cat([R, G, b_embedded], dim=1)

                forward_b_embedded = b_embedded.clone().detach()
                forward_watermarked_images = watermarked_images.clone().detach()
                forward_cover_images = cover_images.clone().detach()
                forward_mask = mask.clone().detach()
                input = [forward_b_embedded, forward_watermarked_images, forward_cover_images, forward_mask]

                cover_images = cover_images.detach().cpu() + 1.0
                embedded_images = watermarked_images.detach().cpu() + 1.0
                metrics["val_psnr"].append(psnr(cover_images, embedded_images))
                metrics["val_ssim"].append(ssim(cover_images, embedded_images))

                val_jpeg_wm_all, regional_wms_jpeg, region_masks = validation_attack(input, b_embedded, jpeg, self.decoder, masks_scaled)
                ori_regional_wms = []
                for mask_tensor in region_masks:
                    ori_region_wm = watermark * mask_tensor
                    ori_regional_wms.append(ori_region_wm)
                ori_regional_wms = torch.stack(ori_regional_wms, dim=1)

                val_resize_wm_all, regional_wms_resize, _ = validation_attack(input, b_embedded, resize, self.decoder, masks_scaled)
                val_medianblur_wm_all, regional_wms_medianblur, _ = validation_attack(input, b_embedded, medianblur, self.decoder, masks_scaled)
                val_gaublur_wm_all, regional_wms_gaublur, _ = validation_attack(input, b_embedded, gau_blur, self.decoder, masks_scaled)
                val_gauNoise_wm_all, regional_wms_gauNoise, _ = validation_attack(input, b_embedded, gau_noise, self.decoder, masks_scaled)
                val_dropout_wm_all, regional_wms_dropout, _ = validation_attack(input, b_embedded, dropout_noise, self.decoder, masks_scaled)
                val_saltPepper_wm_all, regional_wms_saltPepper, _ = validation_attack(input, b_embedded, salt_pepper_noise, self.decoder, masks_scaled)
                val_identity_wm, regional_wms_identity, _ = validation_attack(input, b_embedded, identity, self.decoder, masks_scaled)
                val_stargan_wm, regional_wms_stargan, _ = validation_attack(input, b_embedded, stargan, self.decoder, masks_scaled, type="all")
                val_ganimation_wm, regional_wms_ganimation, _ = validation_attack(input, b_embedded, ganimation, self.decoder, masks_scaled, type="all")
                val_cscs_wm, regional_wms_cscs, _ = validation_attack(input, b_embedded, cscs, self.decoder, masks_scaled, type="all")
                val_hififace_wm, regional_wms_hififace, _ = validation_attack(input, b_embedded, hififace, self.decoder, masks_scaled, type="all")
                val_RAFSwap_wm, regional_wms_RAFSwap, _ = validation_attack(input, b_embedded, RAFSwap, self.decoder, masks_scaled, type="all")
                val_infoswap_wm, regional_wms_infoswap, _ = validation_attack(input, b_embedded, infoswap, self.decoder, masks_scaled, type="all")
                val_unifaceswap_wm, regional_wms_unifaceswap, _ = validation_attack(input, b_embedded, uniface_swap, self.decoder, masks_scaled, type="all")

                metrics["val_jpeg_er_all"].append(decoded_message_error_rate_batch(val_jpeg_wm_all, watermark).detach().cpu())
                metrics["val_resize_er_all"].append(decoded_message_error_rate_batch(val_resize_wm_all, watermark).detach().cpu())
                metrics["val_medianblur_er_all"].append(decoded_message_error_rate_batch(val_medianblur_wm_all, watermark).detach().cpu())
                metrics["val_gaublur_er_all"].append(decoded_message_error_rate_batch(val_gaublur_wm_all, watermark).detach().cpu())
                metrics["val_gauNoise_er_all"].append(decoded_message_error_rate_batch(val_gauNoise_wm_all, watermark).detach().cpu())
                metrics["val_dropout_er_all"].append(decoded_message_error_rate_batch(val_dropout_wm_all, watermark).detach().cpu())
                metrics["val_saltPepper_er_all"].append(decoded_message_error_rate_batch(val_saltPepper_wm_all, watermark).detach().cpu())
                metrics["val_identity_er_all"].append(decoded_message_error_rate_batch(val_identity_wm, watermark).detach().cpu())
                metrics["val_stargan_er_all"].append(decoded_message_error_rate_batch(val_stargan_wm, watermark).detach().cpu())
                metrics["val_ganimation_er_all"].append(decoded_message_error_rate_batch(val_ganimation_wm, watermark).detach().cpu())
                metrics["val_cscs_er_all"].append(decoded_message_error_rate_batch(val_cscs_wm, watermark).detach().cpu())
                metrics["val_hififace_er_all"].append(decoded_message_error_rate_batch(val_hififace_wm, watermark).detach().cpu())
                metrics["val_RAFSwap_er_all"].append(decoded_message_error_rate_batch(val_RAFSwap_wm, watermark).detach().cpu())
                metrics["val_infoswap_er_all"].append(decoded_message_error_rate_batch(val_infoswap_wm, watermark).detach().cpu())
                metrics["val_unifaceswap_er_all"].append(decoded_message_error_rate_batch(val_unifaceswap_wm, watermark).detach().cpu())

                metrics["val_jpeg_er_region"].append(decoded_message_error_rate_batch(ori_regional_wms, regional_wms_jpeg).detach().cpu())
                metrics["val_resize_er_region"].append(decoded_message_error_rate_batch(ori_regional_wms, regional_wms_resize).detach().cpu())
                metrics["val_medianblur_er_region"].append(decoded_message_error_rate_batch(ori_regional_wms, regional_wms_medianblur).detach().cpu())
                metrics["val_gaublur_er_region"].append(decoded_message_error_rate_batch(ori_regional_wms, regional_wms_gaublur).detach().cpu())
                metrics["val_gauNoise_er_region"].append(decoded_message_error_rate_batch(ori_regional_wms, regional_wms_gauNoise).detach().cpu())
                metrics["val_dropout_er_region"].append(decoded_message_error_rate_batch(ori_regional_wms, regional_wms_dropout).detach().cpu())
                metrics["val_saltPepper_er_region"].append(decoded_message_error_rate_batch(ori_regional_wms, regional_wms_saltPepper).detach().cpu())
                metrics["val_identity_er_region"].append(decoded_message_error_rate_batch(ori_regional_wms, regional_wms_identity).detach().cpu())
                metrics["val_stargan_er_region"].append(decoded_message_error_rate_batch(ori_regional_wms, regional_wms_stargan).detach().cpu())
                metrics["val_ganimation_er_region"].append(decoded_message_error_rate_batch(ori_regional_wms, regional_wms_ganimation).detach().cpu())
                metrics["val_cscs_er_region"].append(decoded_message_error_rate_batch(ori_regional_wms, regional_wms_cscs).detach().cpu())
                metrics["val_hififace_er_region"].append(decoded_message_error_rate_batch(ori_regional_wms, regional_wms_hififace).detach().cpu())
                metrics["val_RAFSwap_er_region"].append(decoded_message_error_rate_batch(ori_regional_wms, regional_wms_RAFSwap).detach().cpu())
                metrics["val_infoswap_er_region"].append(decoded_message_error_rate_batch(ori_regional_wms, regional_wms_infoswap).detach().cpu())
                metrics["val_unifaceswap_er_region"].append(decoded_message_error_rate_batch(ori_regional_wms, regional_wms_unifaceswap).detach().cpu())

                data_robust = [
                    ["Attack", "All(er)", "Region(er)"],
                    ["Jpeg", np.mean(metrics["val_jpeg_er_all"]), np.mean(metrics["val_jpeg_er_region"])],
                    ["Resize", np.mean(metrics["val_resize_er_all"]), np.mean(metrics["val_resize_er_region"])],
                    ["MedianBlur", np.mean(metrics["val_medianblur_er_all"]), np.mean(metrics["val_medianblur_er_region"])],
                    ["Gau_blur", np.mean(metrics["val_gaublur_er_all"]), np.mean(metrics["val_gaublur_er_region"])],
                    ["Gau_noise", np.mean(metrics["val_gauNoise_er_all"]), np.mean(metrics["val_gauNoise_er_region"])],
                    ["Dropout", np.mean(metrics["val_dropout_er_all"]), np.mean(metrics["val_dropout_er_region"])],
                    ["SaltPepper", np.mean(metrics["val_saltPepper_er_all"]), np.mean(metrics["val_saltPepper_er_region"])],
                    ["Identity", np.mean(metrics["val_identity_er_all"]), np.mean(metrics["val_identity_er_region"])],
                    ["StarGan", np.mean(metrics["val_stargan_er_all"]), np.mean(metrics["val_stargan_er_region"])],
                    ["Ganimation", np.mean(metrics["val_ganimation_er_all"]), np.mean(metrics["val_ganimation_er_region"])],
                    ["CSCS", np.mean(metrics["val_cscs_er_all"]), np.mean(metrics["val_cscs_er_region"])],
                    ["HifiFace", np.mean(metrics["val_hififace_er_all"]), np.mean(metrics["val_hififace_er_region"])],
                    ["RAFSwap", np.mean(metrics["val_RAFSwap_er_all"]), np.mean(metrics["val_RAFSwap_er_region"])],
                    ["InfoSwap", np.mean(metrics["val_infoswap_er_all"]), np.mean(metrics["val_infoswap_er_region"])],
                    ["UniFaceSwap", np.mean(metrics["val_unifaceswap_er_all"]), np.mean(metrics["val_unifaceswap_er_region"])],
                ]
                data_vis = [
                    ["PSNR", "SSIM"],
                    [np.mean(metrics["val_psnr"]), np.mean(metrics["val_ssim"])],
                ]

                table_str_robust = tabulate(data_robust, headers="firstrow", tablefmt="grid")
                print(table_str_robust)

                table_str_vis = tabulate(data_vis, headers="firstrow", tablefmt="grid")
                print(table_str_vis)

        table_str_robust = tabulate(data_robust, headers="firstrow", tablefmt="grid")
        print(table_str_robust)
        with open(os.path.join(log_dir, "metrics_table_er_average.json"), "at") as file0:
            print(table_str_robust, file=file0)

        table_str_vis = tabulate(data_vis, headers="firstrow", tablefmt="grid")
        print(table_str_vis)
        with open(os.path.join(log_dir, "metrics_table_visual_average.json"), "at") as file1:
            print(table_str_vis, file=file1)

        metrics_avg = {
            k: round(np.mean(v), 7) if len(v) > 0 else "NaN"
            for k, v in metrics.items()
        }
        history.append(metrics_avg)
        pd.DataFrame(history).to_csv(os.path.join(log_dir, "metrics.tsv"), index=False, sep="\t")
        with open(os.path.join(log_dir, "metrics.json"), "at") as out:
            out.write(json.dumps(metrics_avg, indent=2, default=lambda o: str(o)))

        return history


if __name__ == "__main__":
    seed_torch(42)
    model = IWNet()
    checkpoint = torch.load(
        "/root/autodl-tmp/code_bk/exp_segfacemark/2026.07.09-20.58.52/model_state_5.pth",
        map_location="cuda:0",
    )
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"], strict=True)
    else:
        model.load_state_dict(checkpoint, strict=True)

    model.fit()
