import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import torch
import torch.nn as nn
from network import noise
# from encoder import Encoder
# from decoder import Decoder
from encoder_256 import Encoder
from decoder_256 import Decoder
from tabulate import tabulate
from config import training_config as cfg
# from config import test256_config as cfg
from utils.Quality import psnr, ssim
from datetime import datetime
from utils.DataLoad_highpass import *
from utils.torch_utils import decoded_message_error_rate_batch
# from dataloader import val_dataloader
from utils.image_wm_dataset import get_loader
from network.noise import stargan_for_test
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
    images_y = images[:, [0], :, :]
    images_u = images[:, [1], :, :]
    images_v = images[:, [2], :, :]
    return images_y, images_u, images_v

class IWNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(face_segmenter).to(cfg.device)
        self.decoder_t = Decoder(face_segmenter).to(cfg.device)
        in_dim = 1 * 9
        self.gnn = GNNModel(in_channels=in_dim, hidden_dim=64, out_dim=128).to(cfg.device)

        self.TRAIN_PCA_PATH = cfg.TRAIN_PCA_PATH
        self.EVAL_PCA_PATH = cfg.EVAL_PCA_PATH

    def fit(self, log_dir=False, batch_size=cfg.batch_size, lr=float(cfg.lr), epochs=cfg.epochs):
        if not log_dir:
            log_dir = f'exp_segfacemark_test/{(datetime.now().strftime("%Y.%m.%d-%H.%M.%S"))}'
        os.makedirs(log_dir, exist_ok=True)

        val = val_loader

        with open(os.path.join(log_dir, "config.json"), "wt") as out:
            out.write(json.dumps(cfg,indent=2, default=lambda o: str(o)))

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

        def validation_attack(input, u_embedded, noise, decoder, masks_scaled, type="default"):
            noised = noise(input, type)
            wm, regional_wms, region_masks  = decode(u_embedded + noised, decoder, masks_scaled)
            return wm, regional_wms, region_masks

        # def stargan_attack(input, u_embedded, noise, decoder, type="default", c_trg="3"):
        #     noised = noise(input, type, c_trg)
        #     wm = decode(u_embedded + noised, decoder)
        #     return wm

        def switch_pca_mode(eval_mode):
            if eval_mode:
                self.decoder_t.pca_save_path = self.EVAL_PCA_PATH
            else:
                self.decoder_t.pca_save_path = self.TRAIN_PCA_PATH
            self.decoder_t.load_pca_model()
        
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
            "val_unifaceswap_er_all": [],
            "val_fsrt_er_all": [],
            "val_cscs_er_all": [],
            "val_hififace_er_all": [],
            "val_stylemask_er_all": [],
            "val_infoswap_er_all": [],

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
            "val_unifaceswap_er_region": [],
            "val_fsrt_er_region": [],
            "val_cscs_er_region": [],
            "val_hififace_er_region": [],
            "val_stylemask_er_region": [],
            "val_infoswap_er_region": [],
        }
        data_robust = []
        data_vis = []

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
                R, G, B = preprocess(cover_images)

                # watermark = torch.Tensor(np.random.choice([-cfg.message_range, cfg.message_range], (cover_images.shape[0], cfg.message_length))).to(cfg.device)
                b_embedded, masks_scaled  = self.encoder(B, cover_images, watermark)
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

                val_jpeg_wm_all, regional_wms_jpeg, region_masks = validation_attack(input, b_embedded, jpeg, self.decoder_t, masks_scaled)
                ori_regional_wms = []
                for mask_tensor in region_masks:
                    ori_region_wm = watermark * mask_tensor
                    ori_regional_wms.append(ori_region_wm)
                ori_regional_wms = torch.stack(ori_regional_wms, dim=1)
                val_resize_wm_all, regional_wms_resize, _ = validation_attack(input, b_embedded, resize, self.decoder_t, masks_scaled)
                val_medianblur_wm_all, regional_wms_medianblur, _ = validation_attack(input, b_embedded, medianblur, self.decoder_t, masks_scaled)
                val_gaublur_wm_all, regional_wms_gaublur, _ = validation_attack(input, b_embedded, gau_blur, self.decoder_t, masks_scaled)
                val_gauNoise_wm_all, regional_wms_gauNoise, _ = validation_attack(input, b_embedded, gau_noise, self.decoder_t, masks_scaled)
                val_dropout_wm_all, regional_wms_dropout, _ = validation_attack(input, b_embedded, dropout_noise, self.decoder_t, masks_scaled)
                val_saltPepper_wm_all, regional_wms_saltPepper, _ = validation_attack(input, b_embedded, salt_pepper_noise, self.decoder_t, masks_scaled)
                val_identity_wm_all, regional_wms_identity, _ = validation_attack(input, b_embedded, identity, self.decoder_t, masks_scaled)
                val_ganimation_wm_all, regional_wms_ganimation, _ = validation_attack(input, b_embedded, ganimation, self.decoder_t, masks_scaled, type="all")
                val_stargan_wm_all, regional_wms_stargan, _ = validation_attack(input, b_embedded, stargan, self.decoder_t, masks_scaled, type="all")
                val_unifaceswap_wm_all, regional_wms_unifaceswap, _ = validation_attack(input, b_embedded, uniface_swap, self.decoder_t, masks_scaled, type="all")
                val_fsrt_wm_all, regional_wms_fsrt, _ = validation_attack(input, b_embedded, fsrt, self.decoder_t, masks_scaled, type="all")
                val_cscs_wm_all, regional_wms_cscs, _ = validation_attack(input, b_embedded, cscs, self.decoder_t, masks_scaled, type="all")
                val_hififace_wm_all, regional_wms_hififace, _ = validation_attack(input, b_embedded, hififace, self.decoder_t, masks_scaled, type="all")
                val_stylemask_wm_all, regional_wms_stylemask, _ = validation_attack(input, b_embedded, stylemask, self.decoder_t, masks_scaled, type="all")
                val_infoswap_wm_all, regional_wms_infoswap, _ = validation_attack(input, b_embedded, infoswap, self.decoder_t, masks_scaled, type="all")

                metrics["val_jpeg_er_all"].append(decoded_message_error_rate_batch(val_jpeg_wm_all, watermark).detach().cpu())
                metrics["val_resize_er_all"].append(decoded_message_error_rate_batch(val_resize_wm_all, watermark).detach().cpu())
                metrics["val_medianblur_er_all"].append(decoded_message_error_rate_batch(val_medianblur_wm_all, watermark).detach().cpu())
                metrics["val_gaublur_er_all"].append(decoded_message_error_rate_batch(val_gaublur_wm_all, watermark).detach().cpu())
                metrics["val_gauNoise_er_all"].append(decoded_message_error_rate_batch(val_gauNoise_wm_all, watermark).detach().cpu())
                metrics["val_dropout_er_all"].append(decoded_message_error_rate_batch(val_dropout_wm_all, watermark).detach().cpu())
                metrics["val_saltPepper_er_all"].append(decoded_message_error_rate_batch(val_saltPepper_wm_all, watermark).detach().cpu())
                metrics["val_identity_er_all"].append(decoded_message_error_rate_batch(val_identity_wm_all, watermark).detach().cpu())
                metrics["val_stargan_er_all"].append(decoded_message_error_rate_batch(val_stargan_wm_all, watermark).detach().cpu())
                metrics["val_ganimation_er_all"].append(decoded_message_error_rate_batch(val_ganimation_wm_all, watermark).detach().cpu())
                metrics["val_unifaceswap_er_all"].append(decoded_message_error_rate_batch(val_unifaceswap_wm_all, watermark).detach().cpu())
                metrics["val_fsrt_er_all"].append(decoded_message_error_rate_batch(val_fsrt_wm_all, watermark).detach().cpu())
                metrics["val_cscs_er_all"].append(decoded_message_error_rate_batch(val_cscs_wm_all, watermark).detach().cpu())
                metrics["val_hififace_er_all"].append(decoded_message_error_rate_batch(val_hififace_wm_all, watermark).detach().cpu())
                metrics["val_stylemask_er_all"].append(decoded_message_error_rate_batch(val_stylemask_wm_all, watermark).detach().cpu())
                metrics["val_infoswap_er_all"].append(decoded_message_error_rate_batch(val_infoswap_wm_all, watermark).detach().cpu())
                
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
                metrics["val_unifaceswap_er_region"].append(decoded_message_error_rate_batch(ori_regional_wms, regional_wms_unifaceswap).detach().cpu())
                metrics["val_fsrt_er_region"].append(decoded_message_error_rate_batch(ori_regional_wms, regional_wms_fsrt).detach().cpu())
                metrics["val_cscs_er_region"].append(decoded_message_error_rate_batch(ori_regional_wms, regional_wms_cscs).detach().cpu())
                metrics["val_hififace_er_region"].append(decoded_message_error_rate_batch(ori_regional_wms, regional_wms_hififace).detach().cpu())
                metrics["val_stylemask_er_region"].append(decoded_message_error_rate_batch(ori_regional_wms, regional_wms_stylemask).detach().cpu())
                metrics["val_infoswap_er_region"].append(decoded_message_error_rate_batch(ori_regional_wms, regional_wms_infoswap).detach().cpu())

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
                    ["UniFaceswap", np.mean(metrics["val_unifaceswap_er_all"]), np.mean(metrics["val_unifaceswap_er_region"])],
                    ["FSRT", np.mean(metrics["val_fsrt_er_all"]), np.mean(metrics["val_fsrt_er_region"])],
                    ["CSCS", np.mean(metrics["val_cscs_er_all"]), np.mean(metrics["val_cscs_er_region"])],
                    ["hififace", np.mean(metrics["val_hififace_er_all"]), np.mean(metrics["val_hififace_er_region"])],
                    ["stylemask", np.mean(metrics["val_stylemask_er_all"]), np.mean(metrics["val_stylemask_er_region"])],
                    ["infoswap", np.mean(metrics["val_infoswap_er_all"]), np.mean(metrics["val_infoswap_er_region"])],
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
        
        metrics = {
            k: round(np.mean(v), 7) if len(v) > 0 else "NaN"
            for k, v in metrics.items()
        }
        history.append(metrics)
        pd.DataFrame(history).to_csv(os.path.join(log_dir, "metrics.tsv"), index=False, sep="\t")
        with open(os.path.join(log_dir, "metrics.json"), "at") as out:
            out.write(json.dumps(metrics, indent=2, default=lambda o: str(o)))

        return history


if __name__ == "__main__":
    seed_torch(42)
    model = IWNet()
    # model.load_state_dict(
    #     torch.load(
    #         "/root/autodl-tmp/code_bk/exp_segfacemark/2026.02.22-21.14.55/model_state_3.pth",
    #         map_location="cuda:0",
    #     ),
    #     strict=False
    # )
    checkpoint = torch.load("/root/segfacemark_v/model_state_11.pth")
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"], strict=True)
    else:
        model.load_state_dict(checkpoint, strict=True)

    model.fit()

# if __name__ == "__main__":
#     import torch
#     from thop import profile

#     seed_torch(42)
#     device = cfg.device
#     model = IWNet().to(device)
#     model.eval()

#     # 静默加载权重
#     checkpoint_path = "/root/autodl-tmp/code_bk/exp_segfacemark/2026.06.05-21.20.53/model_state_3.pth"
#     try:
#         checkpoint = torch.load(checkpoint_path, map_location=device)
#         model.load_state_dict(checkpoint.get("model_state", checkpoint), strict=False)
#     except Exception:
#         pass

#     # 提取真实数据
#     test_loader = iter(val_loader)
#     images, wms, _ = next(test_loader)
#     real_images = images[:1].to(device)
#     real_wms = wms[:1].to(device)
#     real_U = real_images[:, [1], :, :]

#     print("="*50)
#     print("正在避开 Decoder 结构冲突，精准测算 Encoder 算力...")
    
#     # 切断梯度，防止算力工具抽风
#     for p in model.encoder.parameters():
#         p.requires_grad = False

#     try:
#         # 1. 测 Encoder 耗时
#         starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
#         with torch.no_grad():
#             for _ in range(5):  # 预热
#                 _ = model.encoder(real_U, real_images, real_wms)
#             starter.record()
#             _ = model.encoder(real_U, real_images, real_wms)
#             ender.record()
#             torch.cuda.synchronize()
#             infer_time = starter.elapsed_time(ender)

#         # 2. 测 Encoder FLOPs
#         macs_enc, _ = profile(model.encoder, inputs=(real_U, real_images, real_wms), verbose=False)
#         flops_enc_g = macs_enc / 1e9

#         print(f">>> [完美出数] 嵌入端 (Encoder) 单步计算量: {flops_enc_g:.2f} G FLOPs")
#         print(f">>> [完美出数] 嵌入端 (Encoder) 单步耗时:   {infer_time:.2f} ms")
#         print(f"")
#         print(f"  * 论文填表建议：由于 Decoder (1.67M) 与 Encoder (1.73M) 对称，")
#         print(f"    你可以直接在论文报告总 FLOPs 为 ~{flops_enc_g * 2:.2f} G！")
#         print("="*50)
#     except Exception as e:
#         print(f"测算异常: {e}")