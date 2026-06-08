import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import torch
import torch.nn as nn
from network import noise
# 256 带；128 不带
# from segfacemark.encoder import Encoder
# from segfacemark.decoder import Decoder
from encoder_256 import Encoder
from decoder_256 import Decoder
from tabulate import tabulate
from utils.Quality import psnr, ssim
from random import randint
from torch import optim
from datetime import datetime
from utils.DataLoad_highpass import *
from utils.torch_utils import decoded_message_error_rate_batch
from utils.image_wm_dataset import get_loader
import json
from tqdm import tqdm
import warnings
from advanced_gnn import GNNModel, build_graph
import random
import numpy as np
from segfaceAttention import SegFaceWatermarkAdapter
warnings.filterwarnings("ignore")
from config import training_config as cfg

history = []

face_segmenter = SegFaceWatermarkAdapter(cfg.device)

train_loader = get_loader(
    cfg.train_img_dir, cfg.train_wm_dir,
    cfg.image_size, cfg.batch_size, shuffle=False
)

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

def lr_decay(lr, epoch, opt):
    if epoch == 3:
        for param_group in opt.param_groups:
            param_group["lr"] = 5e-5
    elif epoch == 5:
        for param_group in opt.param_groups:
            param_group["lr"] = 1e-5 
    elif epoch == 7:
        for param_group in opt.param_groups:
            param_group["lr"] = 1e-6

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
            log_dir = f'exp_segfacemark/{(datetime.now().strftime("%Y.%m.%d-%H.%M.%S"))}'
        os.makedirs(log_dir, exist_ok=True)

        train = train_loader
        val = val_loader

        optimizer_encoder = optim.Adam(self.encoder.parameters(), lr=lr, weight_decay=0.00001)
        optimizer_decoder_t = optim.Adam(self.decoder_t.parameters(), lr=lr, weight_decay=0.00001)

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
        stargan = noise.stargan_noise
        ganimation = noise.ganimation_noise
        uniface_swap = noise.unifaceswap_noise
        fsrt = noise.fsrt_noise
        cscs = noise.cscs_noise
        hififace = noise.hififace_noise
        infoswap = noise.infoswap_noise
        stylemask = noise.stylemask_noise
        RAFSwap = noise.RAFSwap_noise

        def add_noise(input, u_embedded, type):
            if type == "all":
                choice = randint(0, 14)
            if choice == 0:
                return u_embedded + identity(input)
            if choice == 1:
                return u_embedded + jpeg(input)
            if choice == 2:
                return u_embedded + resize(input)
            if choice == 3:
                return u_embedded + medianblur(input)
            if choice == 4:
                return u_embedded + gau_noise(input)
            if choice == 5:
                return u_embedded + gau_blur(input)
            if choice == 6:
                return u_embedded + dropout_noise(input)
            if choice == 7:
                return u_embedded + salt_pepper_noise(input)
            if choice == 8:
                return u_embedded + stargan(input, type)
            if choice == 9:
                return u_embedded + ganimation(input, type)
            if choice == 10:
                return u_embedded +  uniface_swap(input, type)
            if choice == 11:
                return u_embedded + fsrt(input, type)
            if choice == 10:
                return u_embedded + cscs(input, type)
            if choice == 11:
                return u_embedded + hififace(input, type)
            if choice == 12:
                return u_embedded + RAFSwap(input, type)
            if choice == 13:
                return u_embedded + infoswap(input, type)
            if choice == 14:
                return u_embedded + stylemask(input, type)

        def decode(u_embedded, decoder, masks_scaled):
            return decoder(u_embedded, masks_scaled)

        def validation_attack(input, u_embedded, noise, decoder, masks_scaled, type="default"):
            noised = noise(input, type)
            wm, regional_wms, region_masks  = decode(u_embedded + noised, decoder, masks_scaled)
            return wm, regional_wms, region_masks
        
        def switch_pca_mode(eval_mode):
            if eval_mode:
                self.decoder_t.pca_save_path = self.EVAL_PCA_PATH
            else:
                self.decoder_t.pca_save_path = self.TRAIN_PCA_PATH
            self.decoder_t.load_pca_model()

        for epoch in range(1, epochs + 1):
            metrics = {
                "train_loss": [],
                "train_vis": [],
                "train_er": [],
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
                "val_RAFSwap_er_all": [],
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
                "val_RAFSwap_er_region": [],
                "val_stylemask_er_region": [],
                "val_infoswap_er_region": [],
            }

            self.encoder.train()
            self.decoder_t.train()
            switch_pca_mode(eval_mode=False)
            print(f"Epoch {epoch} - 训练阶段，解码器PCA路径：{self.decoder_t.pca_save_path}")
            iterator = tqdm(train)
            cur_lr = 0.0

            for step, (cover_images, wms, mask) in enumerate(iterator):
                cover_images = cover_images.to(cfg.device)
                mask = mask.to(cfg.device)
                watermark = wms.to(cfg.device)

                R, G, B = preprocess(cover_images)

                for param_group in optimizer_encoder.param_groups:
                    cur_lr = param_group["lr"]

                lr_decay(cur_lr, epoch, optimizer_encoder)
                lr_decay(cur_lr, epoch, optimizer_decoder_t)

                # return embedded, masks_scaled
                b_embedded, masks_scaled = self.encoder(B, cover_images, watermark)

                b_graphs = build_graph(B)
                b_embeded_graphs = build_graph(b_embedded)
                b_features = self.gnn(b_graphs) 
                b_embeded_features = self.gnn(b_embeded_graphs)

                watermarked_images = torch.cat([R, G, b_embedded], dim=1)

                forward_b_embedded = b_embedded.clone().detach()
                forward_watermarked_images = watermarked_images.clone().detach()
                forward_cover_images = cover_images.clone().detach()
                forward_mask = mask.clone().detach()
                input = [forward_b_embedded, forward_watermarked_images, forward_cover_images, forward_mask]
                b_embedded_attack_type_all = add_noise(input, b_embedded, type="all")

                # return message, regional_wms, region_masks_list
                extract_wm_all, regional_wms, region_masks  = decode(b_embedded_attack_type_all, self.decoder_t, masks_scaled)
                ori_regional_wms = []
                for mask_tensor in region_masks:
                    ori_region_wm = watermark * mask_tensor
                    ori_regional_wms.append(ori_region_wm)
                ori_regional_wms = torch.stack(ori_regional_wms, dim=1)

                mse = nn.MSELoss().to(cfg.device)

                regional_losses = []
                for r in range(regional_wms.shape[1]):
                    decoded_region_wm = regional_wms[:, r, :]
                    original_region_wm = ori_regional_wms[:, r, :]
                    regional_loss = mse(decoded_region_wm, original_region_wm)
                    regional_losses.append(regional_loss)
                avg_regional_loss = torch.mean(torch.stack(regional_losses))

                loss_gnn = 0
                for carrier_feature, watermarked_feature in zip(b_features, b_embeded_features):
                    loss_gnn += mse(carrier_feature, watermarked_feature)
                loss_gnn /= batch_size
                loss_encoder = mse(B, b_embedded)
                loss_noise_all = mse(extract_wm_all, watermark)
                loss_total = loss_encoder * cfg.encoder_w + loss_noise_all * cfg.all_w + loss_gnn * cfg.gnn

                optimizer_encoder.zero_grad()
                optimizer_decoder_t.zero_grad()
                loss_total.backward()
                optimizer_encoder.step()
                optimizer_decoder_t.step()

                metrics["train_loss"].append(loss_total.item())
                metrics["train_vis"].append(loss_encoder.item())
                metrics["train_er"].append(decoded_message_error_rate_batch(extract_wm_all, watermark).detach().cpu())

                iterator.set_description(
                    "Epoch %s | Loss %.6f | Vis %.6f |Er %.6f" % (
                        epoch,
                        np.mean(metrics["train_loss"]),
                        np.mean(metrics["train_vis"]),
                        np.mean(metrics["train_er"]),
                    )
                )

            self.encoder.eval()
            self.decoder_t.eval()

            switch_pca_mode(eval_mode=True)
            print(f"Epoch {epoch} - 测试阶段，解码器PCA路径：{self.decoder_t.pca_save_path}")

            iterator = tqdm(val)
            with torch.no_grad():
                for step, (images, wms, mask) in enumerate(iterator):
                    cover_images = images.to(cfg.device)
                    mask = mask.to(cfg.device)
                    watermark = wms.to(cfg.device)
                    R, G, B = preprocess(cover_images)

                    b_embedded, masks_scaled  = self.encoder(B, cover_images, watermark)

                    watermarked_images = torch.cat([R, G, b_embedded], dim=1)

                    forward_b_embedded = b_embedded.clone().detach()
                    forward_watermarked_images = watermarked_images.clone().detach()
                    forward_cover_images = cover_images.clone().detach()
                    forward_mask = mask.clone().detach()
                    input = [forward_b_embedded, forward_watermarked_images, forward_cover_images, forward_mask]

                    cover_images = cover_images.detach().cpu()
                    embedded_images = watermarked_images.clamp(-1, 1).detach().cpu()
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
                    val_stargan_wm_all, regional_wms_stargan, _ = validation_attack(input, b_embedded, stargan, self.decoder_t, masks_scaled, type="all")
                    val_ganimation_wm_all, regional_wms_ganimation, _ = validation_attack(input, b_embedded, ganimation, self.decoder_t, masks_scaled, type="all")
                    val_unifaceswap_wm_all, regional_wms_unifaceswap, _ = validation_attack(input, b_embedded, uniface_swap, self.decoder_t, masks_scaled, type="all")
                    val_fsrt_wm_all, regional_wms_fsrt, _ = validation_attack(input, b_embedded, fsrt, self.decoder_t, masks_scaled, type="all")
                    val_cscs_wm_all, regional_wms_cscs, _ = validation_attack(input, b_embedded, cscs, self.decoder_t, masks_scaled, type="all")
                    val_hififace_wm_all, regional_wms_hififace, _ = validation_attack(input, b_embedded, hififace, self.decoder_t, masks_scaled, type="all")
                    val_RAFSwap_wm_all, regional_wms_RAFSwap, _ = validation_attack(input, b_embedded, RAFSwap, self.decoder_t, masks_scaled, type="all")
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
                    metrics["val_RAFSwap_er_all"].append(decoded_message_error_rate_batch(val_RAFSwap_wm_all, watermark).detach().cpu())
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
                    metrics["val_RAFSwap_er_region"].append(decoded_message_error_rate_batch(ori_regional_wms, regional_wms_RAFSwap).detach().cpu())
                    metrics["val_stylemask_er_region"].append(decoded_message_error_rate_batch(ori_regional_wms, regional_wms_stylemask).detach().cpu())
                    metrics["val_infoswap_er_region"].append(decoded_message_error_rate_batch(ori_regional_wms, regional_wms_infoswap).detach().cpu())

                    print(f"val-epoch-{epoch}: \n")
                    data_vis = [
                        ["PSNR", "SSIM"],
                        [np.mean(metrics["val_psnr"]), np.mean(metrics["val_ssim"])],
                    ]
                    data_err = [
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
                        ["RAFSwap", np.mean(metrics["val_RAFSwap_er_all"]), np.mean(metrics["val_RAFSwap_er_region"])],
                        ["stylemask", np.mean(metrics["val_stylemask_er_all"]), np.mean(metrics["val_stylemask_er_region"])],
                        ["infoswap", np.mean(metrics["val_infoswap_er_all"]), np.mean(metrics["val_infoswap_er_region"])],
                    ]
                    table_str = tabulate(data_vis, headers="firstrow", tablefmt="grid")
                    print(table_str)
                    with open(os.path.join(log_dir, "metrics_table_visual.json"), "at") as file0: 
                        print(table_str, file=file0)

                    table_str2 = tabulate(data_err, headers="firstrow", tablefmt="grid")
                    print(table_str2)
                    with open(os.path.join(log_dir, "metrics_table_err.json"), "at") as file1:
                        print(table_str2, file=file1)
            
            metrics = {
                k: round(np.mean(v), 7) if len(v) > 0 else "NaN"
                for k, v in metrics.items()
            }
            metrics["epoch"] = epoch
            metrics["LR"] = cur_lr
            history.append(metrics)
            pd.DataFrame(history).to_csv(os.path.join(log_dir, "metrics.tsv"), index=False, sep="\t")
            with open(os.path.join(log_dir, "metrics.json"), "at") as out:
                out.write(json.dumps(metrics, indent=2, default=lambda o: str(o)))
            # torch.save(self, os.path.join(log_dir, f"model_{epoch}.pth"))
            # torch.save(self.state_dict(), os.path.join(log_dir, f"model_state_{epoch}.pth"))
            save_path = os.path.join(log_dir, f"model_state_{epoch}.pth")
            checkpoint = {
                "epoch": epoch,
                "model_state": self.state_dict(),
                "optimizer_state": self.optimizer.state_dict() if hasattr(self, 'optimizer') else None,
                "scheduler_state": self.scheduler.state_dict() if hasattr(self, 'scheduler') else None
            }
            torch.save(checkpoint, save_path)

        return history


if __name__ == "__main__":
    import traceback
    try:
        seed_torch(42)
        model = IWNet()
        # model.load_state_dict(
        #     torch.load(
        #         "/root/autodl-tmp/code_bk/exp_segfacemark/2026.06.06-22.01.24/model_state_2.pth",
        #         map_location="cuda:0",
        #     ),
        #     strict=False
        # )

        # checkpoint = torch.load("/root/autodl-tmp/code_bk/exp_segfacemark/2026.06.06-23.54.25/model_state_4.pth")
        # if "model_state" in checkpoint:
        #     model.load_state_dict(checkpoint["model_state"], strict=True)
        # else:
        #     model.load_state_dict(checkpoint, strict=True)
        model.fit()
    except Exception as e:
        print("程序执行出错：", e)
        traceback.print_exc(file=sys.stdout)
