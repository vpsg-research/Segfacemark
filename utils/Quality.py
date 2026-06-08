import torch
import pytorch_ssim

def psnr(images1, images2):
    batch_size = images1.size(0)
    total_psnr = torch.tensor(0.0, device=images1.device)
    for i in range(batch_size):
        mse = torch.mean((images1[i] - images2[i]) ** 2)
        total_psnr += 20 * torch.log10(2 / torch.sqrt(mse + 1e-8))  # 避免除0
    return total_psnr / batch_size


def ssim(images1, images2):
    batch_size = images1.size(0)
    total_ssim = torch.tensor(0.0, device=images1.device)
    for i in range(batch_size):
        total_ssim += pytorch_ssim.ssim(images1[i:i+1], images2[i:i+1])
    return total_ssim / batch_size
