
from .common_noise import *
from utils.torch_utils import normalize, denormalize, yuv_denormalize, yuv_normalize, yuv2rgb,rgb2yuv
from .stargan.main import StarGAN
from .ganimation.main import GANimation
from .uniface.swap  import UniFaceSwap
from .fsrt.new_test import Fsrt
from .cscs.test import CSCS
from .hififace.test import HifiFaceNoise
from .stylemask.test import StyleMaskModel
from .infoswap.test import InfoSwapNoise
from .RAFSwap.test import RAFSwapNoise
# from matplotlib.mlab import detrend_none
# from .uniface.reenact  import UniFaceReenactment
# from .simswap.test_one_image import SimSwap
# from .e4s.test import E4SNoiseLayer
# from .DiffSwap.diffswap_noise import DiffSwapNoiseLayer

# input = [forward_b_embedded, forward_watermarked_images, forward_cover_images, forward_mask]


def df_closure(df):
    def df_noise(input, type):
        df_input = [input[1], input[3]]
        noised_image = df(df_input)[:, [2], :, :]
        gap = noised_image - input[0]
        return gap if type == "deepfake" else noised_image
    return df_noise


stargan_noise = df_closure(StarGAN())
ganimation_noise = df_closure(GANimation())
unifaceswap_noise = df_closure(UniFaceSwap())
fsrt_noise = df_closure(Fsrt())
cscs_noise = df_closure(CSCS())
hififace_noise = df_closure(HifiFaceNoise())
stylemask_noise = df_closure(StyleMaskModel())
infoswap_noise = df_closure(InfoSwapNoise())
RAFSwap_noise = df_closure(RAFSwapNoise())
