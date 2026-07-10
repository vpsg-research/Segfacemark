from .common_noise import *
from .stargan.main import StarGAN
from .ganimation.main import GANimation
from .cscs.test import CSCS
from .hififace.test import HifiFaceNoise
from .RAFSwap.test import RAFSwapNoise
from .infoswap.test import InfoSwapNoise
from .uniface.swap  import UniFaceSwap
from .fsrt.new_test import Fsrt
# from .stylemask.test import StyleMaskModel

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
        gap = noised_image.clamp(-1, 1) - input[0]
        return gap
    return df_noise



stargan_noise = df_closure(StarGAN())
ganimation_noise = df_closure(GANimation())
cscs_noise = df_closure(CSCS())
hififace_noise = df_closure(HifiFaceNoise())
RAFSwap_noise = df_closure(RAFSwapNoise())
infoswap_noise = df_closure(InfoSwapNoise())
unifaceswap_noise = df_closure(UniFaceSwap())
fsrt_noise = df_closure(Fsrt())
# stylemask_noise = df_closure(StyleMaskModel())

# unifacereenact_noise = df_closure(UniFaceReenactment())
# e4s_noise = df_closure(E4SNoiseLayer())
# simswap_noise = df_closure(SimSwap())
# diffSwap_noise = df_closure(DiffSwapNoiseLayer())

