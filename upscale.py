import sys,os

import torch.cuda

sys.path.insert(0, os.path.join(os.path.abspath(''), "Real-ESRGAN"))

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from basicsr.archs.rrdbnet_arch import RRDBNet

class Upscaler(object):
    def __init__(self, model_choice, enhance_face=False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_choice = model_choice
        if model_choice == "realesrgan_s":
            model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
            model_path = os.path.join('Real-ESRGAN/experiments/pretrained_models/realesr-animevideov3.pth')
        elif model_choice == "realesrgan_4xplus":
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            model_path = os.path.join('Real-ESRGAN/experiments/pretrained_models/RealESRGAN_x4plus.pth')


        netscale = 4
        self.upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            model=model,
            )
        self.enhance_face = enhance_face
        if enhance_face:
            from gfpgan import GFPGANer
            self.face_enhancer = GFPGANer(
                model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
                upscale=netscale,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=self.upsampler)

    def enhance(self, img):
        if self.enhance_face:
            _,_, output = self.face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
        else:
            output, _ = self.upsampler.enhance(img)

        return output