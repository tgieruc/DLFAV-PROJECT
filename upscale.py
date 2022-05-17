import sys,os
sys.path.insert(0, os.path.join(os.path.abspath(''), "Real-ESRGAN"))

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

class Upscaler(object):
    def __init__(self, enhance_face):
        # model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        netscale = 4
        model_path = os.path.join('Real-ESRGAN/experiments/pretrained_models/realesr-animevideov3.pth')
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