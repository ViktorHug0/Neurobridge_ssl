"""SDXL + IP-Adapter reconstructor: predicted CLIP ViT-H/14 image embedding -> image.

Two paths:
  - text2image: embed -> image (default; the ENIGMA-style minimal path).
  - image2image: a low-level init image (decoded from an EEG->VAE-latent decoder) is refined by
    SDXL under the IP-Adapter semantic embed -> injects coarse layout/colour (Phase 1b).

Feed the *raw-scale* (norm ~22) ViT-H/14 embedding the IP-Adapter expects. One ~10GB GPU shared
with training runs, so: no CLIP image-encoder (we pass precomputed embeds) + VAE tiling + CPU offload.

ponytail: SDXL-Turbo only (4 steps, no CFG). Full SDXL + real CFG was tried and regresses
(PixCorr .133 vs .149, CLIP .702 vs .743) -- CFG amplifies the noisy EEG embedding.
"""
import numpy as np
import torch
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image, AutoencoderKL
from PIL import Image


class SDXLReconstructor:
    def __init__(self, cache_dir=None, device="cuda"):
        self.dtype = torch.float16
        self.device = torch.device(device)
        self.num_inference_steps = 4

        pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo", variant="fp16", torch_dtype=self.dtype, cache_dir=cache_dir
        )
        pipe.load_ip_adapter(
            "h94/IP-Adapter", subfolder="sdxl_models",
            weight_name="ip-adapter_sdxl_vit-h.safetensors",
            image_encoder_folder=None, cache_dir=cache_dir,
        )
        pipe.set_ip_adapter_scale(1.0)
        pipe.enable_vae_tiling()
        pipe.enable_model_cpu_offload()  # fits the shared 10GB card
        self.pipe = pipe
        self.img2img = AutoPipelineForImage2Image.from_pipe(pipe)  # shares weights + IP-Adapter
        self.vae_scale = pipe.vae.config.scaling_factor
        # dedicated fp32 VAE for the low-level latent decode (sdxl-vae NaNs in fp16)
        self.vae32 = AutoencoderKL.from_pretrained(
            "stabilityai/sdxl-vae", torch_dtype=torch.float32, cache_dir=cache_dir).to(self.device).eval()

    def _embeds(self, c_i):
        return [torch.as_tensor(c_i, dtype=self.dtype).reshape(1, 1, -1)]

    @torch.no_grad()
    def decode_latent(self, latent):
        """VAE-decode a (4,64,64) *scaled* latent -> PIL init image."""
        z = torch.as_tensor(np.asarray(latent), dtype=torch.float32, device=self.device).reshape(1, 4, 64, 64)
        img = self.vae32.decode(z / self.vae_scale).sample[0]
        arr = ((img / 2 + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        return Image.fromarray(arr)

    @torch.no_grad()
    def reconstruct(self, c_i, init_image=None, strength=0.65):
        """c_i: (1024,) raw-scale ViT-H embedding. init_image=None -> text2img; else img2img."""
        if init_image is None:
            return self.pipe(
                prompt="", ip_adapter_image_embeds=self._embeds(c_i),
                num_inference_steps=self.num_inference_steps, guidance_scale=0.0,
            ).images[0]
        steps = max(self.num_inference_steps, int(np.ceil(2 / strength)))  # keep >=2 denoise steps
        return self.img2img(
            prompt="", image=init_image, strength=strength,
            ip_adapter_image_embeds=self._embeds(c_i),
            num_inference_steps=steps, guidance_scale=0.0,
        ).images[0]


if __name__ == "__main__":
    import sys
    cache = sys.argv[1] if len(sys.argv) > 1 else None
    r = SDXLReconstructor(cache_dir=cache)
    v = torch.nn.functional.normalize(torch.randn(1024), dim=0) * 22.0
    img = r.reconstruct(v)
    assert img.size[0] > 0, "empty image"
    img.save("recon_smoketest.png")
    print("OK reconstruct ->", img.size)
