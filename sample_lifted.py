import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import argparse
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from omegaconf import OmegaConf
from skimage.metrics import peak_signal_noise_ratio as psnr

from data.dataloader import get_dataloader, get_dataset
from ldm.models.diffusion.ddim import DDIMSampler
from ldm_inverse.condition_methods import get_conditioning_method
from ldm_inverse.measurements import get_noise, InpaintingOperator
from model_loader import load_model_from_config, load_yaml
from scripts.utils import clear_color, mask_generator
from rfstudio.io import load_float32_image


def get_model(args):
    config = OmegaConf.load(args.ldm_config)
    model = load_model_from_config(config, args.diffusion_config)

    return model


parser = argparse.ArgumentParser()
parser.add_argument('--ldm_config', default="configs/stable-diffusion/v1-inference.yaml", type=str)
parser.add_argument('--diffusion_config', default="models/v1-5-pruned.ckpt", type=str)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--dir', type=str, default='./outputs/forest')
parser.add_argument('--ddim_steps', default=500, type=int)
parser.add_argument('--ddim_eta', default=0.0, type=float)
parser.add_argument('--ddim_scale', default=1.0, type=float)

args = parser.parse_args()

# Device setting
device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
print(f"Device set to {device_str}.")
device = torch.device(device_str)  

# Loading model
model = get_model(args)
sampler = DDIMSampler(model) # Sampling using DDIM

# Prepare Operator and noise
operator = InpaintingOperator(device=device)
noiser = get_noise(name='gaussian', sigma=0.01)
print("Operation: inpainting / Noise: gaussian")

# Prepare conditioning method
cond_method = get_conditioning_method('ps', model, operator, noiser, scale=0.5)
measurement_cond_fn = cond_method.conditioning
print("Conditioning sampler : resample")

# Instantiating sampler
sample_fn = partial(sampler.posterior_sampler, measurement_cond_fn=measurement_cond_fn, operator_fn=operator.forward,
                                        S=args.ddim_steps,
                                        cond_method='resample',
                                        conditioning=model.get_learned_conditioning([""]),
                                        ddim_use_original_steps=True,
                                        batch_size=1,
                                        shape=[4, 64, 64], # Dimension of latent space
                                        verbose=False,
                                        unconditional_guidance_scale=args.ddim_scale,
                                        unconditional_conditioning=None, 
                                        eta=args.ddim_eta,
                                        eps=1e-2,
                                        max_iters=500,
                                        )

mask = 1 - load_float32_image(Path(args.dir) / 'to_inpaint.png')[..., 1].unsqueeze(0).to(device)
ref_img = load_float32_image(Path(args.dir) / 'warped.png').to(device).permute(2, 0, 1).contiguous() * 2 - 1

# Do inference

print(f"Inference for image {args.dir}/warped.png")

# Exception) In case of inpainting
operator_fn = partial(operator.forward, mask=mask)
measurement_cond_fn = partial(cond_method.conditioning, mask=mask)
sample_fn = partial(sample_fn, measurement_cond_fn=measurement_cond_fn, operator_fn=operator_fn)

# Forward measurement model
y = operator_fn(ref_img)
# y_n = noiser(y)
y_n = y

# Sampling
samples_ddim, _ = sample_fn(measurement=y_n)

x_samples_ddim = model.decode_first_stage(samples_ddim.detach())

# Post-processing samples
label = clear_color(y_n)
recon = clear_color(x_samples_ddim)

# Saving images
plt.imsave(os.path.join(args.dir, 'resample_label.png'), label)
plt.imsave(os.path.join(args.dir, 'resample.png'), recon)
