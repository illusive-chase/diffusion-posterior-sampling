import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import argparse
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf
from rfstudio.io import load_float32_image

from ldm.models.diffusion.ddim import DDIMSampler
from ldm_inverse.measurements import InpaintingOperator, get_noise
from model_loader import load_model_from_config
from scripts.utils import clear_color


def get_model(args):
    config = OmegaConf.load(args.ldm_config)
    model = load_model_from_config(config, args.diffusion_config)

    return model


parser = argparse.ArgumentParser()
parser.add_argument('--ldm_config', default="configs/stable-diffusion/v1-inference.yaml", type=str)
parser.add_argument('--diffusion_config', default="models/v1-5-pruned.ckpt", type=str)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--dir', type=str, default='./outputs/forest')
parser.add_argument('--name', '-n', type=str, default='psld')
parser.add_argument('--ddim_steps', default=500, type=int)
parser.add_argument('--ddim_eta', default=0.0, type=float)
parser.add_argument('--ddim_scale', default=1.0, type=float)
parser.add_argument('--a', '-a', default=1, type=float, help="anneal: a*exp(-b*r)")
parser.add_argument('--b', '-b', default=1, type=float, help="anneal: a*exp(-b*r)")
parser.add_argument('--gamma', default=0.1, type=float, help="inpainting error")
parser.add_argument('--omega', default=0.1, type=float, help="measurement error")

args = parser.parse_args()

# Device setting
device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
print(f"Device set to {device_str}.")
device = torch.device(device_str)  

# Loading model
model = get_model(args)
sampler = DDIMSampler(model, a=args.a, b=args.b) # Sampling using DDIM

# Prepare Operator and noise
operator = InpaintingOperator(device=device)
noiser = get_noise(name='gaussian', sigma=0.01)
print("Operation: inpainting / Noise: gaussian")

print("Conditioning sampler : psld")

mask = 1 - load_float32_image(Path(args.dir) / 'to_inpaint.png')[..., 1].unsqueeze(0).to(device)
ref_img = load_float32_image(Path(args.dir) / 'warped.png').to(device).permute(2, 0, 1).contiguous() * 2 - 1

# Do inference

print(f"Inference for image {args.dir}/warped.png")

# Exception) In case of inpainting
operator_fn = partial(operator.forward, mask=mask)

# Forward measurement model
y = operator_fn(ref_img)
# y_n = noiser(y)
y_n = y

# Sampling

with model.ema_scope():
    samples_ddim, _ = sampler.sample(S=args.ddim_steps,
                                    batch_size=1,
                                    shape=[4, 64, 64],
                                    verbose=False,
                                    conditioning=model.get_learned_conditioning([""]),
                                    unconditional_guidance_scale=args.ddim_scale,
                                    unconditional_conditioning=None,
                                    eta=args.ddim_eta,
                                    x_T=None,
                                    ip_mask = mask,
                                    measurements = y_n,
                                    operator = operator,
                                    gamma = args.gamma,
                                    inpainting = True,
                                    omega = args.omega,
                                    general_inverse=False,
                                    noiser=noiser)

x_samples_ddim = model.decode_first_stage(samples_ddim.detach())

# Post-processing samples
label = clear_color(y_n)
recon = clear_color(x_samples_ddim)

# Saving images
plt.imsave(os.path.join(args.dir, 'psld_label.png'), label)
plt.imsave(os.path.join(args.dir, f'{args.name}.png'), recon)
