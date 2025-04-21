import argparse
import os
from functools import partial

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from omegaconf import OmegaConf
from skimage.metrics import peak_signal_noise_ratio as psnr

from data.dataloader import get_dataloader, get_dataset
from ldm.models.diffusion.ddim import DDIMSampler
from ldm_inverse.condition_methods import get_conditioning_method
from ldm_inverse.measurements import get_noise, get_operator
from model_loader import load_model_from_config, load_yaml
from scripts.utils import clear_color, mask_generator


def get_model(args):
    config = OmegaConf.load(args.ldm_config)
    model = load_model_from_config(config, args.diffusion_config)

    return model


parser = argparse.ArgumentParser()
parser.add_argument('--ldm_config', default="configs/stable-diffusion/v1-inference.yaml", type=str)
parser.add_argument('--diffusion_config', default="models/v1-5-pruned.ckpt", type=str)
parser.add_argument('--task_config', default="configs/tasks/gaussian_deblur_config.yaml", type=str)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--save_dir', type=str, default='./results')
parser.add_argument('--ddim_steps', default=500, type=int)
parser.add_argument('--ddim_eta', default=0.0, type=float)
parser.add_argument('--n_samples_per_class', default=1, type=int)
parser.add_argument('--ddim_scale', default=1.0, type=float)

args = parser.parse_args()


# Load configurations
task_config = load_yaml(args.task_config)

# Device setting
device_str = "cuda:0" if torch.cuda.is_available() else 'cpu'
print(f"Device set to {device_str}.")
device = torch.device(device_str)  

# Loading model
model = get_model(args)
sampler = DDIMSampler(model) # Sampling using DDIM

# Prepare Operator and noise
measure_config = task_config['measurement']
operator = get_operator(device=device, **measure_config['operator'])
noiser = get_noise(**measure_config['noise'])
print(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")

# Prepare conditioning method
cond_config = task_config['conditioning']
cond_method = get_conditioning_method(cond_config['method'], model, operator, noiser, **cond_config['params'])
measurement_cond_fn = cond_method.conditioning
print(f"Conditioning sampler : {task_config['conditioning']['main_sampler']}")

# Instantiating sampler
sample_fn = partial(sampler.posterior_sampler, measurement_cond_fn=measurement_cond_fn, operator_fn=operator.forward,
                                        S=args.ddim_steps,
                                        cond_method=task_config['conditioning']['main_sampler'],
                                        conditioning=model.get_learned_conditioning(args.n_samples_per_class * [""]),
                                        ddim_use_original_steps=True,
                                        batch_size=args.n_samples_per_class,
                                        shape=[4, 64, 64], # Dimension of latent space
                                        verbose=False,
                                        unconditional_guidance_scale=args.ddim_scale,
                                        unconditional_conditioning=None, 
                                        eta=args.ddim_eta)

# Working directory
out_path = os.path.join(args.save_dir)
os.makedirs(out_path, exist_ok=True)
for img_dir in ['input', 'recon', 'progress', 'label']:
    os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

# Prepare dataloader
data_config = task_config['data']
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))] )
dataset = get_dataset(**data_config, transforms=transform)
loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)

# Exception) In case of inpainting, we need to generate a mask 
if measure_config['operator']['name'] == 'inpainting':
  mask_gen = mask_generator(**measure_config['mask_opt'])

# Do inference
for i, ref_img in enumerate(loader):

    print(f"Inference for image {i}")
    fname = str(i).zfill(3)
    ref_img = ref_img.to(device)

    # Exception) In case of inpainting
    if measure_config['operator'] ['name'] == 'inpainting':
      mask = mask_gen(ref_img)
      mask = mask[:, 0, :, :].unsqueeze(dim=0)
      operator_fn = partial(operator.forward, mask=mask)
      measurement_cond_fn = partial(cond_method.conditioning, mask=mask)
      sample_fn = partial(sample_fn, measurement_cond_fn=measurement_cond_fn, operator_fn=operator_fn)

      # Forward measurement model
      y = operator_fn(ref_img)
      y_n = noiser(y)

    else:
      y = operator.forward(ref_img)
      y_n = noiser(y).to(device)

    # Sampling
    samples_ddim, _ = sample_fn(measurement=y_n)
    
    x_samples_ddim = model.decode_first_stage(samples_ddim.detach())

    # Post-processing samples
    label = clear_color(y_n)
    reconstructed = clear_color(x_samples_ddim)
    true = clear_color(ref_img)

    # Saving images
    plt.imsave(os.path.join(out_path, 'input', fname+'_true.png'), true)
    plt.imsave(os.path.join(out_path, 'label', fname+'_label.png'), label)
    plt.imsave(os.path.join(out_path, 'recon', fname+'_recon.png'), reconstructed)

    psnr_cur = psnr(true, reconstructed)

    print('PSNR:', psnr_cur)
