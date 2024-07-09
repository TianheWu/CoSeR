"""make variations of input image"""

import argparse, os, sys, glob
import PIL
import torch
import numpy as np
import torchvision
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torchvision import transforms
from torch import autocast
from contextlib import nullcontext
import time
from pytorch_lightning import seed_everything

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
import math
import copy
from scripts.wavelet_color_fix import wavelet_reconstruction, adaptive_instance_normalization
import torch.nn.functional as F
from util_image import ImageSpliterTh

def space_timesteps(num_timesteps, section_counts):
	"""
	Create a list of timesteps to use from an original diffusion process,
	given the number of timesteps we want to take from equally-sized portions
	of the original process.
	For example, if there's 300 timesteps and the section counts are [10,15,20]
	then the first 100 timesteps are strided to be 10 timesteps, the second 100
	are strided to be 15 timesteps, and the final 100 are strided to be 20.
	If the stride is a string starting with "ddim", then the fixed striding
	from the DDIM paper is used, and only one section is allowed.
	:param num_timesteps: the number of diffusion steps in the original
						  process to divide up.
	:param section_counts: either a list of numbers, or a string containing
						   comma-separated numbers, indicating the step count
						   per section. As a special case, use "ddimN" where N
						   is a number of steps to use the striding from the
						   DDIM paper.
	:return: a set of diffusion steps from the original process to use.
	"""
	if isinstance(section_counts, str):
		if section_counts.startswith("ddim"):
			desired_count = int(section_counts[len("ddim"):])
			for i in range(1, num_timesteps):
				if len(range(0, num_timesteps, i)) == desired_count:
					return set(range(0, num_timesteps, i))
			raise ValueError(
				f"cannot create exactly {num_timesteps} steps with an integer stride"
			)
		section_counts = [int(x) for x in section_counts.split(",")]   #[250,]
	size_per = num_timesteps // len(section_counts)
	extra = num_timesteps % len(section_counts)
	start_idx = 0
	all_steps = []
	for i, section_count in enumerate(section_counts):
		size = size_per + (1 if i < extra else 0)
		if size < section_count:
			raise ValueError(
				f"cannot divide section of {size} steps into {section_count}"
			)
		if section_count <= 1:
			frac_stride = 1
		else:
			frac_stride = (size - 1) / (section_count - 1)
		cur_idx = 0.0
		taken_steps = []
		for _ in range(section_count):
			taken_steps.append(start_idx + round(cur_idx))
			cur_idx += frac_stride
		all_steps += taken_steps
		start_idx += size
	return set(all_steps)

def load_model_from_config(config, ckpt, verbose=False):
	if ckpt:
		print(f"Loading model from {ckpt}")
		pl_sd = torch.load(ckpt, map_location="cpu")
		if "global_step" in pl_sd:
			print(f"Global Step: {pl_sd['global_step']}")
		sd = pl_sd["state_dict"]
	model = instantiate_from_config(config.model)
	if ckpt:
		m, u = model.load_state_dict(sd, strict=False)
		if len(m) > 0 and verbose:
			print("missing keys:")
			print(m)
		if len(u) > 0 and verbose:
			print("unexpected keys:")
			print(u)

	model.cuda()
	model.eval()
	return model


def main():
	parser = argparse.ArgumentParser()

	parser.add_argument(
		"--inputdir",
		type=str,
		help="dir of the input images",
	)
	parser.add_argument(
		"--outdir",
		type=str,
		help="dir to write results to",
		default='results/'
	)
	parser.add_argument(
		"--ddpm_steps",
		type=int,
		default=50,
		help="number of ddpm sampling steps",
	)
	parser.add_argument(
		"--config",
		type=str,
		default="configs/CoSeR/aia_512_imagenet_all_caption_clip_atten_ref.yaml",
		help="path to config which constructs model",
	)
	parser.add_argument(
		"--load_ckpt",
		type=str,
		help="path to checkpoint of model",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=2887,
		help="the seed (for reproducible sampling)",
	)
	parser.add_argument(
		"--precision",
		type=str,
		help="evaluate at this precision",
		choices=["full", "autocast"],
		default="full"
	)
	parser.add_argument(
		"--cfg",
		type=float,
		default=3.0,
		help="cfg rate",
	)
	parser.add_argument(
		"--cfg_ref",
		type=float,
		default=3.0,
		help="cfg rate in reference generation",
	)
	parser.add_argument(
		"--neg_prompt",
		type=str,
		help="negative prompts",
		default="worst quality, normal quality, low quality, low res, blurry, text, watermark, logo, banner, extra digits, cropped, jpeg artifacts, signature, username, error, sketch ,duplicate, ugly, monochrome, horror, geometry, mutation, disgusting",
	)
	parser.add_argument(
		"--user_prompt",
		type=str,
		help="user-defined prompts",
		default=None,
	)
	parser.add_argument(
		"--input_size",
		type=int,
		default=512,
		help="input size",
	)
	parser.add_argument(
		"--dec_w",
		type=float,
		default=0.5,
		help="weight for combining VQGAN and Diffusion",
	)
	parser.add_argument(
		"--upscale",
		type=float,
		default=4.0,
		help="upsample scale",
	)
	parser.add_argument(
		"--colorfix_type",
		type=str,
		default="nofix",
		help="Color fix type to adjust the color of HR result according to LR input: adain (used in paper); wavelet; nofix",
	)
	parser.add_argument(
		"--wocfw",
		action='store_true',
	)
	parser.add_argument(
		"--vqgan_ckpt",
		type=str,
		default=None,
		help="path to checkpoint of VQGAN model",
	)
	parser.add_argument(
		"--save_ref",
		action='store_true',
	)

	opt = parser.parse_args()
	seed_everything(opt.seed)

	print('>>>>>>>>>>color correction>>>>>>>>>>>')
	if opt.colorfix_type == 'adain':
		print('Use adain color correction')
	elif opt.colorfix_type == 'wavelet':
		print('Use wavelet color correction')
	else:
		print('No color correction')
	print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

	# load model
	config = OmegaConf.load(f"{opt.config}")
	model = load_model_from_config(config, f"{opt.load_ckpt}")
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	model = model.to(device)

	model.configs = config

	if not opt.wocfw:
		vqgan_config = OmegaConf.load("configs/autoencoder/autoencoder_kl_64x64x4_resi.yaml")
		vq_model = load_model_from_config(vqgan_config, opt.vqgan_ckpt)
		vq_model = vq_model.to(device)
		vq_model.decoder.fusion_w = opt.dec_w

	# set inputdir and outputdir
	os.makedirs(opt.outdir, exist_ok=True)
	os.makedirs(str(opt.outdir).rstrip('/')+'_gen', exist_ok=True)

	names_list = os.listdir(opt.inputdir)
	temp_list = []
	for name in names_list:
		if not os.path.exists(os.path.join(opt.outdir, name)):
			temp_list.append(name)
	names_list = temp_list
	print(f"Found {len(names_list)} inputs.")

	# set ddpm sampling strategy, following stablesr
	model.register_schedule(given_betas=None, beta_schedule="linear", timesteps=1000,
						  linear_start=0.00085, linear_end=0.0120, cosine_s=8e-3)
	model.num_timesteps = 1000

	sqrt_alphas_cumprod = copy.deepcopy(model.sqrt_alphas_cumprod)
	sqrt_one_minus_alphas_cumprod = copy.deepcopy(model.sqrt_one_minus_alphas_cumprod)

	use_timesteps = set(space_timesteps(1000, [opt.ddpm_steps]))
	last_alpha_cumprod = 1.0
	new_betas = []
	timestep_map = []
	for i, alpha_cumprod in enumerate(model.alphas_cumprod):
		if i in use_timesteps:
			new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
			last_alpha_cumprod = alpha_cumprod
			timestep_map.append(i)
	new_betas = [beta.data.cpu().numpy() for beta in new_betas]
	model.register_schedule(given_betas=np.array(new_betas), timesteps=len(new_betas))
	model.num_timesteps = 1000
	model.ori_timesteps = list(use_timesteps)
	model.ori_timesteps.sort()
	model = model.to(device)

	precision_scope = autocast if opt.precision == "autocast" else nullcontext
	
	# main loop
	trans = transforms.Compose([transforms.ToTensor()])
	with torch.no_grad():
		with precision_scope("cuda"):
			with model.ema_scope():
				for name in names_list:
					# load image
					image = trans(Image.open(os.path.join(opt.inputdir, name)).convert('RGB')).unsqueeze(0).cuda()
					image_ps = model.pre_sr_model(image)

					cur_image = image * 2.0 - 1.0
					cur_image = F.interpolate(cur_image, size=(int(cur_image.size(-2)*opt.upscale), int(cur_image.size(-1)*opt.upscale)), mode='bicubic')
					cur_image = cur_image.clamp(-1, 1)
					im_lq_bs = cur_image

					seed_everything(opt.seed)

					# move lr to latent space
					if model.fixed_cond:
						init_latent = model.get_first_stage_encoding(model.encode_first_stage(im_lq_bs).mode())
					else:
						init_latent = model.get_first_stage_encoding(model.encode_first_stage(im_lq_bs))

					# generate cognitive embedding
					clip_image_features = model.cond_stage_model.encode_with_vision_transformer(model.clip_transfrom(image_ps))

					cog_embed = model.global_adapter(clip_image_features)

					# user-defined prompt embedding & negative prompt embedding
					if opt.user_prompt is not None:
						text_init = opt.user_prompt
						print(f"Processing: {name}, user-defined prompt: {text_init}.")
					else:
						text_init = ['']
						print(f"Processing: {name}, no user-defined prompt.")

					semantic_user, _ = model.cond_stage_model(text_init)
					text_ne = opt.neg_prompt
					semantic_neg, _ = model.cond_stage_model(text_ne)

					# generate noise map
					noise = torch.randn_like(init_latent)
					t = repeat(torch.tensor([999]), '1 -> b', b=im_lq_bs.size(0))
					t = t.to(device).long()
					x_T = model.q_sample_respace(x_start=init_latent, t=t, sqrt_alphas_cumprod=sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod, noise=noise)

					# reference image generation
					condition_dic = {'prompt_emb': semantic_user, 'lr_prompt_emb': cog_embed}
					condition_dic_ne = {'prompt_emb': None, 'lr_prompt_emb': semantic_neg}

					ref_samples, _ = model.sample(cond=condition_dic, cond_ne=condition_dic_ne, cfg=opt.cfg_ref, batch_size=im_lq_bs.size(0), timesteps=opt.ddpm_steps, time_replace=opt.ddpm_steps, x_T=x_T, return_intermediates=True, gen_mode=True)

					# do super-resolution
					init_latent_zero = model.get_first_stage_encoding(model.encode_first_stage(torch.zeros(im_lq_bs.shape, device=im_lq_bs.device)).mode())
					condition_dic = {'prompt_emb': semantic_user, 'lr_prompt_emb': cog_embed, 'lr': init_latent, 'reference': ref_samples}
					condition_dic_ne = {'prompt_emb': None, 'lr_prompt_emb': semantic_neg, 'lr': init_latent, 'reference': init_latent_zero}

					samples, _ = model.sample(cond=condition_dic, cond_ne=condition_dic_ne, cfg=opt.cfg, batch_size=im_lq_bs.size(0), timesteps=opt.ddpm_steps, time_replace=opt.ddpm_steps, x_T=x_T, return_intermediates=True)

					# cfw and color correction from stablesr
					if not opt.wocfw:
						_, enc_fea_lq = vq_model.encode(im_lq_bs)
						x_samples = vq_model.decode(samples * 1. / model.scale_factor, enc_fea_lq)
					else:
						x_samples = model.decode_first_stage(samples)
					if opt.colorfix_type == 'adain':
						x_samples = adaptive_instance_normalization(x_samples, im_lq_bs)
					elif opt.colorfix_type == 'wavelet':
						x_samples = wavelet_reconstruction(x_samples, im_lq_bs)

					# save results
					im_sr = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
					im_sr = im_sr.cpu().numpy().transpose(0,2,3,1)*255

					ref_x_samples = model.decode_first_stage(ref_samples)
					im_ref = torch.clamp((ref_x_samples + 1.0) / 2.0, min=0.0, max=1.0)
					im_ref = im_ref.cpu().numpy().transpose(0,2,3,1)*255

					outpath = f"{opt.outdir}/{name.split('.')[0]}.png"
					Image.fromarray(im_sr[0].astype(np.uint8)).save(outpath)

					outpath = f"{opt.outdir.rstrip('/')}_gen/{name.split('.')[0]}.png"
					Image.fromarray(im_ref[0].astype(np.uint8)).save(outpath)

	print(f"Done.")


if __name__ == "__main__":
	main()
