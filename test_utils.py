import os.path

import torch
from Dataset import sample_rays_np
from Render import render_rays
import numpy as np
from tqdm import tqdm
import imageio
import time
import os


def rot_phi(phi):
    return np.array([
        [1, 0, 0, 0],
        [0, np.cos(phi), -np.sin(phi), 0],
        [0, np.sin(phi), np.cos(phi), 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)


def rot_theta(th):
    return np.array([
        [np.cos(th), 0, -np.sin(th), 0],
        [0, 1, 0, 0],
        [np.sin(th), 0, np.cos(th), 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)


def generate_video_nearby(net, ref_dataset, bound, N_samples, device, eval_path, r=5.0):
    f = ref_dataset.f
    img_size = ref_dataset.img_size
    c2w = ref_dataset.c2w[0]
    frames = list()
    img_path = eval_path + "rgb/"
    os.makedirs(img_path, exist_ok=True)
    img_idx = 0
    for th in tqdm(np.linspace(-1.0, 1.0, 120, endpoint=False)):
        theta = rot_theta(r * np.sin(np.pi * 2.0 * th) / 180.0 * np.pi)
        phi = rot_phi(r * np.cos(np.pi * 2.0 * th) / 180.0 * np.pi)
        rgb = generate_frame(net, theta @ phi @ c2w, f, img_size, bound, N_samples, device, ref_dataset)
        rgb = (255 * np.clip(rgb.cpu().numpy(), 0, 1)).astype(np.uint8)
        imageio.imwrite(f'{img_path}{img_idx:03d}.png', rgb)
        frames.append(rgb)
        img_idx += 1

    imageio.mimwrite(eval_path+time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())+".mp4", frames, fps=30, quality=7)


@torch.no_grad()
def generate_frame(net, c2w, f, img_size, bound, N_samples, device, ref_dataset):
    rays_o, rays_d = sample_rays_np(img_size, img_size, f, c2w)
    rays_o = torch.tensor(rays_o, device=device)
    rays_d = torch.tensor(rays_d, device=device)
    img_lines = list()
    for i in range(rays_d.shape[0]):
        rays_od = (rays_o[i], rays_d[i])
        img_lines.append(render_rays(net, rays_od, bound=bound, N_samples=N_samples, device=device, ref=ref_dataset))

    return torch.cat([img[0].unsqueeze(dim=0) for img in img_lines], dim=0)
