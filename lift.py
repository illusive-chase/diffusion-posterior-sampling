from __future__ import annotations

import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import base64
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import pyexr
import rfviser
import rfviser.transforms as tf
import torch
from moge.model.v1 import MoGeModel
from torch import Tensor

from rfstudio.data import RelightDataset
from rfstudio.engine.task import Task, TaskGroup
from rfstudio.graphics import Cameras, DepthImages, RGBAImages, RGBDImages, RGBDNImages, RGBImages, TriangleMesh
from rfstudio.graphics.shaders import DepthShader
from rfstudio.io import dump_float32_image, load_float32_image, load_float32_masked_image
from rfstudio.utils.colormap import IntensityColorMap


def make_base64_image(
    img: Tensor,
    expected_size: int = 320 * 180,
) -> str:
    img = (img[..., [2, 1, 0]].contiguous().cpu().numpy() * 255).astype(np.uint8)
    h, w = img.shape[:2]
    rescale_factor = min(1, (expected_size / (w * h)) ** 0.5)
    w, h = int(w * rescale_factor), int(h * rescale_factor)
    base64_data = cv2.imencode(".jpg", cv2.resize(img, (w, h)))[1].tobytes()
    return base64.b64encode(base64_data).decode("ascii")

@dataclass
class Vis(Task):

    input: Path = ...

    port: int = 6789

    step: float = 0.1

    def run(self) -> None:
        server = rfviser.ViserServer(host='localhost', port=self.port)

        with pyexr.open(self.input / 'depth.exr') as file:
            depth_map = torch.from_numpy(file.get()).float().to(self.device)
        mask = load_float32_image(self.input / 'mask.png')[..., :1].to(self.device)
        rgb1 = load_float32_image(self.input / 'input.png').to(self.device)
        rgb2 = load_float32_image(self.input / 'warped.png', alpha_color=(1., 1., 1.)).to(self.device)

        camera1 = Cameras.from_lookat(
            eye=(0, 0, 1),
            resolution=(rgb1.shape[1], rgb1.shape[0]),
            device=self.device,
        )
        camera2 = Cameras.from_lookat(
            eye=(self.step, 0, 1 - self.step * 2),
            resolution=(rgb1.shape[1], rgb1.shape[0]),
            device=self.device,
        )
        pts = RGBDImages([torch.cat((rgb1, depth_map, mask), dim=-1)]).deproject(camera1)

        pose1 = tf.SE3.from_rotation_and_translation(
            tf.SO3.from_matrix(camera1.c2w[:3, :3].cpu().numpy()).multiply(tf.SO3.from_x_radians(np.pi)),
            camera1.c2w[:3, 3].cpu().numpy(),
        )
        server.scene.add_camera_frustum(
            '/cam/1',
            fov=np.pi / 2,
            aspect=4 / 3,
            color=(210 / 255, 143 / 255, 81 / 255),
            scale=0.005,
            wxyz=pose1.rotation().wxyz,
            position=pose1.translation()
        )
        pose2 = tf.SE3.from_rotation_and_translation(
            tf.SO3.from_matrix(camera2.c2w[:3, :3].cpu().numpy()).multiply(tf.SO3.from_x_radians(np.pi)),
            camera2.c2w[:3, 3].cpu().numpy(),
        )
        server.scene.add_camera_frustum(
            '/cam/2',
            fov=np.pi / 2,
            aspect=4 / 3,
            color=(210 / 255, 143 / 255, 81 / 255),
            scale=0.005,
            wxyz=pose2.rotation().wxyz,
            position=pose2.translation()
        )
        server.gui.add_image_viewer(
            images={
                '1': make_base64_image(rgb1),
                '2': make_base64_image(rgb2),
            },
            cameras={
                '1': pose1.as_matrix()[:3, :],
                '2': pose2.as_matrix()[:3, :],
            }
        )
        server.scene.add_point_cloud(
            '/pcd',
            points=pts.positions.cpu().numpy(),
            colors=pts.colors.cpu().numpy(),
            point_size=0.003,
        )
        try:
            while True:
                time.sleep(10)
        except KeyboardInterrupt:
            pass
        finally:
            server.stop()


@dataclass
class Real(Task):

    input: Path = ...
    output: Path = ...
    masked: bool = True
    colormap: Literal['viridis', 'plasma', 'inferno', 'magma', 'cividis'] = 'magma'
    geowizard: Path = Path('..') / 'GeoWizard'
    domain: Literal['object', 'indoor', 'outdoor'] = 'outdoor'
    res: int = 512
    fov: int = 35
    skip_infer: bool = False
    step: float = 0.1

    def run(self) -> None:

        assert not self.output.exists() or self.output.is_dir()
        self.output.mkdir(exist_ok=True, parents=True)

        # Read the input image and convert to tensor (3, H, W) and normalize to [0, 1]
        input_image = RGBImages([
            load_float32_image(self.input, alpha_color=(1., 1., 1.)).to(self.device)
        ]).resize_to(self.res, self.res)

        if not self.skip_infer:

            # Load the model from huggingface hub (or load from local).
            model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(self.device)

            # Infer
            output = model.infer(input_image.item().permute(2, 0, 1))
            # `output` has keys "points", "depth", "mask" and "intrinsics",
            # The maps are in the same size as the input image.
            # {
            #     "points": (H, W, 3),    # scale-invariant point map in OpenCV camera coordinate system (x right, y down, z forward)
            #     "depth": (H, W),        # scale-invariant depth map
            #     "mask": (H, W),         # a binary mask for valid pixels.
            #     "intrinsics": (3, 3),   # normalized camera intrinsics
            # }
            # For more usage details, see the `MoGeModel.infer` docstring.

            depth_map = output["depth"].unsqueeze(-1)
            mask = output["mask"].unsqueeze(-1)
            if not self.masked:
                mask = torch.ones_like(mask)

            dump_float32_image(self.output / 'depth.exr', depth_map * mask)
            dump_float32_image(self.output / 'mask.png', mask.repeat(1, 1, 3))
            dump_float32_image(self.output / 'input.png', input_image.item())
            vis_depth = DepthImages([
                torch.cat((depth_map.nan_to_num(0., 0., 0.), mask), dim=-1)
            ]).visualize(IntensityColorMap(style=self.colormap))
            dump_float32_image(self.output / 'depth.png', vis_depth.item())
            subprocess.run([
                'python', str(self.geowizard / 'geowizard' / 'run_infer_v2.py'),
                '--domain', self.domain,
                '--input_file', str(self.output / 'input.png'),
                '--output_dir', str(self.output),
                '--ensemble_size', '10',
                '--denoise_steps', '50',
                '--seed', '0',
            ], check=False)

            assert (self.output / 'normal.npy').exists() and (self.output / 'normal.png').exists()
            normal_map = torch.from_numpy(np.load(self.output / 'normal.npy')).float().to(self.device)

        else:

            with pyexr.open(self.output / 'depth.exr') as file:
                depth_map = torch.from_numpy(file.get()).float().to(self.device)
            normal_map = torch.from_numpy(np.load(self.output / 'normal.npy')).float().to(self.device)
            mask = load_float32_image(self.output / 'mask.png')[..., :1].to(self.device)

        camera1 = Cameras.from_lookat(
            eye=(0, 0, 1),
            resolution=(self.res, self.res),
            hfov_degree=self.fov,
            device=self.device,
        )
        camera2 = Cameras.from_lookat(
            eye=(self.step, 0, 1 + self.step * 2),
            resolution=(self.res, self.res),
            hfov_degree=self.fov,
            device=self.device,
        )
        warped, warpability = RGBDNImages([
            torch.cat((input_image.item(), depth_map, normal_map, mask), dim=-1)
        ]).warping_with_warpability(camera1, camera2)
        to_inpaint = (warped.item()[..., 3:] < 0.5).float()
        dump_float32_image(self.output / 'warped.png', warped.blend((1, 1, 1)).item())
        dump_float32_image(self.output / 'to_inpaint.png', to_inpaint.repeat(1, 1, 3))
        dump_float32_image(self.output / 'warpability.png', warpability.blend((1, 1, 1)).item())


@dataclass
class Synthetic(Task):

    dataset: RelightDataset = RelightDataset(path=Path('data') / 'tensoir' / 'hotdog')

    mesh: Path = Path('exports') / 'hotdog.ply'

    output: Path = ...

    res: int = 512

    view: int = 0

    offset: int = 20

    def run(self) -> None:
        assert not self.output.exists() or self.output.is_dir()
        self.output.mkdir(exist_ok=True, parents=True)
        self.dataset.to(self.device)
        mesh = TriangleMesh.from_file(self.mesh).to(self.device)
        mesh.replace_(vertices=mesh.vertices * (2 / 3))
        transformation = torch.tensor([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0],
        ]).float().to(self.device) @ torch.tensor([
            [0, 0, -1],
            [0, 1, 0],
            [1, 0, 0],
        ]).float().to(self.device)
        cameras = self.dataset.get_inputs(split='test')[...]
        cameras.replace_(
            c2w=torch.cat((
                transformation @ cameras.c2w[..., :3, :3],
                transformation @ cameras.c2w[..., :3, 3:],
            ), dim=-1)
        )
        cameras = cameras.resize(self.res / 800)
        source_view = (self.view + self.offset) % len(cameras)
        camera1 = cameras[self.view]
        camera2 = cameras[source_view]
        with pyexr.open(self.dataset.path / f'test_{self.view:03d}' / 'normal.exr') as file:
            normal = RGBImages([
                torch.from_numpy(file.get()).float().to(self.device)[..., :3]
            ]).resize_to(self.res, self.res).item()
        depth = mesh.render(camera1, shader=DepthShader(culling=False)).item()
        rgba = RGBAImages([
            load_float32_masked_image(self.dataset.path / f'test_{self.view:03d}' / 'rgba.png').to(self.device)
        ]).resize_to(self.res, self.res).item()
        nvs = RGBAImages([
            load_float32_masked_image(self.dataset.path / f'test_{source_view:03d}' / 'rgba.png').to(self.device)
        ]).resize_to(self.res, self.res)
        dump_float32_image(self.output / 'normal.exr', normal)
        dump_float32_image(self.output / 'depth.exr', depth[..., :1])
        dump_float32_image(self.output / 'input.png', rgba)
        warped, warpability = RGBDNImages([
            torch.cat((rgba[..., :3], depth[..., :1], normal, rgba[..., 3:] * depth[..., 1:]), dim=-1)
        ]).warping_with_warpability(camera1, camera2)
        to_inpaint = ((nvs.item()[..., 3:] > 0.5) & (warped.item()[..., 3:] < 0.5)).float()
        dump_float32_image(self.output / 'warped.png', warped.blend((1, 1, 1)).item())
        dump_float32_image(self.output / 'gt_warped.png', nvs.blend((1, 1, 1)).item())
        dump_float32_image(self.output / 'gt_warped_mask.png', nvs.item()[..., 3:].repeat(1, 1, 3))
        dump_float32_image(self.output / 'to_inpaint.png', to_inpaint.repeat(1, 1, 3))
        dump_float32_image(self.output / 'warpability.png', warpability.blend((1, 1, 1)).item())



if __name__ == '__main__':
    TaskGroup(
        hotdog=Synthetic(cuda=0),
        real=Real(cuda=0),
        vis=Vis(cuda=0),
    ).run()
