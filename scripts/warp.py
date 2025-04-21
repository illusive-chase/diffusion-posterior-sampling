from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from rfstudio.engine.task import Task
from rfstudio.graphics import Cameras, RGBDImages, RGBDNImages
from rfstudio.graphics.math import safe_normalize
from rfstudio.io import (
    dump_float32_image,
    load_float32_image,
)
from rfstudio.visualization import Visualizer


@dataclass
class Script(Task):

    rgb: Path = ...
    depth: Path = ...
    normal: Path = ...
    output: Path = ...
    deg: int = 45
    vis: bool = False
    far: float = 1e3
    fov: float = 45

    def run(self) -> None:
        rgb = load_float32_image(self.rgb)
        H, W = rgb.shape[:2]
        if self.depth.suffix == '.npy':
            depth = torch.from_numpy(np.load(self.depth)).float().view(H, W, 1)
        elif self.depth.suffix == '.exr':
            import pyexr
            depth = torch.from_numpy(pyexr.read(self.depth)).float().view(H, W, 1)
        normal = safe_normalize(torch.from_numpy(np.load(self.normal)).float().view(H, W, 3))
        alpha = (depth < self.far) & (depth > 1e-3)
        image = RGBDNImages([torch.cat((rgb, depth, normal, alpha), dim=-1)]).to(self.device)
        for trial in [1, -1]:
            camera_from = Cameras.from_lookat(
                eye=(0, 0, 0),
                target=(0, 0, -1),
                resolution=(W, H),
                hfov_degree=self.fov,
                device=self.device,
            )
            pts = RGBDImages([torch.cat((rgb, depth * trial, alpha), dim=-1)]).to(self.device).deproject(camera_from)
            scale = pts.positions.view(-1, 3)[..., 2].mean().item()
            if scale > 0:
                break
        # print(scale)
        if self.vis:
            Visualizer().show(pts=pts)
        rad = torch.tensor(self.deg).to(rgb).deg2rad()
        camera_to = Cameras.from_lookat(
            eye=(scale * rad.sin(), 0, -scale * rad.cos() + scale),
            target=(scale * rad.sin(), 0, -scale * rad.cos()),
            resolution=(W, H),
            hfov_degree=self.fov,
            device=self.device,
        )
        result = image.warping(camera_from, camera_to).item()
        dump_float32_image(self.output, result)


if __name__ == '__main__':
    Script(cuda=0).run()
