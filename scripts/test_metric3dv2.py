from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from rfstudio.engine.task import Task
from rfstudio.io import load_float32_image


@dataclass
class Script(Task):

    input: Path = ...

    def run(self) -> None:
        model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_large', pretrain=True)
        rgb = load_float32_image(self.input)
        pred_depth, confidence, output_dict = model.inference({'input': rgb})
        pred_normal = output_dict['prediction_normal'][:, :3, :, :] # only available for Metric3Dv2 i.e., ViT models
        normal_confidence = output_dict['prediction_normal'][:, 3, :, :] # see https://arxiv.org/abs/2109.09881 for details


if __name__ == '__main__':
    Script(cuda=0).run()
