from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rfstudio.engine.task import Task
from rfstudio.graphics import RGBAImages, RGBImages
from rfstudio.io import (
    dump_float32_image,
    load_float32_image,
    load_float32_masked_image,
)


@dataclass
class Script(Task):

    input: Path = ...

    output: Path = ...

    rgba: bool = False

    def run(self) -> None:
        if self.rgba:
            img = load_float32_masked_image(self.input)
            assert img.shape[0] == img.shape[1]
            dump_float32_image(self.output, RGBAImages([img]).resize_to(256, 256).blend((1, 1, 1)).item())
        else:
            img = load_float32_image(self.input)
            assert img.shape[0] == img.shape[1]
            dump_float32_image(self.output, RGBImages([img]).resize_to(256, 256).item())


if __name__ == '__main__':
    Script(cuda=0).run()
