from diffusers import DiffusionPipeline
import torch


import tvm
from tvm import relax
from tvm.relax.frontend.torch import dynamo_capture_subgraphs
from tvm.relax.frontend.torch import from_fx
from tvm.script import relax as R

import torch
from torch import fx

from web_stable_diffusion.utils import get_clip

print(tvm.__file__)


pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")


clip2 = get_clip(pipe)

is_import = True


if is_import:
    graph = fx.symbolic_trace(clip2)
    mod = from_fx(
        graph,
        [((1, 77), "int32")],
        keep_params_as_input=True,
    )
else:
    text_input_ids = torch.rand((1, 77)).to(torch.int32)
    out = clip2(text_input_ids)

    print(out)

    ref_out = pipe.text_encoder_2(text_input_ids)


    assert torch.allclose(out[0], ref_out[0], atol=1e-4)
    assert torch.allclose(out[1], ref_out[1], atol=1e-4)

    print("same result")