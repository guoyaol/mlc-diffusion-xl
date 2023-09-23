from web_stable_diffusion.models.transformer_2d import Transformer2DModel
import torch

model = Transformer2DModel(
                        10,
                        64,
                        in_channels=640,
                        num_layers=2,
                        cross_attention_dim=2048,
                        norm_num_groups=32,
                        use_linear_projection=True,
                        only_cross_attention=False,
                        upcast_attention=False,
                    )

print(model)

input = torch.rand((2, 640, 64, 64)).to(torch.float32)
encoder_hidden_states = torch.rand((2, 77, 2048)).to(torch.float32)

with torch.no_grad():
    ref_result = model(
    input,
    encoder_hidden_states=encoder_hidden_states,
    cross_attention_kwargs=None,
    attention_mask=None,
    encoder_attention_mask=None,
    return_dict=False,
    )[0]

print(ref_result.shape)

# import this model into TVM


# compare results with reference result

