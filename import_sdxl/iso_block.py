from web_stable_diffusion.models.attention import BasicTransformerBlock
import torch

inner_dim = 640
num_attention_heads = 10
attention_head_dim = 64
dropout = 0.0
cross_attention_dim = 2048
activation_fn = "geglu"
num_embeds_ada_norm = None
attention_bias = False
only_cross_attention = False
upcast_attention = False
norm_type = "layer_norm"
norm_elementwise_affine = True

model = BasicTransformerBlock(
    inner_dim,
    num_attention_heads,
    attention_head_dim,
    dropout=dropout,
    cross_attention_dim=cross_attention_dim,
    activation_fn=activation_fn,
    num_embeds_ada_norm=num_embeds_ada_norm,
    attention_bias=attention_bias,
    only_cross_attention=only_cross_attention,
    upcast_attention=upcast_attention,
    norm_type=norm_type,
    norm_elementwise_affine=norm_elementwise_affine,
)

input1 = torch.rand((2, 4096, 640)).to(torch.float32)
input2 = torch.rand((2, 77, 2048)).to(torch.float32)

print("referece result")
ref_out = model(input1, attention_mask = None, encoder_hidden_states = input2,
            timestep = None, cross_attention_kwargs = None, class_labels = None)

# input_size torch.Size([2, 4096, 640])
# attention_mask None
# encoder_hidden_states torch.Size([2, 77, 2048])
# encoder_attention_mask None
# timestep None
# cross_attention_kwargs None
# class_labels None
print(ref_out)

# import this model into TVM


# compare results with reference result

