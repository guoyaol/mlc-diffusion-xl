from torch import nn
import torch
from tvm.relax.frontend.torch import dynamo_capture_subgraphs

torch.manual_seed(42)

class my_model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, last_hidden_state, input_ids):

        index1 = torch.arange(last_hidden_state.shape[0])
        index2 = input_ids.argmax(dim=-1)

        # pooled_output = last_hidden_state[index1, index2, ]
        # squeezed_hidden_states = last_hidden_state.squeeze(0)
        pooled_output = last_hidden_state[torch.arange(1), ]

        return pooled_output


text_input_ids = torch.rand((1, 77, 1280))
random_ids = torch.rand((1, 77))
# print(text_input_ids)
# Capture CLIP's computational graph.

model = my_model()



output = model(text_input_ids, random_ids)
print("model output", output.shape)

mod = dynamo_capture_subgraphs(
    model.forward,
    text_input_ids,
    random_ids,
    keep_params_as_input=True,
)

print(mod)