from torch import nn
import torch
from tvm.relax.frontend.torch import dynamo_capture_subgraphs

class my_model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, last_hidden_state):
        index2 = last_hidden_state.argmax(dim=-1)
        print("index2")
        print(index2)
        pooled_output = last_hidden_state[[0],last_hidden_state.argmax(dim=-1)]
        return pooled_output


text_input_ids = torch.rand((1, 77))
# print(text_input_ids)
# Capture CLIP's computational graph.

model = my_model()



# output = model(text_input_ids)
# print(output)

mod = dynamo_capture_subgraphs(
    model.forward,
    text_input_ids,
    keep_params_as_input=True,
)