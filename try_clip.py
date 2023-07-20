import torch
import tvm.relax as R
import tvm

mask_cond = torch.arange(mask.size(-1), device=device)
mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)


def test_new_ones():
    input_info = [([1, 2, 3], "float32")]

    class NewOnes(Module):
        def forward(self, x):
            return x.new_ones(1, 2, 3)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(x: R.Tensor((1, 2, 3), dtype="float32")) -> R.Tensor((1, 2, 3), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 2, 3), dtype="float32") = R.full(
                    (1, 2, 3), R.const(1, "float32"), dtype="float32"
                )
                gv: R.Tensor((1, 2, 3), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(NewOnes(), input_info, {}, expected1)
