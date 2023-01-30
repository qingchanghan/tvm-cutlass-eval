import tempfile
import torch
import numpy as np

import tvm
from tvm import relay
import tvm.relay.testing
from tvm import meta_schedule as ms
from bert_rewrite import rewrite_reshape_gelu


def build_by_ms(mod, params):
    target = tvm.target.Target("nvidia/nvidia-a10")
    def convert_layout(mod):
        seq = tvm.transform.Sequential(
            [relay.transform.ConvertLayout({"nn.conv2d": ["NHWC", "OHWI"]})]
        )
        with tvm.transform.PassContext(opt_level=3):
            mod = seq(mod)
        return mod

    with tempfile.TemporaryDirectory() as work_dir:
        with ms.Profiler() as profiler:
            converted_mod = convert_layout(mod)
            database = ms.relay_integration.tune_relay(
                mod=converted_mod,
                target=target,
                work_dir=work_dir,
                max_trials_global=3000,
                params=params,
            )
            lib = ms.relay_integration.compile_relay(
                database=database,
                mod=converted_mod,
                target=target,
                params=params,
            )
        print(profiler.table())
    dev = tvm.device("cuda", 0)
    rt_mod = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
    return rt_mod, dev


with open("models/bert_large.json", "r") as fi:
    mod = tvm.ir.load_json(fi.read())
with open("models/bert_large.params", "rb") as fi:
    params = relay.load_param_dict(fi.read())

mod = rewrite_reshape_gelu(mod)

rt_mod, dev = build_by_ms(mod, params)

batch_size = 8
inputs = (torch.randint(high=100, size=(batch_size, 128), dtype=torch.int64),
          torch.randint(high=100, size=(batch_size, 128), dtype=torch.int64),
          torch.randint(high=100, size=(batch_size, 128), dtype=torch.int64))

np.save("input_ids", inputs[0].numpy())
np.save("attention_mask", inputs[1].numpy())
np.save("token_type_ids", inputs[2].numpy())

rt_mod.set_input("input_ids", inputs[0].numpy())
rt_mod.set_input("attention_mask", inputs[1].numpy())
rt_mod.set_input("token_type_ids", inputs[2].numpy())

print("Evaluate inference time cost...")
print(rt_mod.benchmark(dev, number=1, repeat=100, end_to_end=True))
