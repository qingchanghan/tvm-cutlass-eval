import torch
import numpy as np

import tvm
from tvm import relay
import tvm.relay.testing
from tvm.relay.op.contrib.cutlass import partition_for_cutlass
from tvm.contrib.cutlass import (
    num_cutlass_partitions,
    finalize_modules,
)
from bert_rewrite import rewrite_reshape_gelu


def profile_and_build(mod, params, sm, tmp_dir="./tmp", lib_path="compile.so"):
    mod = partition_for_cutlass(mod)
    print(mod)

    num_cutlass_partition = num_cutlass_partitions(mod)
    host = tvm.target.Target("llvm")
    cuda = tvm.target.Target("cuda", host=host)
    cutlass = tvm.target.Target(
        {
            "kind": "cutlass",
            "sm": sm,
            "use_3xtf32": True,
            "split_k_slices": [1],
            "profile_all_alignments": False,
            "find_first_valid": False,
            "use_multiprocessing": True,
            "use_fast_math": False,
            "tmp_dir": tmp_dir,
        },
        host=host,
    )
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=[cuda, cutlass], params=params)
    lib = finalize_modules(lib, "compile.so", tmp_dir)
    dev = tvm.device("cuda", 0)
    rt_mod = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
    return rt_mod, dev, num_cutlass_partition


with open("models/bert_large.json", "r") as fi:
    mod = tvm.ir.load_json(fi.read())
with open("models/bert_large.params", "rb") as fi:
    params = relay.load_param_dict(fi.read())

mod = rewrite_reshape_gelu(mod)

sm  = 80
rt_mod, dev, num_partition = profile_and_build(mod, params, sm)
assert num_partition > 0

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
