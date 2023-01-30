import torch
import numpy as np
import os
import tvm
from tvm import relay
import tvm.relay.testing
from tvm import auto_scheduler
from bert_rewrite import rewrite_reshape_gelu


def profile_and_build(mod, params, sm, tmp_dir="./tmp", lib_path="compile.so", num_trials=3000):
    host = tvm.target.Target("llvm")
    cuda = tvm.target.Target("cuda", host=host)

    # extract tasks
    with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
        tasks, task_weights = auto_scheduler.extract_tasks(
                mod, params, cuda, include_simple_tasks=True, opt_level=3)
    for idx, (task, task_weight) in enumerate(zip(tasks, task_weights)):
        print(f"==== Task {idx}: {task.desc} (weight {task_weight} key: {task.workload_key}) =====")
        print(task.compute_dag)

    # auto-tuning
    os.makedirs("logs", exist_ok=True)
    log_file = "logs/ansor.log"
    measure_ctx = auto_scheduler.LocalRPCMeasureContext(
        repeat=3, min_repeat_ms=200, timeout=10
    )
    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tuner.tune(
        auto_scheduler.TuningOptions(
            num_measure_trials=num_trials,
            runner=measure_ctx.runner,
            measure_callbacks=[
                auto_scheduler.RecordToFile(log_file),
            ],
        )
    )

    with auto_scheduler.ApplyHistoryBest(log_file):
        with tvm.transform.PassContext(
            opt_level=3,
            config={"relay.backend.use_auto_scheduler": True},
        ):
            lib = relay.build(
                mod,
                target=cuda,
                target_host=host,
                params=params,
            )
    dev = tvm.device("cuda", 0)
    rt_mod = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
    return rt_mod, dev


with open("models/bert_large.json", "r") as fi:
    mod = tvm.ir.load_json(fi.read())
with open("models/bert_large.params", "rb") as fi:
    params = relay.load_param_dict(fi.read())

mod = rewrite_reshape_gelu(mod)

sm  = 80
rt_mod, dev = profile_and_build(mod, params, sm)

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
