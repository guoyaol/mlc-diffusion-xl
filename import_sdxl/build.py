import tvm
from tvm import meta_schedule as ms
from tvm import relax

import pickle

# mod_deploy = pickle.load(open("dist/before_scheduling.pkl", "rb"))

# with open("dist/before_scheduling.py", "w") as f:
#     print(mod_deploy.script(show_meta=True), file=f)

from dist.before_scheduling import Module

mod_deploy = Module


target = tvm.target.Target(
            "webgpu", host="llvm -mtriple=wasm32-unknown-unknown-wasm"
        )

# # with target, tvm.transform.PassContext(opt_level=3):
# #     mod_deploy = tvm.tir.transform.DefaultGPUSchedule()(mod_deploy)
db = ms.database.create(work_dir="log_db_prune_main")
with target, db, tvm.transform.PassContext(opt_level=3):
    mod_deploy = relax.transform.MetaScheduleApplyDatabase(enable_warning=True)(mod_deploy)
    mod_deploy = tvm.tir.transform.DefaultGPUSchedule()(mod_deploy)

# # with open("dist/after_scheduling.py", "w") as f:
# #     print(mod_deploy.script(show_meta=True), file=f)

# print("start transforming")
# mod = tvm.tir.transform.ForceNarrowIndexToInt32()(mod_deploy)

# with open("dist/after_narrow.py", "w") as f:
#     print(mod.script(show_meta=True), file=f)

ex = relax.build(mod=mod_deploy, target=target)
ex.export_library("dist/stable_diffusio_xl.wasm")