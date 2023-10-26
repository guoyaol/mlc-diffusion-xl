import tvm
import pickle
from tvm.ir.module import IRModule

with open("deploy_mod.pkl", "rb") as f:
    mod_deploy = pickle.load(f)

with open("mod_script.py", "w") as f:
    print(mod_deploy.script(show_meta=True), file = f)
# mod_deploy.show()