import tvm
import pickle

with open("deploy_mod.pkl", "rb") as f:
    mod_deploy = pickle.load(f)

def tune(mod: tvm.IRModule) -> None:
    from tvm import meta_schedule as ms

    ms.relax_integration.tune_relax(
        mod=mod,
        target=tvm.target.Target("apple/m1-gpu-restricted"),
        params={},
        builder=ms.builder.LocalBuilder(
            max_workers=6,
            timeout_sec=200,
        ),
        # runner=ms.runner.RPCRunner(
        #     ms.runner.RPCConfig(
        #         tracker_host="192.168.10.1",
        #         tracker_port=9191,
        #         tracker_key="m2-mac-mini",
        #         session_timeout_sec=50,
        #     )
        # ),
        runner = ms.runner.LocalRunner(timeout_sec=200),
        work_dir="log_db_tuning",
        max_trials_global=101000,
        max_trials_per_task=500,
        strategy=ms.search_strategy.EvolutionarySearch(init_min_unmeasured=10, max_fail_count=20),
    )

tune(mod_deploy)