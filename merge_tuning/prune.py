from tvm import meta_schedule as ms

db = ms.database.create(work_dir="sdxl_tune")

db_prune = ms.database.create(work_dir="log_db_prune")

print("start pruning")
db.dump_pruned(db_prune)