import os
os.environ["REACTION_CHECKPOINT"] = r"C:\Users\zakar\homelab-data\checkpoints\8117bfe1-85ea-4074-a967-5ebeedd40698\best_model.pt"
from reaction.infer_core import load_model, model_info
load_model()
print(model_info())
