import sys

print("python", sys.version)
try:
    import torch

    print("cuda_available", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device", torch.cuda.get_device_name(0))
except Exception as e:
    print("torch_error", e)
    raise

try:
    import bitsandbytes as bnb

    print("bitsandbytes", bnb.__version__)
except Exception as e:
    print("bnb_error", e)

try:
    from transformers import AutoModelForCausalLM

    print("transformers_ok")
except Exception as e:
    print("transformers_error", e)
