# import torch
# from safetensors.torch import save_file


# for name, path_in, path_out in [
#     ("rapidparc",    "trained_models/rapidparc/model/snapshot_10000.pt",    "rapidparc.safetensors"),
#     ("rapidparc_v8", "trained_models/rapidparc_v8/model/snapshot_10000.pt", "rapidparc_v8.safetensors"),
#     ("hemiaug",      "trained_models/hemiaug/model/snapshot_10000.pt",       "hemiaug.safetensors"),
# ]:
#     checkpoint = torch.load(path_in, map_location="cpu")

#     # DDP state dict handling
#     if isinstance(checkpoint, dict):
#         if "model" in checkpoint:
#             state_dict = checkpoint["model"]
#         elif "state_dict" in checkpoint:
#             state_dict = checkpoint["state_dict"]
#         else:
#             state_dict = checkpoint 
#     else:
#         state_dict = checkpoint

#     state_dict = {
#         k.replace("module.", ""): v
#         for k, v in state_dict.items()
#     }

#     save_file(state_dict, path_out)
#     print(f"✓ {path_out}")


# from huggingface_hub import HfApi

# api = HfApi()

# files = [
#     ("rapidparc.safetensors",    "rapidparc.safetensors"),
#     ("rapidparc_args.yaml",      "rapidparc_args.yaml"),
#     ("rapidparc_v8.safetensors", "rapidparc_v8.safetensors"),
#     ("rapidparc_v8_args.yaml",   "rapidparc_v8_args.yaml"),
#     ("hemiaug.safetensors",      "hemiaug.safetensors"),
#     ("hemiaug_args.yaml",        "hemiaug_args.yaml"),
# ]

# for local, remote in files:
#     api.upload_file(
#         path_or_fileobj=local,
#         path_in_repo=remote,
#         repo_id="Valentin-von-Bornhaupt/RapidParc",
#         repo_type="model",
#     )


import torch
import numpy as np 
from huggingface_hub import HfApi
from safetensors.torch import save_file

api = HfApi()
# mapping = torch.from_numpy(np.load("src/rapidparc/utils/mapping_from_800_800_to_43.npy"))
# torch.save(mapping, "mapping_from_800_800_to_43.pt" )
# api.upload_file(
#         path_or_fileobj="mapping_from_800_800_to_43.pt",
#         path_in_repo="mapping_from_800_800_to_43.pt",
#         repo_id="Valentin-von-Bornhaupt/RapidParc",
#     )

# api.upload_file(
#         path_or_fileobj="src/rapidparc/utils/int_to_label.npy",
#         path_in_repo="int_to_label.npy",
#         repo_id="Valentin-von-Bornhaupt/RapidParc",
#     )


api.upload_file(
        path_or_fileobj="src/rapidparc/utils/int_to_label.npy",
        path_in_repo="splits/.npy",
        repo_id="Valentin-von-Bornhaupt/RapidParc",
    )