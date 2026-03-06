from typing import Union
from pathlib import Path
import tarfile
import pooch
import os
import numpy as np
import torch
from .args_from_yaml import load_args_from_yaml
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from .model import TransformerModel, get_embedding_layer

PRETRAINED_MODELS = ["rapidparc", "rapidparc_v8", "hemiaug"]
REPO_ID = "Valentin-von-Bornhaupt/RapidParc"

def load_model_and_args_local_or_hf(name_or_path: Union[str, os.PathLike], device: torch.device):

    if isinstance(name_or_path, str):
        if name_or_path in PRETRAINED_MODELS:
            ########## download args and model ##########
            path_args = hf_hub_download(
                repo_id=REPO_ID,
                filename=name_or_path +"_args.yaml",
            )
            args = load_args_from_yaml(path_args)

            ########## define model architecture ##########
            n_vertices = getattr(args, "n_vertices", None)
            if n_vertices is None:
                n_vertices = getattr(args, "num_support_points_per_streamline", None)
            if n_vertices is None:
                raise AttributeError(
                    "Configuration is missing 'n_vertices'. Please add it to args.yml or provide "
                    "'num_support_points_per_streamline'."
                )

            embedding = get_embedding_layer(
                num_support_points_per_streamline=n_vertices,
                d_model=args.d_model)

            model = TransformerModel(num_layers=args.num_layers, 
                             d_model=args.d_model,
                             nhead=args.nhead,
                             embedding_layer=embedding,
                             dim_feedforward=args.dim_feedforward,
                             dropout=args.dropout,
                             dim_class_hidden=args.dim_class_hidden,
                             dim_out=args.dim_out)
            
            ########## load weights of the model ##########
            path_model = hf_hub_download(
                repo_id=REPO_ID,
                filename=name_or_path +".safetensors",
            )
            state_dict = load_file(path_model) 
            model = model.to(device)  # deine Modellklasse
            model.load_state_dict(state_dict)
            model.eval()
            return model, args
        
    raise NotImplementedError()


def get_mapping_800_800_to_43_hf() -> torch.Tensor:
    path = hf_hub_download(
                repo_id=REPO_ID,
                filename="mapping_from_800_800_to_43.pt")
    return torch.load(path)


def get_int_to_label_hf() -> np.ndarray:
    path = hf_hub_download(
                repo_id=REPO_ID,
                filename="int_to_label.npy")
    return np.load(path)



def get_tractCloud_dataset_path(consent_given: bool = False) -> Path:
    cache_dir = Path(pooch.os_cache("RapidParc"))
    extracted_folder = cache_dir / "TrainData_800clu800ol"

    # If the Dataset is downloaded, user already accepted licence agreement 
    if extracted_folder.exists() and extracted_folder.is_dir():
        return extracted_folder

    print("\nFor running this code, you have to download the TractCloud dataset. \nPlease check out the Slicer Licence agreement before downloading the dataset:\n")
    print("\thttps://github.com/SlicerDMRI/TractCloud/blob/main/LICENSE\n")
    
    while not consent_given:
        answer = input("Do you accept the license agreement? [y/n]: ").strip().lower()
        if answer in ("y", "yes"):
            consent_given = True
        elif answer in ("n", "no"):
            raise PermissionError("Download aborted: License agreement not accepted.")
        print("Invalid input. Please answer with 'y' or 'n'.")

    # Run download:
    tar_url = "https://github.com/SlicerDMRI/TractCloud/releases/download/v1.0.0/TrainData_800clu800ol.tar.gz"
    xlsx_url = "https://github.com/SlicerDMRI/TractCloud/raw/2c2869cde12b8d617d21be8e19fd33ae73b5f3bc/datasets/FiberClusterAnnotation_Updated20230110.xlsx"

    tar_path = pooch.retrieve(
        url=tar_url, 
        known_hash="72d70d4cf7a465a824ff421f7df29ef9116ab2c890ad428d312f020704cdfac7", 
        path=cache_dir
    )
    
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=cache_dir)
    
    pooch.retrieve(
        url=xlsx_url,
        known_hash="f4ff0162c5fe0a584d735f475260018bfdf52fdb9193b5303a8d24d02c1a6da9",
        path=extracted_folder,  # Put annotation data in the same folder. 
        fname="FiberClusterAnnotation_Updated20230110.xlsx"
    )
    
    # Delete tar
    Path(tar_path).unlink(missing_ok=True)
    
    print(f"\nDownload complete - TractCloud Dataset is now stored at: {extracted_folder}\n")
    return extracted_folder

