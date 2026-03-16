from typing import Union
from pathlib import Path
import tarfile
import pooch
import os
import numpy as np
import torch
from safetensors.torch import load_file

from .model_snapshot_handling import get_latest_snapshot
from .args_from_yaml import load_args_from_yaml
from .model import TransformerModel, get_embedding_layer

CACHE_DIR = Path(pooch.os_cache("RapidParc"))
MODEL_DIR = CACHE_DIR / "Models"
TRACTCLOUD_FOLDER = CACHE_DIR / "TrainData_800clu800ol"
PRETRAINED_MODELS = ["rapidparc", "rapidparc_v8", "hemiaug"]
BASE_DOWNLOAD_URL = "https://github.com/MedVisBonn/RapidParc/releases/download/v1.0.0"
HASHES = {
    "hemiaug.safetensors": "sha256:b66dfce6135aac56c3f835264e5ae9630b2497859425ad98cdc6d842c6f94435",
    "hemiaug_args.yaml": "sha256:4afe6fc28ee16f26ce98e529503a8cadaf792b5951a4293b1b89ad20c63c890c",
    "int_to_label.npy": "sha256:bbb0d6e8d2eb208677eded80ccc93f5d00bd1120e81e13fb5d5c6e20ba2650c0",
    "mapping_from_800_800_to_43.pt": "sha256:8836303db236fe885fbdcd827559cce69f4673b772fb6ae5d6caf78544fe9522",
    "rapidparc.safetensors": "sha256:6a158dad6a0b6124946da102b6037a6f7f8a16fb24e7caec15eaf95213e7b793",
    "rapidparc_args.yaml": "sha256:5c2b13a9843dc1fb73a066ccbfd0ce3455112048786f7e7ff594e6a4974bac4d",
    "rapidparc_v8.safetensors": "sha256:16a0e678b19a825177ae6bedcb068999befd7ba06163766f1b459b52a58b39e9",
    "rapidparc_v8_args.yaml": "sha256:1c54411e9eab69b0854f309ca47f6dfe05f125235df584664d28d2007cdf28dd"
}


def load_model_and_args_local_or_online(name_or_path: Union[str, os.PathLike], device: torch.device):
    if not isinstance(name_or_path, str) and not isinstance(name_or_path, os.PathLike):
        raise ValueError('Expected varaible name_or_path to be either one of "rapidparc", "rapidparc_v8" and "hemiaug"'\
                         +f'or to be an experiment path (containing model and args.yaml), but got {name_or_path}')
    if isinstance(name_or_path, str) and name_or_path in PRETRAINED_MODELS:
        ########## download args and model from Github ##########
        args_filename = f"{name_or_path}_args.yaml"
        pooch.retrieve(
            url=f"{BASE_DOWNLOAD_URL}/{args_filename}",
            known_hash=HASHES[args_filename],
            path=MODEL_DIR,
            fname=args_filename
        )
        
        args = load_args_from_yaml(MODEL_DIR / args_filename)

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
        model_filename = f"{name_or_path}.safetensors"
        pooch.retrieve(
            url=f"{BASE_DOWNLOAD_URL}/{model_filename}",
            known_hash=HASHES[model_filename],
            path=MODEL_DIR,
            fname=model_filename
        )

        state_dict = load_file(MODEL_DIR / model_filename) 
        model = model.to(device)  # deine Modellklasse
        model.load_state_dict(state_dict)
        model.eval()
        return model, args
    
    # Load Model from local folder:
    experiment_path = Path(name_or_path)
    assert experiment_path.is_dir(), f"The directiory {experiment_path} does not exist but should."
    
    if (experiment_path / "args" / "args.yaml").is_file():
        args = load_args_from_yaml(yaml_file_path=experiment_path / "args" / "args.yaml")
    elif (experiment_path / "args" / "args.yml").is_file():
        args = load_args_from_yaml(yaml_file_path=experiment_path / "args" / "args.yml")
    else:
        raise ValueError(f"File {experiment_path / "args" / "args.yaml"} does not exist but should.")
    
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
    
    # Model
    model = TransformerModel(num_layers=args.num_layers, 
                             d_model=args.d_model,
                             nhead=args.nhead,
                             embedding_layer=embedding,
                             dim_feedforward=args.dim_feedforward,
                             dropout=args.dropout,
                             dim_class_hidden=args.dim_class_hidden,
                             dim_out=args.dim_out)
    model = model.to(device)
    assert (experiment_path/"model").is_dir(), f"The folder {experiment_path/"model"} does not exist but should"
    model_file = get_latest_snapshot(experiment_path/"model")
    if model_file is None:
        raise ValueError(f"Could not load the latest model. There is no model in {experiment_path/"model"}.")
    checkpoint = torch.load(model_file, weights_only=True, map_location=device)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['model'])
    model = model.module
    model.eval()

    return model, args




def get_mapping_800_800_to_43_online() -> torch.Tensor:
    filename = "mapping_from_800_800_to_43.pt"
    pooch.retrieve(
        url=f"{BASE_DOWNLOAD_URL}/{filename}",
        known_hash=HASHES[filename],
        path=CACHE_DIR,
        fname=filename
    )
    return torch.load(CACHE_DIR/filename)


def get_int_to_label_online() -> np.ndarray:
    filename = "int_to_label.npy"
    pooch.retrieve(
        url=f"{BASE_DOWNLOAD_URL}/{filename}",
        known_hash=HASHES[filename],
        path=CACHE_DIR,
        fname=filename
    )
    return np.load(CACHE_DIR/filename)



def get_tractCloud_dataset_path(consent_given: bool = False) -> Path:
    # If the Dataset is downloaded, user already accepted licence agreement 
    if TRACTCLOUD_FOLDER.exists() and TRACTCLOUD_FOLDER.is_dir():
        return TRACTCLOUD_FOLDER

    print("\nFor running this code, you have to download the TractCloud dataset. \nPlease check out the Slicer Licence agreement before downloading the dataset:\n")
    print("\thttps://github.com/SlicerDMRI/TractCloud/blob/main/LICENSE\n")
    
    while not consent_given:
        answer = input("Do you accept the license agreement? [y/n]: ").strip().lower()
        if answer in ("y", "yes"):
            consent_given = True
        elif answer in ("n", "no"):
            raise PermissionError("Download aborted: License agreement not accepted.")
        else:
            print("Invalid input. Please answer with 'y' or 'n'.")

    # Run download:
    tar_url = "https://github.com/SlicerDMRI/TractCloud/releases/download/v1.0.0/TrainData_800clu800ol.tar.gz"
    xlsx_url = "https://github.com/SlicerDMRI/TractCloud/raw/2c2869cde12b8d617d21be8e19fd33ae73b5f3bc/datasets/FiberClusterAnnotation_Updated20230110.xlsx"

    tar_path = pooch.retrieve(
        url=tar_url, 
        known_hash="72d70d4cf7a465a824ff421f7df29ef9116ab2c890ad428d312f020704cdfac7", 
        path=CACHE_DIR
    )
    
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=CACHE_DIR)
    
    pooch.retrieve(
        url=xlsx_url,
        known_hash="f4ff0162c5fe0a584d735f475260018bfdf52fdb9193b5303a8d24d02c1a6da9",
        path=TRACTCLOUD_FOLDER,  # Put annotation data in the same folder. 
        fname="FiberClusterAnnotation_Updated20230110.xlsx"
    )
    
    # Delete tar
    Path(tar_path).unlink(missing_ok=True)
    
    print(f"\nDownload complete - TractCloud Dataset is now stored at: {TRACTCLOUD_FOLDER}\n")
    return TRACTCLOUD_FOLDER

