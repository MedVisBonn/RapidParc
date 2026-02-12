import os 
import sys
import argparse
import yaml
from pathlib import Path
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.elastic.multiprocessing.errors import record
import numpy as np
from datetime import datetime
from typing import Union

from utils.dataset import get_TractCloud_train_returner, get_TractCloud_eval_returner, get_mapping_from_800_800_to_43, get_int_to_label
from utils.model import TransformerModel, get_embedding_layer
from utils.trainer import train
from utils.visualizations import create_confusion_matrix
from utils.transforms3D import RandomNoiseBatch, Rotate3DBatch_UnifDist, RandomNoisedNormalizeToIdentetyCube
from utils.ddp_handling import ddp_setup, ddp_cleanup, ddp_is_running
from utils.args_from_yaml import load_args_from_yaml
from test import test


def parse_args():

    parser = argparse.ArgumentParser(description="Argument parser for streamline classification")
    job_id = os.environ.get("WANDB_RUN_ID", None)
    if job_id is None:
        job_id = os.environ.get("SLURM_JOB_ID", default=str(datetime.today().strftime('%Y-%m-%d_%H:%M:%S')))
    

    # learning_params
    parser.add_argument("--slurm_job_id", type=str, default=job_id, help="SLURM_JOB_ID important for restarts")
    parser.add_argument("--use_ddp", type=bool, default=False, help="If the elastic distributed backend should be used")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--context_size", type=int, default=2000, help="Context size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--global_batch_size", type=int, default=8, help="Resulting global batch size after DDP")
    parser.add_argument("--snapshot_every", type=int, default=1000, help="Save snapshot every n epochs")
    parser.add_argument("--store_all_snapshots", type=bool, default=True, help="Store all snapshots")

    # model Params
    parser.add_argument("--embedding_type", type=str, default="flip_aug", help="Can be 'flip_inv' or 'flip_aug'")
    parser.add_argument("--num_layers", type=int, default=1, help="Number of layers")
    parser.add_argument("--d_model", type=int, default=45, help="Model dimension, at least 66 for flip-inv or 45 for flip-aug")
    parser.add_argument("--nhead", type=int, default=1, help="Number of attention heads")
    parser.add_argument("--dim_feedforward", type=int, default=32, help="Feedforward dimension of the transformer")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--dim_class_hidden", type=int, default=256, help="Hidden dimension for classification")

    # Data Augmentation
    parser.add_argument("--aug_rotate_unif_x", type=list, default=[-45, 45], help="Rotation uniform distribution of x-axis in degrees")
    parser.add_argument("--aug_rotate_unif_y", type=list, default=[-10, 10], help="Rotation uniform distribution of y-axis in degrees")
    parser.add_argument("--aug_rotate_unif_z", type=list, default=[-10, 10], help="Rotation uniform distribution of z-axis in degrees")
    parser.add_argument("--aug_norm_shift_std", type=float, default=0.01, help="Standard deviation of the shift parameter of normalizing layer")
    parser.add_argument("--aug_norm_scale_std", type=float, default=0.01, help="Standard deviation of the scale parameter of normalizing layer")
    parser.add_argument("--aug_noise_sigma", type=float, default=0.001, help="Standard deviation of noise")
    parser.add_argument("--aug_p_hemi", type=float, default=0.3, help="Probability for sampling subtractogram from a single hemisphere")
    parser.add_argument("--aug_method_hemi", type=str, default='random', help="Method for augmenting with a subtractogram hemisphere")

    # Downsampling for limits on input data
    parser.add_argument("--n_vertices", type=int, default=15, help="Sample to this number of vertices per streamline.")

    # Pathing
    parser.add_argument("--out_dir", type=str, default="pretrained_models",help="Output directory for the experiment")
    parser.add_argument("--in_dir", type=str, default="TractCloud_Dataset/TrainData_800clu800ol", help="Input directory")


    args = parser.parse_args()

    assert args.n_vertices >= 2, "n_vertices must be at least 2"
    assert args.n_vertices <= 15, "n_vertices must be at most 15, this is the maximum resolution of the data provided in TractCloud dataset"

    # Constants:
    args.dim_out = 1600 # 43
    args.sched_args = {
        "scheduler": "CosineAnnealingLR",
        "eta_min": 0.0 # args.lr / 5000
    }
    return args


def create__output_folders(output_folder_name: Union[str, os.PathLike], subfolders: list = ["model", "args", "slurm", "plots"]):
    output_folder_name = Path(output_folder_name)
    if not output_folder_name.is_dir(): 
        output_folder_name.mkdir()
    for subfolder in subfolders:
        if not (output_folder_name / subfolder).is_dir(): 
            (output_folder_name / subfolder).mkdir()


@record
def main(args) -> None:
    
    torch.set_float32_matmul_precision('high')
    if args.use_ddp:   
        ddp_setup()
        args.local_batch_size = args.global_batch_size // dist.get_world_size() 
        device = torch.device(type="cuda", index=int(os.environ['LOCAL_RANK']))
    else: 
        args.local_batch_size = args.global_batch_size
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    os.chdir(str(Path(__file__).parent.resolve()))

    # Create output folders and wandb init
    output_folder = Path(args.out_dir)
    if not args.use_ddp or dist.get_rank() == 0:
        if not output_folder.is_dir():
            output_folder.mkdir()
        create__output_folders(output_folder_name = output_folder / args.slurm_job_id)
    output_folder = output_folder / args.slurm_job_id

    # Load Dataloaders
    data_path = Path(args.in_dir).absolute() 
    get_train_Tensor = get_TractCloud_train_returner(
        inputPath=data_path,
        device=device,
        n_vertices=args.n_vertices)
    get_val_Tensor = get_TractCloud_eval_returner(
        inputPath=data_path,
        device=device,
        eval_context_size=args.context_size,
        n_vertices=args.n_vertices,
        split="val")
    
    map_800_800_to_43 = torch.from_numpy(get_mapping_from_800_800_to_43(data_path))
    
    # Augmentions (on GPU)
    trainAugmentations = nn.Sequential(
        Rotate3DBatch_UnifDist(range_x=args.aug_rotate_unif_x, range_y=args.aug_rotate_unif_y, range_z=args.aug_rotate_unif_z),
        RandomNoiseBatch(mu=0, sigma=args.aug_noise_sigma),
        RandomNoisedNormalizeToIdentetyCube(shift_std=args.aug_norm_shift_std, scale_std=args.aug_norm_scale_std)
    ).to(device)
    
    embedding= get_embedding_layer(
        num_support_points_per_streamline=args.n_vertices,
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
    model.to(device)

    if not args.use_ddp or dist.get_rank() == 0:
        print(f"Number Devices available: {torch.cuda.device_count()}")
        print("############### MODEL ###############")
        print(model)
        print(f"Number trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)} \n")
    if args.use_ddp:
        print(f"Rank {dist.get_rank()} is online and handles GPU {device}")
        model = DDP(model, device_ids=[device], find_unused_parameters=True)
    else:
        model = nn.DataParallel(model)


    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.001)

    if args.sched_args["scheduler"] == "CosineAnnealingWarmRestarts":
        # If this scheduler is used, I recommend to add gradient clipping
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.sched_args["T_0"], T_mult=args.sched_args["T_mult"], eta_min=args.sched_args["eta_min"])
    elif args.sched_args["scheduler"] == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=args.sched_args["eta_min"])
    else:
        raise NotImplementedError(f"Scheduler {args.sched_args['scheduler']} not implemented")
    
    # Store args
    if not ddp_is_running() or dist.get_rank() == 0:
        with open(str(Path(output_folder) / 'args' / 'args.yml'), 'w') as outfile:
            yaml.dump(vars(args), outfile, default_flow_style=False, sort_keys=False)

    # Training
    model = train(
        get_train_Tensor=get_train_Tensor,
        get_val_Tensor=get_val_Tensor,
        num_epochs=args.num_epochs,
        batch_size=args.local_batch_size,
        context_size=args.context_size,
        optimizer=optimizer,
        scheduler=scheduler,
        model=model,
        device=device,
        criterion=criterion,
        snapshot_path=str(Path(output_folder) / "model"),
        snapshot_every=args.snapshot_every,
        store_all_snapshots=args.store_all_snapshots,
        transforms_on_gpu=trainAugmentations,
        aug_p_hemi = args.aug_p_hemi,
        aug_method_hemi = args.aug_method_hemi,
        map_800_800_to_43=map_800_800_to_43
        )
    
    # Confusion Matrix and dumping args
    if not ddp_is_running() or dist.get_rank() == 0: 
        sys.stdout.flush()
        print("\n\n##### Confusion Matricies #####")

        for split in ["train", "val", "test"]:
            print(f"\nCreating confusion matricies on {split} set")
            create_confusion_matrix(
                model=model,
                split=split,
                input_path=data_path,
                save_path=str(Path(output_folder) / "plots" / split),
                device=device,
                map_800_800_to_43=map_800_800_to_43,
                int_to_label=get_int_to_label(data_path),
                n_vertices = args.n_vertices,
                eval_batch_size=64
            )

        
        accuracy_STA, macro_f1_score_STA = test(
            experiment_path = output_folder, 
            tractCloudDatasetPath=data_path,
            applyTestSetAugmentations=True,
            eval_batch_size=8, 
            evaluation_context_size=2000, 
            device=device,
            print_classification_report=False)
        
        accuracy, macro_f1_score = test(
            experiment_path = output_folder, 
            tractCloudDatasetPath=data_path,
            applyTestSetAugmentations=False,
            eval_batch_size=8, 
            evaluation_context_size=2000, 
            device=device,
            print_classification_report=False)

        print("Final results on TractCloud test set\n" 
              +f"- Acc: {100*accuracy:.2f}%, macro F1: {100*macro_f1_score:.2f}% \n"\
              +f"- STA Acc: {100*accuracy_STA:.2f}%, STA Macro F1: {100*macro_f1_score_STA:.2f}%")
    if ddp_is_running():
        print(f"Max memory used by device {device}: {torch.cuda.max_memory_allocated(device=device) / 1024**3}")
        sys.stdout.flush()
        dist.barrier()
        ddp_cleanup()



if __name__ == '__main__':
    main(args = parse_args())