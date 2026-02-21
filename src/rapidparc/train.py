import os 
import sys
import yaml
from pathlib import Path
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.elastic.multiprocessing.errors import record
from typing import Union

from .utils.dataset import get_TractCloud_train_returner, get_TractCloud_eval_returner, get_mapping_from_800_800_to_43, get_int_to_label
from .utils.model import TransformerModel, get_embedding_layer
from .utils.trainer import train as _train
from .utils.visualizations import create_confusion_matrix
from .utils.transforms3D import RandomNoiseBatch, Rotate3DBatch_UnifDist, RandomNoisedNormalizeToIdentetyCube
from .utils.ddp_handling import ddp_setup, ddp_cleanup, ddp_is_running
from .utils.args_from_yaml import load_args_from_yaml
from .test import test



def create__output_folders(output_folder_name: Union[str, os.PathLike], subfolders: list = ["model", "args", "slurm", "plots"]):
    output_folder_name = Path(output_folder_name)
    if not output_folder_name.is_dir(): 
        output_folder_name.mkdir()
    for subfolder in subfolders:
        if not (output_folder_name / subfolder).is_dir(): 
            (output_folder_name / subfolder).mkdir()


@record
def train(args) -> None: 
    # For definition of the args-Object, take a look at `cli_train.py`
    # If you want to call this function from python, you may define args as a dataclass instance.
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
    model = _train(
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