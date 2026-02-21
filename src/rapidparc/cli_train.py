import argparse
import os
from .train import train
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(prog="rapidparc-train", description="Argument parser for streamline classification")
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
    return train(args=args)