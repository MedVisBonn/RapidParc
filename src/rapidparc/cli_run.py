import argparse
from .run import rapidParcTckEval
import torch

def main():
    parser = argparse.ArgumentParser(prog="rapidparc-run", description="Apply RapidParc on a .tck tractogram")
    parser.add_argument("--tck_path", type=str, required=True, help="Path to the input tck file.")
    parser.add_argument("--out_path", type=str, default="output_tcks", help="Path to the output folder, where the resulting tck files will be stored. If the folder does not exist, it will be created.")
    parser.add_argument("--model_folder_path", type=str, required=True, help="Path to the folder containing the trained model and the args.yml file.")
    parser.add_argument("--eval_batch_size", type=int, default=512, help="Batch size for evaluation.")
    parser.add_argument("--eval_context_size", type=int, default=2000, help="Context size for evaluation.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for evaluation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--print_time", action="store_true", help="Whether to print time measurements.")
    parser.add_argument("--print_class_distributiion", action="store_true", help="Whether the predicted class distribution should be printed")
    args = parser.parse_args()

    rapidParcTckEval(tck_path=args.tck_path,
                        out_path=args.out_path,
                        model_folder_path=args.model_folder_path,
                        eval_batch_size=args.eval_batch_size,
                        eval_context_size=args.eval_context_size,
                        seed=args.seed,
                        device=torch.device(args.device),
                        print_time=args.print_time,
                        print_class_distributiion=args.print_class_distributiion)
