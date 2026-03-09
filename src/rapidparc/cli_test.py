import argparse
from .test import test
import torch

def main():
    parser = argparse.ArgumentParser(prog="rapidparc-test", description="Program to benchmark RapidParc on the test split of the TractCloud dataset")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Name of the Model, i.e. `rapidparc`, `rapidparc_v8` or `hemiaug`. Alternatively path to the experiment directory, which should contain the trained model and args.yml")
    parser.add_argument("--tractCloudDatasetPath", type=str, required=True, help="Path to the TractCloud dataset")
    parser.add_argument("--applyTestSetAugmentations", action="store_true", help="Whether to apply test set augmentations")
    parser.add_argument("--eval_batch_size", type=int, default=512, help="Evaluation batch size")
    parser.add_argument("--evaluation_context_size", type=int, default=2000, help="Evaluation context size")
    parser.add_argument("--device", type=str, default=None, help="Device to use (e.g., 'cuda', 'cpu')")
    parser.add_argument("--print_classification_report", action="store_true", help="Whether to print the classification report")
    parser.add_argument("--tractcloud_licence_consent_given", action="store_true", help="Whether you do agree to the slicer licence.")
    args = parser.parse_args()
    
    test(model_name_or_path=args.model_name_or_path,
         applyTestSetAugmentations=args.applyTestSetAugmentations,
         eval_batch_size=args.eval_batch_size,
         evaluation_context_size=args.evaluation_context_size,
         device=torch.device(args.device) if args.device is not None else None,
         print_classification_report=args.print_classification_report,
         tractcloud_licence_consent_given=args.tractcloud_licence_consent_given
    )