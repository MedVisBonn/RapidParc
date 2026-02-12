import torch
import torch.nn as nn
from utils.args_from_yaml import load_args_from_yaml
from pathlib import Path
from utils.model import TransformerModel, get_embedding_layer
from utils.transforms3D import normalize_to_identity_cube, Rotate3DBatch_UnifDist
from utils.model_snapshot_handling import get_latest_snapshot
from utils.dataset import get_TractCloud_eval_returner
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, classification_report
from utils.visualizations import create_plot
from typing import Optional, Union
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress
import time
import argparse
import os


@torch.inference_mode()
def test(
        experiment_path: Union[str, os.PathLike], 
        tractCloudDatasetPath: Union[str, os.PathLike],
        applyTestSetAugmentations: bool,
        eval_batch_size: int,
        evaluation_context_size: int = 2000,
        device: Optional[torch.device] = None, 
        augmentations: Optional[nn.Module] = None,
        print_classification_report: bool = True):
    """ Test the model on the test set with 30 augmentations """
    experiment_path = Path(experiment_path)
    if applyTestSetAugmentations:
        print(f"Testing experiment {experiment_path.stem} on the test set with 30 augmentations")
    else: 
        print(f"Testing experiment {experiment_path.stem} on the test set without augmentations")
    torch.set_float32_matmul_precision('high')
    if device is None:
        if torch.cuda.is_available(): device = torch.device("cuda")
        elif torch.backends.mps.is_available(): device = torch.device("mps")
        else: device = torch.device("cpu")
    args = load_args_from_yaml(yaml_file_path=experiment_path / "args/args.yml")
    map_800_800_to_43 = torch.from_numpy(np.load("utils/mapping_from_800_800_to_43.npy")).to(device)

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
    model_file = get_latest_snapshot(experiment_path/"model")
    if model_file is None:
        raise ValueError(f"Could not load the latest model. There is no model in {experiment_path/"model"}.")
    checkpoint = torch.load(model_file, weights_only=True, map_location=device)
    model = nn.DataParallel(model)
    model.load_state_dict(checkpoint['model'])
    model = model.module
    model.eval()

    # Load test data
    subject_streamlines, subject_labels, subject_labels_800_800 = \
        get_TractCloud_eval_returner(
            inputPath=tractCloudDatasetPath,
            device=torch.device("cpu"),
            eval_context_size=evaluation_context_size,
            n_vertices=n_vertices,
            split="test")()
    
    # Apply Augmentations
    if applyTestSetAugmentations:
        if augmentations is None:
            # Use TractCloud Augmentations (without translation and stretching because we normalize anyway)
            augmentations = nn.Sequential(
                Rotate3DBatch_UnifDist(range_x=[-45, 45], range_y=[-10, 10], range_z=[-10, 10])
            )# .to(device)
        # Copy data 30 times as TractCloud does
        streamlines = subject_streamlines.repeat_interleave(30, dim=0)
        labels_43 = subject_labels.repeat_interleave(30, dim=0) 
        labels_800_800 = subject_labels_800_800.repeat_interleave(30, dim=0)
        # Augment Streamlines
        streamlines = augmentations(streamlines)  
        # Add the original streamlines to the augmented ones (as TractCloud does)
        streamlines = torch.cat([subject_streamlines, streamlines], dim=0) # Added the original streamlines to the augmented ones
        labels_43 = torch.cat([subject_labels, labels_43], dim=0) # Shape [numSubjects * 31, numStreamlines]
        labels_800_800 = torch.cat([subject_labels_800_800, labels_800_800], dim=0)
        del subject_streamlines, subject_labels, subject_labels_800_800
    else:
        # Nothing is done here
        streamlines = subject_streamlines # Added the original streamlines to the augmented ones
        labels_43 = subject_labels # Shape [numSubjects * 31, numStreamlines]
        labels_800_800 = subject_labels_800_800
    
    # Shuffle Streamlines for each Subject
    streamlines, labels_43, labels_800_800 = torch.vmap(random_permutate_streamlines, randomness="different")(streamlines, labels_43, labels_800_800)
    _, numStreamlinesPerSubject, seqLength, spaceDim = streamlines.shape
    assert numStreamlinesPerSubject % evaluation_context_size == 0, \
        f"In this script, we assume that numStreamlinesPerSubject % evaluation_context_size == 0, "\
        +"but got numStreamlinesPerSubject={numStreamlinesPerSubject} and evaluation_context_size={evaluation_context_size}"
    streamlines = streamlines.cpu()
    labels_43 = labels_43.flatten().cpu().numpy()
    labels_800_800 = labels_800_800.flatten().cpu().numpy()
    
    # Start testing and timing
    duration = time.time()
    streamlines = streamlines.to(device, non_blocking=True)
    map_800_800_to_43 = map_800_800_to_43.to(device, non_blocking=True)
    # torch.cuda.synchronize(device)
    streamlines = streamlines.reshape(-1, evaluation_context_size, seqLength, spaceDim)
    streamlines = normalize_to_identity_cube(streamlines)

    print("Whole array shape:", streamlines.shape)
    y_pred_800_800 = []
    for i in range(0, len(streamlines), eval_batch_size):
        batch = streamlines[i : i + eval_batch_size]
        y_pred_800_800_batch = model(batch).argmax(dim=-1).flatten() # Shape [batch_size * context_size]
        y_pred_800_800.append(y_pred_800_800_batch)
    
    y_pred_800_800 = torch.cat(y_pred_800_800, dim=0) # Shape [numSubjects * 31 * context_size]
    y_pred = map_800_800_to_43[y_pred_800_800].cpu().numpy()
    y_pred_800_800 = y_pred_800_800.cpu().numpy()

    duration = time.time() - duration
    print(f"Duration: {duration:.2f} seconds")
    target_names = np.load(file = "utils/int_to_label.npy")
    plot_support_vs_metrics(y_true=labels_43, y_pred=y_pred, target_names=target_names, outdir=experiment_path / "plots" / "support_vs_metrics_30augs", prefix="aug")
    if print_classification_report:
        print(classification_report(y_true=labels_43, y_pred=y_pred, digits=5, target_names=target_names, zero_division=0))
        # print(classification_report(y_true=y_true_800_800, y_pred=y_pred_800_800, digits=5, target_names=["cluster_"+str(i).zfill(5) for i in range(800)] + ["outlayer_"+str(i).zfill(5) for i in range(800)]))
    
    if applyTestSetAugmentations:
        save_dir = Path(experiment_path / "plots" / "testset_STA")
    else:
        save_dir = Path(experiment_path / "plots" / "testset")
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    create_plot(y_true=labels_43, 
                y_pred=y_pred, 
                subject_number=None, 
                duration=duration, 
                save_path=save_dir,
                classnames=target_names, 
                unnormalized=False)
    return accuracy_score(y_true=labels_43, y_pred=y_pred), f1_score(y_true=labels_43, y_pred=y_pred, average="macro")



@torch.inference_mode()
def random_permutate_streamlines(streamlines, labels, labels_800_800):
    assert len(streamlines.shape) == 3, f"Streamlines have wrong shape: {streamlines.shape}, but assumed something like [numStreamlines, 15, 3]"
    indices = torch.randperm(streamlines.shape[0])
    return streamlines[indices], labels[indices], labels_800_800[indices]


def plot_support_vs_metrics(y_true: np.ndarray, 
                            y_pred: np.ndarray, 
                            target_names: np.ndarray, 
                            outdir: Union[str, os.PathLike], 
                            prefix: str) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    target_names = np.asarray(target_names).astype(str)
    classes = np.unique(y_true)
    records = []
    for cls in classes:
        mask = y_true == cls
        support = int(mask.sum())
        if support == 0:
            continue
        tract_name = target_names[cls] if cls < len(target_names) else f"class_{cls}"
        if tract_name.strip().lower() == "other":
            continue
        per_class_accuracy = float((y_pred[mask] == cls).mean())
        per_class_f1 = float(f1_score(mask.astype(int), (y_pred == cls).astype(int), zero_division=0))
        records.append({
            "tract": tract_name,
            "support": support,
            "accuracy": per_class_accuracy,
            "f1": per_class_f1,
        })

    if len(records) < 2:
        return

    df = pd.DataFrame.from_records(records)

    def _plot(metric: str, ylabel: str) -> None:
        plt.figure(figsize=(7, 5))
        ax = sns.regplot(data=df, x="support", y=metric, ci=95, scatter_kws={"s": 60, "alpha": 0.8}, line_kws={"color": "#d62728"})
        _, _, r_value, p_value, _ = linregress(df["support"], df[metric])
        ax.set_xlabel("Streamline support per tract")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs. Support ({prefix})")
        annotation = f"r={r_value:.3f}\np={p_value:.3g}"
        ax.annotate(annotation, xy=(0.02, 0.95), xycoords="axes fraction", va="top", ha="left", fontsize=10, bbox=dict(boxstyle="round", fc="white", alpha=0.7))
        plt.tight_layout()
        plt.savefig(outdir / f"{prefix}_{metric}_vs_support.png", dpi=300)
        plt.close()

    sns.set_theme(style="whitegrid")
    _plot("accuracy", "Per-tract accuracy")
    _plot("f1", "Per-tract F1 score")

    df_sorted = df.sort_values("support", ascending=False)
    bar_height = max(4, 0.25 * len(df_sorted))
    plt.figure(figsize=(8, bar_height))
    ax = sns.barplot(data=df_sorted, x="support", y="tract", color="#1f77b4")
    ax.set_xlabel("Streamline support")
    ax.set_ylabel("Tract")
    ax.set_title(f"Streamline support per tract ({prefix})")
    plt.tight_layout()
    plt.savefig(outdir / f"{prefix}_support_barplot.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test RapidParc")
    parser.add_argument("--experiment_path", type=str, required=True, help="Path to the experiment directory, which should contain the trained model and args.yml")
    parser.add_argument("--tractCloudDatasetPath", type=str, required=True, help="Path to the TractCloud dataset")
    parser.add_argument("--applyTestSetAugmentations", action="store_true", help="Whether to apply test set augmentations")
    parser.add_argument("--eval_batch_size", type=int, default=512, help="Evaluation batch size")
    parser.add_argument("--evaluation_context_size", type=int, default=2000, help="Evaluation context size")
    parser.add_argument("--device", type=str, default=None, help="Device to use (e.g., 'cuda', 'cpu')")
    parser.add_argument("--print_classification_report", action="store_true", help="Whether to print the classification report")
    args = parser.parse_args()
    
    test(experiment_path=args.experiment_path,
         tractCloudDatasetPath=args.tractCloudDatasetPath,
         applyTestSetAugmentations=args.applyTestSetAugmentations,
         eval_batch_size=args.eval_batch_size,
         evaluation_context_size=args.evaluation_context_size,
         device=torch.device(args.device) if args.device is not None else None,
         print_classification_report=args.print_classification_report
    )