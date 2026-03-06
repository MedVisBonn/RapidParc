import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score, classification_report
from typing import Optional, Union
import time
import os


from .utils.transforms3D import normalize_to_identity_cube, Rotate3DBatch_UnifDist
from .utils.dataset import get_TractCloud_eval_returner
from .utils.visualizations import create_plot
from .utils.pypi_package_helper import load_model_and_args_local_or_hf, get_tractCloud_dataset_path, get_mapping_800_800_to_43_hf, get_int_to_label_hf


@torch.inference_mode()
def test(
        model_name_or_path: Union[str, os.PathLike], 
        applyTestSetAugmentations: bool,
        eval_batch_size: int = 128,
        evaluation_context_size: int = 2000,
        device: Optional[torch.device] = None, 
        augmentations: Optional[nn.Module] = None,
        print_classification_report: bool = True,
        slicer_licence_consent_given: bool = False
        ):
    """ Test the model on the test set with 30 augmentations """
    
    if applyTestSetAugmentations:
        print(f"Testing experiment {model_name_or_path} on the test set with 30 augmentations")
    else: 
        print(f"Testing experiment {model_name_or_path} on the test set without augmentations")

    torch.set_float32_matmul_precision('high')
    if device is None:
        if torch.cuda.is_available(): device = torch.device("cuda")
        elif torch.backends.mps.is_available(): device = torch.device("mps")
        else: device = torch.device("cpu")
    map_800_800_to_43 = get_mapping_800_800_to_43_hf().to(device)

    model, args = load_model_and_args_local_or_hf(name_or_path=model_name_or_path, device=device)
    n_vertices = getattr(args, "n_vertices", None)
    
    if n_vertices is None:
        n_vertices = getattr(args, "num_support_points_per_streamline", None)
    if n_vertices is None:
        raise AttributeError(
            "Configuration is missing 'n_vertices'. Please add it to args.yml or provide "
            "'num_support_points_per_streamline'."
        )

    # Load test data
    subject_streamlines, subject_labels, subject_labels_800_800 = \
        get_TractCloud_eval_returner(
            inputPath=get_tractCloud_dataset_path(consent_given=slicer_licence_consent_given),
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
        streamlines = subject_streamlines 
        labels_43 = subject_labels 
        labels_800_800 = subject_labels_800_800
    
    # Shuffle Streamlines for each Subject
    streamlines, labels_43, labels_800_800 = torch.vmap(random_permutate_streamlines, randomness="different")(streamlines, labels_43, labels_800_800)
    _, numStreamlinesPerSubject, seqLength, spaceDim = streamlines.shape
    assert numStreamlinesPerSubject % evaluation_context_size == 0, \
        f"In this script, we assume that numStreamlinesPerSubject % evaluation_context_size == 0, "\
        +"but got numStreamlinesPerSubject={numStreamlinesPerSubject} and evaluation_context_size={evaluation_context_size}"
    streamlines = streamlines.cpu()
    labels_43 = labels_43.flatten().cpu().numpy()
    # labels_800_800 = labels_800_800.flatten().cpu().numpy()
    
    # Start testing and timing
    duration = time.time()
    streamlines = streamlines.to(device)
    map_800_800_to_43 = map_800_800_to_43.to(device)
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
    # y_pred_800_800 = y_pred_800_800.cpu().numpy()

    duration = time.time() - duration
    print(f"Duration: {duration:.2f} seconds")
    target_names = get_int_to_label_hf()

    experiment_path = Path(model_name_or_path)
    if not experiment_path.is_dir():
        experiment_path.mkdir()

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
