import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score, classification_report
from typing import Optional, Union, Tuple
import numpy as np
import time
import os


from .utils.transforms3D import normalize_to_identity_cube, Rotate3DBatch_UnifDist
from .utils.dataset import get_TractCloud_eval_returner
from .utils.visualizations import create_plot
from .utils.pypi_package_helper import load_model_and_args_local_or_online, get_tractCloud_dataset_path, get_mapping_800_800_to_43_online, get_int_to_label_online


@torch.inference_mode()
def test(
        model_name_or_path: Union[str, os.PathLike], 
        applyTestSetAugmentations: bool,
        eval_batch_size: int = 128,
        evaluation_context_size: int = 2000,
        device: Optional[torch.device] = None, 
        augmentations: Optional[nn.Module] = None,
        print_classification_report: bool = True,
        tractcloud_licence_consent_given: bool = False
        ) -> Tuple[float, Union[float, np.ndarray]]:
    """ Benchmark function to test RapidParc on the tractcloud dataset test split. Creates Confusion matrices.

        Parameters:
            model_name_or_pathmodel_folder_path (PathLike | str):
                One of the following:
                A pretrained model, i.e. one of 'rapidparc', 'rapidparc_v8' or 'hemiaug'
                Or a path of an experiment run (i.e. a folder of the 'pretrained_models' folder after training)

            applyTestSetAugmentations (bool):
                Whether the 30 instances of the testset should be randomly rotated and appended to the test set. 

            eval_batch_size (int):
                Batch size during forward pass. One batch has the size of eval_batch_size * evaluation_context_size * n_verticies * 3
                Depending on your system, this value should be choosen to be small to prevent cuda-out-of-memory issues.

            evaluation_context_size (int):
                Context size during evaluation. Assumed to be a integral fraction of 10000. 
                Default is 2000. Larger values lead to slightly better results, but at the expense of higher computational effort.

            device (torch.device):
                Torch device used for evaluation. 'None' means: automatically chosen to be the best one available (if only one GPU is available).

            augmentations (torch.nn.Module):
                Optional augmentations used instead of the pre-defined random rotations. Only active if applyTestSetAugmentations is True.

            print_classification_report (bool):
                Whether the scipy classification report should be printed to the terminal.

            tractcloud_licence_consent_given (bool):
                Whether you agree to the slicer licence agreement (Only asked for once on each machine).
                This is the licence for using the TractCloud Dataset. For further information, take a look at
                https://github.com/SlicerDMRI/TractCloud/blob/main/LICENSE   

                
        Returns:
            Tuple of Accuracy and Macro F1 Score for y_true vs. y_pred.

    """
    
    if applyTestSetAugmentations:
        print(f"Testing experiment {model_name_or_path} on the test set with 30 augmentations")
    else: 
        print(f"Testing experiment {model_name_or_path} on the test set without augmentations")

    torch.set_float32_matmul_precision('high')
    if device is None:
        if torch.cuda.is_available(): device = torch.device("cuda")
        elif torch.backends.mps.is_available(): device = torch.device("mps")
        else: device = torch.device("cpu")
    map_800_800_to_43 = get_mapping_800_800_to_43_online().to(device)

    model, args = load_model_and_args_local_or_online(name_or_path=model_name_or_path, device=device)
    n_vertices = getattr(args, "n_vertices", None)
    
    if n_vertices is None:
        n_vertices = getattr(args, "num_support_points_per_streamline", None)
    if n_vertices is None:
        raise AttributeError(
            "Configuration is missing 'n_vertices'. Please add it to args.yml or provide "
            "'num_support_points_per_streamline'."
        )

    # Load test data
    subject_streamlines, subject_labels, _ = \
        get_TractCloud_eval_returner(
            inputPath=get_tractCloud_dataset_path(consent_given=tractcloud_licence_consent_given),
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
        streamlines = augmentations(streamlines)  
        # Add the original streamlines to the augmented ones (as TractCloud does)
        streamlines = torch.cat([subject_streamlines, streamlines], dim=0) # Added the original streamlines to the augmented ones
        labels_43 = torch.cat([subject_labels, labels_43], dim=0) # Shape [numSubjects * 31, numStreamlines]
    else:
        # Nothing is done here
        streamlines = subject_streamlines 
        labels_43 = subject_labels 
    
    # Shuffle Streamlines for each Subject
    streamlines, labels_43 = torch.vmap(random_permutate_streamlines, randomness="different")(streamlines, labels_43)
    _, numStreamlinesPerSubject, seqLength, spaceDim = streamlines.shape
    assert numStreamlinesPerSubject % evaluation_context_size == 0, \
        f"In this script, we assume that numStreamlinesPerSubject % evaluation_context_size == 0, "\
        +"but got numStreamlinesPerSubject={numStreamlinesPerSubject} and evaluation_context_size={evaluation_context_size}"
    streamlines = streamlines.cpu()
    labels_43 = labels_43.flatten().cpu().numpy()
    
    # Start testing and timing
    duration = time.time()
    streamlines = streamlines.to(device)
    map_800_800_to_43 = map_800_800_to_43.to(device)
    streamlines = streamlines.reshape(-1, evaluation_context_size, seqLength, spaceDim)
    streamlines = normalize_to_identity_cube(streamlines)

    y_pred_800_800 = []
    for i in range(0, len(streamlines), eval_batch_size):
        batch = streamlines[i : i + eval_batch_size]
        y_pred_800_800_batch = model(batch).argmax(dim=-1).flatten() # Shape [batch_size * context_size]
        y_pred_800_800.append(y_pred_800_800_batch)
    
    y_pred_800_800 = torch.cat(y_pred_800_800, dim=0) # Shape [numSubjects * 31 * context_size]
    y_pred = map_800_800_to_43[y_pred_800_800].cpu().numpy()

    duration = time.time() - duration
    print(f"Duration: {duration:.2f} seconds")
    target_names = get_int_to_label_online()

    experiment_path = Path(model_name_or_path)
    if not experiment_path.is_dir():
        experiment_path.mkdir()

    if print_classification_report:
        print(classification_report(y_true=labels_43, y_pred=y_pred, digits=5, target_names=target_names, zero_division=0))
    
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
def random_permutate_streamlines(streamlines, labels):
    assert len(streamlines.shape) == 3, f"Streamlines have wrong shape: {streamlines.shape}, but assumed something like [numStreamlines, 15, 3]"
    indices = torch.randperm(streamlines.shape[0])
    return streamlines[indices], labels[indices]
