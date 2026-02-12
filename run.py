import sys
import os
from pathlib import Path
import nibabel as nib
from typing import Union, List, Tuple
import numpy as np
import torch
import torch.nn as nn
torch.set_float32_matmul_precision('high')
import time
from tqdm import tqdm
import argparse

from utils.model import TransformerModel, get_embedding_layer
from utils.model_snapshot_handling import get_latest_snapshot
from utils.transforms3D import normalize_to_identity_cube
from utils.args_from_yaml import load_args_from_yaml
from utils import tck_io



def rapidParcTckEval(tck_path: Union[str, os.PathLike],
                     out_path: Union[str, os.PathLike],
                     model_folder_path: Union[str, os.PathLike],
                     eval_batch_size: int = 512,
                     eval_context_size: int = 2000,
                     device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                     seed: int = 42,
                     print_time: bool = True):
    
    data_tensor, origStreamlines = tck_to_tensor(tck_path=tck_path)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    y_pred_43 = rapidParc(model_folder_path = model_folder_path,
                          streamlines = data_tensor,
                          eval_batch_size=eval_batch_size,
                          eval_context_size=eval_context_size,
                          return_anatomical_clusters=True,
                          return_label_names=False,
                          device=device,
                          seed=seed,
                          print_time=print_time)
    
    int_to_label = np.load("utils/int_to_label.npy")

    results_to_tck(origStreamlines=origStreamlines, 
                   y_pred_43=y_pred_43, 
                   int_to_label=int_to_label,
                   output_path=out_path)
    



def rapidParc(model_folder_path: Union[str, os.PathLike],
              streamlines: Union[list, torch.Tensor, np.ndarray],
              eval_batch_size: int = 512,
              eval_context_size: int = 2000,
              return_anatomical_clusters: bool = True,
              return_label_names: bool = False,
              device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
              seed: int = 42,
              print_time: bool = True) -> Union[torch.Tensor, np.ndarray]:
    """
    Main evaluation function.

    Parameters:
        model_folder_path (PathLike | str):
            Folder of the training run. Should contain a "model"-subfolder

        streamlines (list | Tensor | ndarray):
            The streamlines of a single tractogram. Assumed to be one of the following: 
            NumPy Array of length numStreamlines, where each entry has shape [n_i, 3]
            NumPy Array of shape [numStreamlines, lenStreamline, 3] where lenStreamline >= 15
            Torch Tensor of shape [numStreamlines, lenStreamline, 3] where lenStreamline >= 15
            List of length numStreamlines, where each entry has shape [n_i, 3]
        
        eval_batch_size (int):
            The batch size during evaluation. Scales GPU memory and computational complexity linearly.
            The total number of Streamlines inside one batch = eval_batch_size * eval_context_size

        eval_context_size (int):
            The context size during evaluation. Scales GPU memory and computational complexity quadraticly.
            The total number of Streamlines inside one batch = eval_batch_size * eval_context_size

        return_anatomical_clusters (bool):
            Whether the 43 anatomical clusters should be returned (default). If not, the 800 + 800 clusters
            of the ORG atlas are used.

        return_label_names (bool):
            Whether the cluster names (i.e. "AF", ...) should be returned instead of just the cluster indices.
            This option is only possible if return_anatomical_clusters == True

        device (torch.device):
            Torch device used for evaluation

        seed (int):
            Random seed for reproducible results

        print_time (bool):
            Whether the time measurements should be printed.

    Returns:
        y_pred (Tensor | ndarray):
            Class predictions in the domain of the 42+1 anatomical regions, 800+800 (ORG atlas) or anatomical cluster names.
    """
    
    time_start = time.time()
    _, model = load_args_and_model(run_path=Path(model_folder_path), device=device)
    if print_time: print(f"Time to load model: {time.time() - time_start:.2f}s")
    
    # Load Mappings
    mapping_800_800_to_43 = torch.from_numpy(np.load("utils/mapping_from_800_800_to_43.npy")).to(device)
    int_to_label = np.load("utils/int_to_label.npy")

    # Resample Streamlines to 15 supporting points if necessary and change datatype to torch.Tensor
    if isinstance(streamlines, np.ndarray):
        if streamlines.dtype == object or streamlines.shape[-1] != 15:
            data_tensor, origStreamlines = resample_number_supporting_points_to_15(input_streamlines=streamlines, verbose=print_time)
        else:
            origStreamlines = streamlines
            data_tensor = torch.from_numpy(streamlines).pin_memory()
    elif isinstance(streamlines, torch.Tensor):
        if streamlines.size(-2) == 15:
            origStreamlines = streamlines
            data_tensor = streamlines.pin_memory()
        else:
            data_tensor, origStreamlines = resample_number_supporting_points_to_15(input_streamlines=streamlines, verbose=print_time)
    elif isinstance(streamlines, list) or isinstance(streamlines, nib.streamlines.array_sequence.ArraySequence):
        data_tensor, origStreamlines = resample_number_supporting_points_to_15(input_streamlines=streamlines, verbose=print_time)
    else:
        raise ValueError(f"Streamlines have to be a list, numpy.ndarray or torch.Tensor, but got {type(streamlines)}.")
   
    assert data_tensor.shape[1] == 15 and data_tensor.shape[2] == 3 and len(data_tensor.shape) == 3, "Assumed the input tensor to have a format of [numStreamlines, 15, 3]"

    # Normalize Streamlines to the [-1, 1] cube
    data_tensor = normalize_to_identity_cube(data_tensor.unsqueeze(0)).squeeze(0)

    # Shuffle data + Split in context size
    rng = torch.Generator(device="cpu").manual_seed(seed)
    time_to_shuffle_data = time.time()
    data_tensor, origStreamlines, permutation_indeces = shuffle_and_split_in_context_size(data_tensor=data_tensor, 
                                                                    origStreamlines=origStreamlines, 
                                                                    context_size=eval_context_size,
                                                                    rng=rng)
    if print_time: 
        print(f"Time to shuffle data: {time.time() - time_to_shuffle_data:.2f}s")
        print(f"{len(origStreamlines)} Streamlines are loaded and preprocessed.")

    # Forward Pass
    start_eval_time = time.time()
    y_pred = model_forward_pass(model=model, 
                                   eval_batch_size=eval_batch_size,
                                   data_tensor=data_tensor,
                                   mapping_800_800_to_43=mapping_800_800_to_43,
                                   device=device, 
                                   return_anatomical_clusters=return_anatomical_clusters)
    
    duration_eval_time = time.time() - start_eval_time
    if print_time: print(f"Prediction is done in {duration_eval_time:.2f}s")

    # Shorten Result-tensor to the original length
    y_pred = y_pred[:len(origStreamlines)]

    # Undo the permutation
    y_pred = y_pred[torch.argsort(permutation_indeces)]

    if print_time: print(f"Total time: {time.time() - time_start:.2f}s")
    
    if return_anatomical_clusters and return_label_names:
        return int_to_label[y_pred.numpy()]
    return y_pred


@torch.inference_mode()
def resample_number_supporting_points_to_15(input_streamlines: Union[list, np.ndarray, torch.Tensor], verbose: bool = False) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Function that resamples the number of supporting points to 15.

    Args:
        input_streamlines (list): List of streamlines. Each streamline is a numpy array of shape [N_i, 3].
        verbose (bool): If True, print the time it took to shorten the streamlines.

    Returns:
        resampled_streamlines (torch.Tensor): The resampled streamlines. Shape [N, 15, 3].
        origStreamlines (np.ndarray): The original streamlines as a numpy array object (for slicing operations).
    """
    time_to_shorten_streamlines = time.time()

    origStreamlines = []
    resampled_streamlines = []
    # Select all Streamlines with length ≥ 15
    for i, streamline in enumerate(input_streamlines):
        if len(streamline) < 15:
            print(f"""Streamline at index {i} has a length {len(streamline)}, which is less than 15.
                  As a result, the classification result might be faulty. 
                  Make sure to use streamlines with at least 15 supporting points and that the shape
                  of each streamline is [N_i, 3].""")
        assert len(streamline[0]) == 3, f"Streamline at index {i} has a shape {streamline[0].shape}, but assumed [N, 3]."
        origStreamlines.append(streamline)
        indices = np.round(np.linspace(start=0, stop=len(streamline) - 1, num=15)).astype(int)
        selected_points = streamline[indices]
        resampled_streamlines.append(selected_points)
    resampled_streamlines = torch.from_numpy(np.array(resampled_streamlines)).pin_memory() # Shape [N, 15, 3]
    if verbose: print(f"Time to shorten streamlines: {time.time() - time_to_shorten_streamlines:.2f}s")
    return resampled_streamlines, np.array(origStreamlines, dtype=object)


@torch.inference_mode()
def shuffle_and_split_in_context_size(data_tensor: torch.Tensor, 
                                      origStreamlines: np.ndarray, 
                                      context_size: int,
                                      rng: torch.Generator):
    """
    Function that splits the streamlines into context_size parts. If the number of streamlines is not divisible by context_size,
    the function adds streamlines from the beginning of the tensor to the end.

    Args:
        data_tensor: The shortened streamlines.
        origStreamlines: The original streamlines.
        context_size: The size of the context.

    Returns:
        context_streamlines: The streamlines split into context_size parts. The length might differ from the original length.
        context_streamlines_original: The original streamlines split into context_size parts.
    """
    assert len(data_tensor) == len(origStreamlines), "The number of streamlines and the number of data tensor do not match."
    n_streamlines, nPoints, spaceDim = data_tensor.size()
    random_indices = torch.randperm(len(data_tensor), generator=rng)
    data_tensor = data_tensor[random_indices]
    if n_streamlines % context_size > 0:
        streamlines_to_add = context_size - n_streamlines % context_size
        data_tensor = torch.cat([data_tensor, data_tensor[: streamlines_to_add]], dim=0)
    origStreamlines = origStreamlines[random_indices] 
    return data_tensor.reshape(-1, context_size, nPoints, spaceDim), origStreamlines, random_indices



@torch.inference_mode()
def model_forward_pass(model: nn.Module, 
                       eval_batch_size: int,
                       data_tensor: torch.Tensor,
                       return_anatomical_clusters: bool,
                       mapping_800_800_to_43: torch.Tensor,
                       device: torch.device) -> torch.Tensor:
    """
    Forward pass of the data through the model. This function asumes that the set of streamlines is shuffled.
    Device transfer is done in an asynchronous fashion.  
    Args:
        model: The model to be evaluated.
        eval_batch_size: The batch size for the evaluation.
        device: The device on which the model is evaluated.
        data_tensor: The input data tensor (CPU). It is already normalized to the [-1,1] cube, Shape [N, context_size, 15, 3]
    Returns:
        output_tensor: The output tensor (CPU) of the model.
    """
    non_blocking = False
    
    current_batch = data_tensor[0: eval_batch_size].to(device, non_blocking=non_blocking)
    model.eval()
    model = model.to(device, non_blocking=non_blocking)
    y_pred = []
    for i in tqdm(range(0, len(data_tensor), eval_batch_size)):
        next_batch = data_tensor[i + eval_batch_size : i + 2 * eval_batch_size].to(device, non_blocking=non_blocking)
        batch = current_batch # normalization_function(current_batch)
        output = model(batch) # Shape [bs, dim_out, context_size, dim_out]
        if return_anatomical_clusters:
            y_pred.append(mapping_800_800_to_43[output.argmax(dim=-1).flatten()].to("cpu", non_blocking=non_blocking)) # Shape [bs * context_size] in the range [0, 42]
        else:
            y_pred.append(output.argmax(dim=-1).flatten().to("cpu", non_blocking=non_blocking)) # Shape [bs * context_size] in the range [0, 1599]
        current_batch = next_batch
    
    # torch.cuda.synchronize(device)
    return torch.cat(y_pred, dim=0) # Shape [N x context_size]



def load_args_and_model(run_path: Union[Path, str], device: torch.device):
    run_path = Path(run_path)
    args = load_args_from_yaml(run_path / "args" / "args.yml")

    # Load Embedding
    embedding = get_embedding_layer(num_support_points_per_streamline=args.num_support_points_per_streamline, 
                                       d_model=args.d_model)

    # Load Model w.r.t. that config
    model = TransformerModel(num_layers=args.num_layers, 
                            d_model=args.d_model,
                            nhead=args.nhead,
                            dim_feedforward=args.dim_feedforward,
                            dropout=args.dropout,
                            embedding_layer=embedding,
                            dim_class_hidden=args.dim_class_hidden,
                            dim_out=args.dim_out)
    model = model.to(device)
    model = nn.DataParallel(model)
    latestModel = get_latest_snapshot(run_path / "model")
    if latestModel is None:
        raise ValueError(f"The Path {run_path} does not contain a model to evaluate")
    checkpoint = torch.load(latestModel, weights_only=True, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model = model.module

    return args, model


def tck_to_tensor(tck_path: Union[os.PathLike, str]) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Function that reads a tck file.
    Args:
        tck_path: Path to the tck file.
    Returns:
        shortened_streamlines: A tensor of shape (N, 15, 3) where N is the number of streamlines in the tck file.
        origStreamlines: A list of numpy arrays where each numpy array is a streamline from the tck file.
    """
    tck_path = Path(tck_path)
    assert tck_path.suffix == ".tck", f"The file {tck_path} is not a '.tck' file."
    # Load Streamlines from tck file using nibabel
    time_to_load_tck = time.time()
    tractogram = nib.streamlines.load(tck_path)
    loaded_streamlines = tractogram.streamlines
    print(f"Time to load tck file with {len(loaded_streamlines)}: {time.time() - time_to_load_tck:.2f}s")
    time_to_shorten_streamlines = time.time()
    origStreamlines = []
    shortened_streamlines = []
    # Select all Streamlines with length ≥ 15
    for streamline in loaded_streamlines:
        origStreamlines.append(streamline)
        indices = np.round(np.linspace(start=0, stop=len(streamline) - 1, num=15)).astype(int)
        selected_points = streamline[indices]
        shortened_streamlines.append(selected_points)
    shortened_streamlines = torch.from_numpy(np.array(shortened_streamlines)).pin_memory() # Shape [N, 15, 3]
    print(f"Time to shorten streamlines: {time.time() - time_to_shorten_streamlines:.2f}s")
    return shortened_streamlines, np.array(origStreamlines, dtype=object)



def results_to_tck(origStreamlines: Union[list, torch.Tensor, np.ndarray], 
                   y_pred_43: torch.Tensor, 
                   int_to_label: np.ndarray, 
                   output_path: Union[os.PathLike, str]):
    """
    Function that writes the results to a tck file.
    Args:
        origStreamlines: A list of numpy arrays where each numpy array is a streamline from the tck file.
        y_pred_43: The predicted labels. Shape [N,] integer values in the range [0, 42].
        int_tpo_label: The mapping from integer to label. Each of the 43 labels can be accessed by the index.
    """
    assert len(origStreamlines) == len(y_pred_43), "The number of streamlines and the number of predictions do not match."
    # Create a new tractogram
    output_path = Path(output_path)
    y_pred_43_np = y_pred_43.numpy()
    if not output_path.exists():
        output_path.mkdir(parents=True)
    tcks = [tck_io.Tck(output_path / f"{label}.tck") for label in int_to_label]
    for tck in tcks:
        tck.force = True  # Force overwrite if the file already exists
    [tck.write({}) for tck in tcks]
    for k in range(len(int_to_label)):
        to_save = origStreamlines[y_pred_43_np == k]
        print(f"Number of streamlines in {int_to_label[k]}: {len(to_save)}")
        if len(to_save) > 0:
            to_save = np.ascontiguousarray(np.vstack([np.vstack([s, np.array([np.nan]*3)]) for s in to_save]), dtype='<f4')
            with open(output_path / f"{int_to_label[k]}.tck", "ab") as f:
                f.write(np.ndarray.tobytes(to_save))
    for tck in tcks:
        tck.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained rapidParc model on a tck file.")
    parser.add_argument("--tck_path", type=str, required=True, help="Path to the input tck file.")
    parser.add_argument("--out_path", type=str, default="output_tcks", help="Path to the output folder, where the resulting tck files will be stored. If the folder does not exist, it will be created.")
    parser.add_argument("--model_folder_path", type=str, required=True, help="Path to the folder containing the trained model and the args.yml file.")
    parser.add_argument("--eval_batch_size", type=int, default=512, help="Batch size for evaluation.")
    parser.add_argument("--eval_context_size", type=int, default=2000, help="Context size for evaluation.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for evaluation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--print_time", action="store_true", help="Whether to print time measurements.")
    args = parser.parse_args()

    rapidParcTckEval(tck_path=args.tck_path,
                     out_path=args.out_path,
                     model_folder_path=args.model_folder_path,
                     eval_batch_size=args.eval_batch_size,
                     eval_context_size=args.eval_context_size,
                     seed=args.seed,
                     device=torch.device(args.device),
                     print_time=args.print_time)
    