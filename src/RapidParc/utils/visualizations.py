import numpy as np
import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, classification_report
import time
from typing import Optional, Union
import os

from .dataset import get_TractCloud_eval_returner




def set_seed(seed: int):
    """
    Set the seed for reproducibility.

    Args:
        seed (int): Seed to set
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def annot_format(val):
    """
    Format the annotations of the confusion matrix such that only values greater than 0 are shown.

    Args:
        val (np.ndarray): Confusion matrix

    Returns:
        np.ndarray: Formatted confusion matrix
    """
    if np.issubdtype(val.dtype, np.integer):
        return np.where(val > 0, np.char.mod('%d', val), '')
    elif np.issubdtype(val.dtype, np.floating):
        return np.where(val > 0.01, np.char.mod('%.2f', val), '')
    else:
        return np.full(val.shape, '')



def create_plot(y_true, y_pred, subject_number, duration, classnames, save_path, unnormalized=False):
    """
    Create a confusion matrix plot.

    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        subject_number (int): Number of the subject
        accuracy (float): Accuracy of the model
        macro_f1_score (float): Macro F1 score of the model
        duration (float): Duration of the prediction
        classnames (np.ndarray): Names of the classes
        save_path (str): Path to save the confusion matrix plot
        unnormalized (bool, optional): Whether to create an unnormalized confusion matrix (default: False)
    """

    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred) * 100
    macro_f1_score = f1_score(y_true=y_true, y_pred=y_pred, average='macro') * 100

    plt.figure(figsize=(25, 20))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    if subject_number:
        if unnormalized:
            cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
            sns.heatmap(cm, annot=annot_format(cm), xticklabels=classnames, yticklabels=classnames, fmt='s', cmap='Blues', linewidth=.5)
            plt.title(f"Unnormalized Confusion Matrix for Subject {subject_number} \n Accuracy: {accuracy:.2f} %, Macro F1: {macro_f1_score:.2f}, Prediction duration: {duration:.2f} s for {len(y_true)} Streamlines")
            plt.savefig(Path(save_path) / f"confusion_matrix_subject_{subject_number}_unnormalized.pdf")
        else:
            cm = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize='true')
            sns.heatmap(cm, annot=annot_format(cm), xticklabels=classnames, yticklabels=classnames, fmt='s', cmap='Blues', linewidth=.5)
            plt.title(f"Confusion Matrix for Subject {subject_number} \n Accuracy: {accuracy:.2f} %, Macro F1: {macro_f1_score:.2f}, Prediction duration: {duration:.2f} s for {len(y_true)} Streamlines")
            plt.savefig(Path(save_path) / f"confusion_matrix_subject_{subject_number}.pdf")
    else:
        if unnormalized:
            cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
            sns.heatmap(cm, annot=annot_format(cm), xticklabels=classnames, yticklabels=classnames, fmt='s', cmap='Blues', linewidth=.5)
            plt.title(f"Unnormalized Confusion Matrix for all Subjects \n Accuracy: {accuracy:.2f} %, Macro F1: {macro_f1_score:.2f}, Prediction duration: {duration:.2f} s for {len(y_true)} Streamlines")
            plt.savefig(Path(save_path) / f"confusion_matrix_all_subjects_unnormalized.pdf")
        else:
            cm = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize='true')
            sns.heatmap(cm, annot=annot_format(cm), xticklabels=classnames, yticklabels=classnames, fmt='s', cmap='Blues', linewidth=.5)
            plt.title(f"Confusion Matrix for all Subjects \n Accuracy: {accuracy:.2f} %, Macro F1: {macro_f1_score:.2f}, Prediction duration: {duration:.2f} s for {len(y_true)} Streamlines")
            plt.savefig(Path(save_path) / f"confusion_matrix_all_subjects.pdf") 
    plt.close()



@torch.inference_mode()
def create_confusion_matrix(model: nn.Module, 
                            split: str, 
                            input_path: Union[str, os.PathLike], 
                            save_path: Union[str, os.PathLike], 
                            map_800_800_to_43: torch.Tensor, 
                            int_to_label: np.ndarray,
                            eval_context_size: int = 2000, 
                            device: Optional[torch.device] = None, 
                            seed: int = 42, 
                            eval_batch_size: int = 64,
                            n_vertices: Optional[int] = None):
    """
    Create a confusion matrix for a given model and test dataloader.

    Parameters:
        model (torch.nn.Module): 
            The trained model

        split (str): 
            What kind of split - in ["train", "val", "test"]

        input_path (str): 
            Path to the input data
        
        save_path (str): 
            Path to save the confusion matrix plot

        map_800_800_to_43 (torch.tensor): 
            Mapping from 800x800 labels to 43 labels

        eval_context_size (int):
            The context size used for evaluation, assumed to be a devider of 10_000 as TractCloud 
            has 10_000 streamlines per subjects
        
        device (torch.device, optional): 
            Device to use (default: None)

        seed (int):
            random seed, cuda backend is set to be deterministic

        n_vertices (int):
            for downsampling
    """

    set_seed(seed)
    if n_vertices:
        print(f'Streamlines will be downsampled to {n_vertices} points for Confusion plots.')

    if Path(save_path).exists() == False:
        Path(save_path).mkdir(parents=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()
    map_800_800_to_43 = map_800_800_to_43.to(device)

    duration = time.time()
    streamlines, labels_43, _ = get_TractCloud_eval_returner(inputPath=input_path, 
                                     device=device, 
                                     eval_context_size=eval_context_size, 
                                     seed=seed,  
                                     n_vertices=n_vertices, 
                                     split=split)()
    prediction_43 = []
    
    # Shapes: [bs, context_size, n_verticies or 15, 3], [bs, context_size], [bs, context_size]
    for i in range(0, len(streamlines), eval_batch_size):
        streamlines_batch = streamlines[i : i + eval_batch_size]
        prediction_800_800_batch = model(streamlines_batch).argmax(dim=-1).flatten() # shape [num_subj * num_streamlines]
        prediction_43.append(map_800_800_to_43[prediction_800_800_batch].cpu().numpy())
    prediction_43 = np.concatenate(prediction_43, axis=0)
    labels_43 = labels_43.flatten().cpu().numpy()  # Shape [num_subj * context_size]

    duration = time.time() - duration


    # Create confusion matrix and classification report
    print(classification_report(y_true=labels_43, y_pred=prediction_43, digits=5, target_names=int_to_label, zero_division=0))

    create_plot(y_true=labels_43, 
                y_pred=prediction_43, 
                subject_number=None, 
                duration=duration, 
                save_path=save_path,
                classnames=int_to_label, 
                unnormalized=False)
    
    create_plot(y_true=labels_43, 
                y_pred=prediction_43, 
                subject_number=None, 
                duration=duration, 
                save_path=save_path,
                classnames=int_to_label, 
                unnormalized=True)
    

def plot_loss_curves(train_losses, val_losses, save_path):
    """
    Plot training and validation loss curves.

    Args:
        train_losses (list): List of training losses
        val_losses (list): List of validation losses
        save_path (str): Path to save the loss curves plot
    """ 
    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(1,2)
    fig.set_size_inches(18,5)

    ax[0].plot(train_losses, c="blue", label="Training Loss", linewidth=3, alpha=0.5)
    ax[0].plot(val_losses, c="red", label="Validation Loss", linewidth=3, alpha=0.5)
    ax[0].legend(loc="best")
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("CE Loss")
    ax[0].set_title("Training Progress (linearscale)")

    ax[1].plot(train_losses, c="blue", label="Training Loss", linewidth=3, alpha=0.5)
    ax[1].plot(val_losses, c="red", label="Validation Loss", linewidth=3, alpha=0.5)
    ax[1].legend(loc="best")
    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel("CE Loss")
    ax[1].set_yscale("log")
    ax[1].set_title("Training Progress (logscale)")
    plt.savefig(Path(save_path) / "loss_curves.pdf")
    plt.close()