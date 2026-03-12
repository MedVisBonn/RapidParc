from typing import Union, Tuple
from pathlib import Path
import torch
import torch.nn as nn
import os


def get_latest_snapshot(snapshot_path: Union[str, os.PathLike]) -> Union[os.PathLike, None]:
    """
        Function to get the latest snapshot file name

        Args:
            snapshot_path (Union[str, Path]): Path to the snapshot directory

        Returns:
            Union[Path, None]: Path to the latest snapshot file
    """
    snapshots = list(Path(snapshot_path).glob("snapshot*.pt"))
    if not snapshots:
        print(f"No snapshots found in {snapshot_path}")
        return None
    if len(snapshots) == 1 and snapshots[0].name == "snapshot.pt":
        return snapshots[0]
    return max(snapshots, key=lambda x: int(x.stem.split("_")[-1]) if "_" in x.stem else -1)



def load_snapshot(snapshot_file_path: Union[str, Path],
                  model: nn.Module,
                  scheduler: torch.optim.lr_scheduler.LRScheduler,
                  optimizer: torch.optim.Optimizer):
    """
        Function to load a snapshot of the model, optimizer and scheduler
    """
    checkpoint = torch.load(snapshot_file_path, weights_only=True)
    model.load_state_dict(checkpoint['model'])
    epoch = checkpoint['epoch'] + 1
    scheduler.load_state_dict(checkpoint['scheduler'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, epoch, scheduler, optimizer



def store_snapshot(snapshot_file_path: Union[str, Path],
                   model: nn.Module,
                   epoch: int,
                   scheduler: torch.optim.lr_scheduler.LRScheduler,
                   optimizer: torch.optim.Optimizer) -> None:
    """
        Function to store a snapshot of the model, optimizer and scheduler
    """
    checkpoint = {
        'model': model.state_dict(),
        'epoch': epoch,
        'scheduler': scheduler.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, snapshot_file_path)



def load_only_model_snapshot(snapshot_path: Union[str, Path],
                  model: nn.Module) -> nn.Module:
    """
        Function to load a snapshot of the model

        Args:
            snapshot_file_path (Union[str, Path]): Path to the snapshot file
            model (nn.Module): Model to be loaded

        Returns:
            nn.Module: Model with loaded weights
    """
    checkpoint = torch.load(snapshot_path, weights_only=True)
    model.load_state_dict(checkpoint['model'])
    return model