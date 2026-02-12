import torch
import torch.nn as nn
from torch.utils import data
from tqdm import tqdm
import numpy as np
from typing import Union, Tuple, List, Callable, Optional
from pathlib import Path
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from utils.ddp_handling import ddp_is_running
from utils.model_snapshot_handling import get_latest_snapshot, load_snapshot, store_snapshot


def train(get_train_Tensor: Callable[[Union[List[int], torch.Tensor]], Tuple[torch.Tensor, torch.Tensor]],
          get_val_Tensor: Callable[[], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
          num_epochs: int,
          batch_size: int,
          context_size: int, 
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler.LRScheduler,
          model: nn.Module,
          device: torch.device,
          criterion: nn.Module,
          snapshot_path: Union[str, Path],
          snapshot_every: int,
          store_all_snapshots: bool,
          map_800_800_to_43: torch.Tensor,
          transforms_on_gpu: nn.Module,
          aug_p_hemi: float = 0.0,
          aug_method_hemi: str = 'replace') -> nn.Module:
    """ 
    Train function to train the model
    
    Parameters:
        get_train_Tensor (Callable): Function to get the training data
        get_val_Tensor (Callable): Function to get the validation data
        num_epochs (int): Number of epochs to train
        batch_size (int): Batch size
        context_size (int): Context size
        optimizer (torch.optim.Optimizer): Optimizer
        scheduler (torch.optim.lr_scheduler): Scheduler
        model (nn.Module): Model
        device (torch.device): Device
        criterion (nn.Module): Criterion
        snapshot_path (Union[str, Path]): Path to store the snapshots
        snapshot_every (int): Store a snapshot every n epochs
        store_all_snapshots (bool): Store all snapshots
        map_800_800_to_43 (torch.tensor): Mapping from 800x800 labels to 43 labels
        transforms_on_gpu (nn.ModuleList): List of transformations to be applied on the GPU

    Returns:
        nn.Module: Trained model
    """
    if ddp_is_running():
        dist_sampler = DistributedSampler(dataset=torch.utils.data.TensorDataset(torch.arange(70)), 
                                            num_replicas=dist.get_world_size(), 
                                            rank=dist.get_rank(), 
                                            shuffle=True)
    else:
        dist_sampler = torch.arange(70)
    snapshot_path = Path(snapshot_path)
    map_800_800_to_43 = map_800_800_to_43.to(device)

    ####### Load checkpoints if needed #######
    start_epoch = 0
    start_snapshot = get_latest_snapshot(snapshot_path)
    if start_snapshot is not None:
        if dist.get_rank() == 0: print(f"Loading Snapshot {start_snapshot}")
        model, start_epoch, scheduler, optimizer = load_snapshot(snapshot_file_path=start_snapshot, model=model, scheduler=scheduler, optimizer=optimizer)


    if not ddp_is_running() or dist.get_rank() == 0: 
        print(f"Starting from epoch {start_epoch}")
        epoch_range = tqdm(range(start_epoch, num_epochs))
    else: 
        epoch_range = range(start_epoch, num_epochs)

    for epoch in epoch_range:
        ######## validation epoch ########
        if not ddp_is_running() or dist.get_rank() == 0:
            # Evaluate the model only on rank 0
            accuracy_43, accuracy_800_800, val_loss = eval_model(get_val_Tensor=get_val_Tensor, 
                                        model=model, 
                                        criterion=criterion,
                                        map_800_800_to_43=map_800_800_to_43)

        ######## training epoch ########
        cur_loss_iters = train_epoch(model=model,
                                     epoch=epoch,
                                     batch_size=batch_size,
                                     context_size=context_size,
                                     get_train_Tensor=get_train_Tensor,
                                     optimizer=optimizer,
                                     scheduler=scheduler,
                                     criterion=criterion,
                                     transforms_on_gpu=transforms_on_gpu,
                                     aug_p_hemi=aug_p_hemi,
                                     aug_method_hemi=aug_method_hemi,
                                     device=device,
                                     distributed_sampler=dist_sampler)

        if ddp_is_running() and dist.get_world_size() > 1:
            dist.all_reduce(cur_loss_iters, op=dist.ReduceOp.SUM)
            cur_loss_iters = cur_loss_iters.cpu() / dist.get_world_size()
            mean_loss = cur_loss_iters.mean()
        else:
            cur_loss_iters = cur_loss_iters.cpu()
            mean_loss = cur_loss_iters.mean()

        ######## logging ########
        if not ddp_is_running() or dist.get_rank() == 0:

            ######## Checkpoint ######## # 
            if (epoch + 1) % snapshot_every == 0 or epoch == num_epochs - 1:
                print(f"Epoch {epoch+1}/{num_epochs}")
                print(f"    Train loss: {mean_loss}")
                # print(f"    Valid loss: {val_loss}")
                # print(f"    Valid accuracy 43: {accuracy_43}")
                # print(f"    Valid accuracy 800_800: {accuracy_800_800}")
                print("\n")
                snapshot_file_path = snapshot_path / f"snapshot_{epoch+1}.pt" if store_all_snapshots else snapshot_path / "snapshot.pt"
                store_snapshot(snapshot_file_path, model, epoch, scheduler, optimizer)
                print(f"Snapshot stored at {snapshot_file_path}")

    return model



def train_epoch(model: nn.Module,
                epoch: int,
                batch_size: int,
                get_train_Tensor: Callable[[Union[List[int], torch.Tensor]], Tuple[torch.Tensor, torch.Tensor]],
                optimizer: torch.optim.Optimizer,
                criterion: nn.Module,
                scheduler: torch.optim.lr_scheduler.LRScheduler,
                distributed_sampler: Union[DistributedSampler, torch.Tensor],
                context_size: int,
                transforms_on_gpu: nn.Module,
                aug_p_hemi: float,
                aug_method_hemi: str,
                device: Union[int, torch.device]
                ) -> torch.Tensor:
    """ 
    Training the model for one epoch
    Parameters:
        model (nn.Module): Model
        epoch (int): Epoch
        batch_size (int): Batch size
        get_train_Tensor (Callable): Function to get the training data
        optimizer (torch.optim.Optimizer): Optimizer
        criterion (nn.Module): Criterion
        scheduler (torch.optim.lr_scheduler): Scheduler
        distributed_sampler (DistributedSampler | torch.Tensor): Distributed Sampler
        context_size (int): Context size
        transforms_on_gpu (nn.ModuleList): List of transformations to be applied on the GPU
        device (Union[int, torch.device]): Device

    Returns:
        torch.tensor: Loss list
    """
    with torch.no_grad():
        if ddp_is_running():
            distributed_sampler.set_epoch(epoch)
        data, labels_800_800 = get_train_Tensor(list(distributed_sampler))

        # Random subtractogram-partitioning: 
        # -> Shuffle Streamlines for each Subject, bring in ContextSize_bits
        num_subjects, num_streamlines, num_points, stml_dim = data.shape
        perms = torch.stack([torch.randperm(num_streamlines, device=device) for _ in range(num_subjects)])
        data = data[torch.arange(num_subjects).unsqueeze(-1), perms]
        labels_800_800 = labels_800_800[torch.arange(num_subjects).unsqueeze(-1), perms] 
    
        # Single hemisphere augmentation
        if aug_p_hemi == 0.0:
            data = transforms_on_gpu(data.reshape(-1, context_size, num_points, stml_dim))
        else:
            data, labels_800_800 = _apply_hemi_aug(data, labels_800_800, aug_p_hemi, aug_method_hemi, context_size)
            data = transforms_on_gpu(data)

        labels_800_800 = labels_800_800.reshape(-1, context_size) # [X, context_size] with X = num_subjects * num_streamlines / context_size

        # Shuffle data, note that this operation can be done on every rank independently and we do not need a distributed sampler
        # since the data is already distributed (using a distributed sampler inside the get_train_Tensor function)
        indices = torch.randperm(data.shape[0])
        data = data[indices]
        labels_800_800 = labels_800_800[indices]


    loss_list = torch.zeros((int(np.ceil(len(data) / batch_size))), device=device, requires_grad=False)
    
    model.train()
    for iterIndex, i in enumerate(torch.arange(0, len(data), batch_size)):
        data_batch = data[i:i+batch_size].requires_grad_() # [bs, context_size, num_points, stml_dim]
        labels_800_800_batch = labels_800_800[i:i+batch_size] # [bs, context_size]

        # Clear gradients w.r.t. parameters
        optimizer.zero_grad(set_to_none=True)

        # Forward pass to get output/logits
        outputs = model(data_batch)

        # Calculate Loss: softmax --> cross entropy loss
        # Transpose to have dim_out as 1st dimension (required by CrossEntropyLoss)
        loss = criterion(torch.transpose(outputs, 1, 2), labels_800_800_batch)
        loss_list[iterIndex] = loss

        # Updating parameters
        loss.backward()
        optimizer.step()
    scheduler.step()

    return loss_list



@torch.no_grad()
def _apply_hemi_aug(data, labels_800_800, aug_p_hemi, aug_method_hemi, context_size):
    _, num_streamlines, num_points, stml_dim = data.shape
    data_subjectwise = data.clone()  # before reshaping to [B, context_size, ...]
    labels_subjectwise = labels_800_800.clone()
    
    data = data.reshape(-1, context_size, num_points, stml_dim)        # [B, C, P, D]
    labels_800_800 = labels_800_800.reshape(-1, context_size)          # [B, C]

    for i in range(data.shape[0]):
        if torch.rand(1).item() < aug_p_hemi:
            subtractogram = data[i]                  # [C, P, D]
            centroids = subtractogram.mean(dim=1)    # [C, D]
            x_coords = centroids[:, 0]

            left_mask = x_coords < 0
            right_mask = x_coords >= 0

            assert aug_method_hemi == 'random', 'Only random augmentation is implemented for hemi-augmentation.'

            if aug_method_hemi == 'random':
                subj_id = i // (num_streamlines // context_size)

                subj_data = data_subjectwise[subj_id]           # [N, P, D]
                subj_labels = labels_subjectwise[subj_id]       # [N]

                centroids = subj_data.mean(dim=1)               # [N, D]
                x_coords = centroids[:, 0]
                left_mask = x_coords < 0
                right_mask = x_coords >= 0

                mode = torch.randint(0, 3, (1,), device=data.device).item()  # 0 = left, 1 = right, 2 = both

                if mode == 0 and left_mask.sum() > 0:
                    pool_data = subj_data[left_mask]
                    pool_labels = subj_labels[left_mask]
                elif mode == 1 and right_mask.sum() > 0:
                    pool_data = subj_data[right_mask]
                    pool_labels = subj_labels[right_mask]
                else:
                    pool_data = subj_data
                    pool_labels = subj_labels

                pool_size = pool_data.shape[0]
                sample_idx = torch.randint(0, pool_size, (context_size,), device=data.device)

                data[i] = pool_data[sample_idx]
                labels_800_800[i] = pool_labels[sample_idx]
    
    return data, labels_800_800




@torch.inference_mode()
def eval_model(get_val_Tensor: Callable, 
               model: nn.Module, 
               criterion: nn.Module, 
               map_800_800_to_43: torch.Tensor) -> Tuple[float, float, float]:
    """ Evaluating the model for one epoch

    Parameters:
        get_val_Tensor (Callable): Function to get the validation data
        model (nn.Module): Model
        criterion (nn.Module): Criterion
        map_800_800_to_43 (torch.tensor): Mapping from 800x800 labels to 43 labels

    Returns:
        Tuple[float, float, float]: Accuracy_43, Accuracy_800_800, Loss
    """
    correct_800_800 = 0
    correct_43 = 0

    model.eval()
    data, labels_43, labels_800_800 = get_val_Tensor()

    # Forward pass only to get logits/output
    outputs = model(data) # Shape [bs, dim_out, context_size, dim_out]

    # Transpose to have dim_out as 1st dimension (required by CrossEntropyLoss)
    loss = criterion(torch.transpose(outputs, 1, 2), labels_800_800).item()

    # Get predictions from the maximum value
    preds = torch.argmax(outputs, dim = -1) # [bs, contextSize]
    correct_800_800 = len( torch.where(preds==labels_800_800)[0])
    correct_43 = len( torch.where(map_800_800_to_43[preds]==labels_43)[0])

    total = labels_43.shape[0] * labels_43.shape[1]

    # Total correct predictions and loss
    accuracy_800_800 = 100 * correct_800_800 / total
    accuracy_43 = 100 * correct_43 / total

    return accuracy_43, accuracy_800_800, loss
