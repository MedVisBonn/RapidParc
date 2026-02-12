import torch
from pathlib import Path
from torch.utils.data import Dataset
from utils.transforms3D import normalize_to_identity_cube
import numpy as np
import pickle
import pandas as pd
from collections import defaultdict
from typing import Tuple, Optional, Union, Dict, List
from collections.abc import Callable
import os


def get_TractCloud_train_returner(inputPath: Union[str, os.PathLike],
                 device: torch.device,
                 n_vertices: Optional[int] = None) \
                    -> Callable[[Union[List[int], torch.Tensor]], Tuple[torch.Tensor, torch.Tensor]]:
    """
        This functions is an alternative to Pytorch Datasets and Dataloaders.
        In our case, transferring the data to the GPU took a relevant amount of time.
        Therefore, this closure leaves the data permanently on the GPU. 
        It returns two functions: one for the training batch and one for the validation set. 

        Parameters:
            inputPath:
                Path to the TractCloud Dataset Folder

            device:
                The device for the train set

            n_vertices (optional, int): 
                The number of verticies requested for further downsampling

        Returns:
            train_returner ((idx) -> streamlines[idx], labels_800_800[idx]):
                A function that returns the tractogram of a subject with this idx
    """

    inputPath = Path(inputPath)
    assert inputPath.exists(), f"Path {inputPath} does not exist"
    streamlines, _, labels_800_800 = getTractCloudDataset(tractCloud_Dataset_dir=inputPath, split= "train")

    # Convert to torch tensors and move to device
    streamlines = torch.from_numpy(streamlines).to(device, non_blocking=True) # Shape [numSubjects, numStreamlines, seqLength, spaceDim]
    labels_800_800 = torch.from_numpy(labels_800_800).to(device, non_blocking=True) # Shape [numSubjects, numStreamlines]
    streamlines = normalize_to_identity_cube(streamlines) # Shape [numSubjects, numStreamlines, seqLength, spaceDim]
    if n_vertices is not None:
        streamlines = _downsample_streamlines(streamlines, n_vertices=n_vertices) # Shape [numSubjects, numStreamlines, n_vertices, spaceDim]

    def train_returner(indices: Union[List[int], torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # the streamlines are normalized at that point
        return streamlines[indices], labels_800_800[indices] 
    
    return train_returner




def get_TractCloud_eval_returner(inputPath: Union[str, os.PathLike],
                 device: torch.device,
                 eval_context_size: int,
                 seed: int = 42,
                 n_vertices: Optional[int] = None,
                 split: str = "val") \
                    -> Callable[[], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
        This functions is an alternative to Pytorch Datasets and Dataloaders.
        In our case, transferring the data to the GPU took a relevant amount of time.
        Therefore, this closure leaves the data permanently on the GPU. 
        It returns two functions: one for the training batch and one for the validation set. 

        Parameters:
            inputPath:
                Path to the TractCloud Dataset Folder

            device:
                The device for the train set
            
            eval_context_size:
                The constext size used for evaluation

            seed (int, default 42):
                The random seed used for permuting the validation tractograms.

            n_vertices (optional, int): 
                The number of verticies requested for further downsampling

            split (str):
                Defines what split should be returned, in ["train", "val", "test"]

        Returns:
            eval_returner (() -> streamlines, labels_43, labels_800_800):
                A function that returns the whole validation set with shape [bs, eval_context_size, seqLength, 3]
    """
    assert split in ["train", "val", "test"], f'split = {split} must be in ["train", "val", "test"]'
    inputPath = Path(inputPath)
    assert inputPath.exists(), f"Path {inputPath} does not exist"
    streamlines, labels_43, labels_800_800 = getTractCloudDataset(tractCloud_Dataset_dir=inputPath, split=split)

    rng = np.random.default_rng(seed)
    
    # Permute the streamlines for random subtractogram partitioning
    for i in range(len(streamlines)):
        random_permutation = rng.permutation(streamlines[i].shape[0])
        streamlines[i] = streamlines[i][random_permutation]
        labels_43[i] = labels_43[i][random_permutation]
        labels_800_800[i] = labels_800_800[i][random_permutation]

    # Convert to torch tensors
    streamlines = torch.from_numpy(streamlines).to(device, non_blocking=True) # Shape [numSubjects, numStreamlines, seqLength, spaceDim]
    labels_43 = torch.from_numpy(labels_43).to(device, non_blocking=True) # Shape [numSubjects, numStreamlines]
    labels_800_800 = torch.from_numpy(labels_800_800).to(device, non_blocking=True) # Shape [numSubjects, numStreamlines]

    # Reshape the streamlines to context size bits
    _, numStreamlinesPerSubject, seqLength, spaceDim = streamlines.shape
    assert numStreamlinesPerSubject % eval_context_size == 0, \
        f"For faster evaluation, we assume that numStreamlinesPerSubject % context_size == 0, " +\
        f"but got numStreamlinesPerSubject={numStreamlinesPerSubject} and context_size={eval_context_size}"
    streamlines = streamlines.reshape(-1, eval_context_size, seqLength, spaceDim)
    streamlines = normalize_to_identity_cube(streamlines)
    labels_43 = labels_43.reshape(-1, eval_context_size)
    labels_800_800 = labels_800_800.reshape(-1, eval_context_size)
    
    if n_vertices is not None:
        # Downsample streamlines to n_vertices preserving both endpoints!
        streamlines = _downsample_streamlines(streamlines, n_vertices)

    def eval_returner() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # they are normed at that point
        return streamlines, labels_43, labels_800_800
    
    return eval_returner



def getTractCloudDataset(tractCloud_Dataset_dir: Union[str, os.PathLike], split: str) \
    -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
        This function returns the TractCloud dataset as reshaped Numpy Arrays and two helper arrays
        Parameters:
            tractCloud_Dataset_dir: 
                Path of the TractCloud TrainData_800clu800ol folder

            split (str):
                Defines what split should be returned, in ["train", "val", "test"]
        
        Returns:
            streamlines (np.ndarray, shape [N, 10000, 15, 3]): 
                10000 streamlines of N subjects of length 15 each 

            labels_43 (np.ndarray, shape [N, 10000]):
                The labels of the streamlines for the TractCloud dataset, each value in 0, ... 42 
                
            labels_800_800 (np.ndarray, shape [N, 10000]): 
                The labels of the streamlines in the org atlas, each value in 0, ..., 1599
    """
    assert split in ["train", "val", "test"], f'Split must be in ["train", "val", "test"] but got {split}'

    input_path = Path(tractCloud_Dataset_dir)
    assert input_path.is_dir(), f"Expected {input_path} to be the path of the unzipped TractCloud Train dataset"

    with open(input_path/f'{split}.pickle', 'rb') as f:
        data = pickle.load(f)
        
    streamlines = data["feat"].reshape(-1, 10000, 15, 3)
    labels_800_800 = data["label"].reshape(-1, 10000)
    labels_43 = get_mapping_from_800_800_to_43(input_path=input_path)[labels_800_800]

    # # Possible check if reshaping is correctly
    # subjectIDs = data["subject_id"].reshape(-1, 10000)
    # flat_stls = data["feat"]
    # flat_subIDS = data["subject_id"]
    # flat_labels = data["label"]
    # iFlat, iRsh, jRsh = 0, 0, 0
    # while iFlat < len(flat_stls):
    #     assert (streamlines[iRsh, jRsh] == flat_stls[iFlat]).all()
    #     assert (subjectIDs[iRsh, jRsh] == flat_subIDS[iFlat]).all()
    #     assert (labels_800_800[iRsh, jRsh] == flat_labels[iFlat]).all()
    #     iFlat += 1
    #     jRsh += 1
    #     if jRsh >= 10000:
    #         jRsh = 0
    #         iRsh += 1

    return streamlines, labels_43, labels_800_800



def get_int_to_label(input_path: Union[str, os.PathLike], save_path: Optional[Union[str, os.PathLike]] = None) -> np.ndarray:
    """
    Getter function for array that maps all tract indices 0, 1, ..., 42 
    to their names, i.e. "AF", "CB", ..., "Other"

    Parameters:
        input_path:
            Input folder path of tractCloud training data

    Returns:
        int_to_label (np.ndarray, shape [43], str):
            Represents a mapping from the 43 classes to their names
    """
    input_path = Path(input_path)
    int_to_label = np.array(list(obtain_TractClusterMapping(input_path).keys()))
    if save_path:
        save_path = Path(save_path) / "int_to_label.npy"
        if not save_path.exists():
            np.save(arr=int_to_label, file=save_path)
    return int_to_label



def get_mapping_from_800_800_to_43(input_path: Union[str, os.PathLike], save_path: Optional[Union[str, os.PathLike]] = None) -> np.ndarray:
    input_path = Path(input_path)
    mapping_from_800_800_to_43 = np.array(
        cluster2tract_label(
            lst = [i for i in range(1600)], 
            ordered_tract_cluster_mapping_dict=obtain_TractClusterMapping(input_path), 
            output_lst=False)
        )
    if save_path:
        save_path = Path(save_path) / "mapping_from_800_800_to_43.npy"
        if not save_path.exists():
            np.save(arr=mapping_from_800_800_to_43, file=save_path)
    return mapping_from_800_800_to_43



def _downsample_streamlines(streamlines: torch.Tensor, n_vertices: int) -> torch.Tensor:
    # Downsample streamlines to n_vertices preserving both endpoints!
    num_subjects, num_streamlines, seqLength, spaceDim = streamlines.shape
    if seqLength <= n_vertices:
        return streamlines  # No downsampling needed
    indices = torch.linspace(0, seqLength - 1, steps=n_vertices).long().to(streamlines.device)
    return streamlines[:, :, indices, :]




# ********************* Functions that originate from the TractCloud repository *********************

def cluster2tract_label(lst: Union[List, np.ndarray], 
                        ordered_tract_cluster_mapping_dict: Dict[str, List[str]], 
                        output_lst: bool = True
                        ) -> Union[List, np.ndarray]:
    """
        Convert cluster labels to tract labels

        This function is copied from TractCloud
        https://github.com/SlicerDMRI/TractCloud/blob/2c2869cde12b8d617d21be8e19fd33ae73b5f3bc/utils/funcs.py#L124
    """
    if type(lst) != np.ndarray:
        array = np.array(lst)
    else:
        array = lst
        
    org_array = np.copy(array)   # The original label/pred lst
    
    num_tracts = len(ordered_tract_cluster_mapping_dict.keys())  # anatomically plausible tracts + 1 other tract 43
    
    # convert cluster idx to tract idx. Iterate tracts then clusters
    for idx_tract, tract_name in enumerate(ordered_tract_cluster_mapping_dict.keys()):
        cur_clusters_names = ordered_tract_cluster_mapping_dict[tract_name]
        for cluster_name in cur_clusters_names:
            cluster_idx = int(cluster_name.split('_')[1][2:])-1   # '00031'->30,'00322'->321.  Don't use strip('0'), it will strip 0 in '80' as well.
            array[org_array==cluster_idx] = idx_tract  # plausible tract, starts from 0
            array[org_array==cluster_idx+800] = num_tracts-1  # implausible tract (outlier)
    
    if output_lst:   # when array is 1D
        new_lst = list(array)
    else:           # We could get array with 2D/3D/nD shape, e.g., (num_fiber_per_brian, num_points_per_fiber)
        new_lst = array
    
    return new_lst



def obtain_TractClusterMapping(input_path: Union[str, os.PathLike]) -> Dict[str, List[str]]:
    """
        The obtained dictionary is {'tract name': ['cluster_xxx','cluster_xxx', ... 'cluster_xxx']}. Cluster index starts from 1.

        This function is copied from TractCloud
        https://github.com/SlicerDMRI/TractCloud/blob/2c2869cde12b8d617d21be8e19fd33ae73b5f3bc/utils/funcs.py#L99
    """
    ordered_tract_names = list(obtain_TractFullName().keys())
    ordered_tract_cluster_mapping_dict = {}   # keep the order we like association, projection, commissural, cerebellar, superficial 
    
    cluster_annotation_pd = pd.read_excel(Path(input_path)/'FiberClusterAnnotation_Updated20230110.xlsx')
    cluster_tract_mapping = {clu:tra for clu, tra in zip(cluster_annotation_pd['Cluster Index'], cluster_annotation_pd['Final'])}

    # only include clusters into the anatomical tracts
    tract_cluster_mapping_dict = defaultdict(list)  # {'tract_x': ['cluster_xxx','cluster_xxx']}
    for key, value in cluster_tract_mapping.items():
        tract_cluster_mapping_dict[value].append(key)   
    
    # sort the tract order in the dictionary and add clusters belong to other tracts
    all_clusters_lst = list(cluster_annotation_pd['Cluster Index'])
    clusters_belong_anatomical_tracts_lst = []
    for tract_name in ordered_tract_names:
        ordered_tract_cluster_mapping_dict[tract_name] = tract_cluster_mapping_dict[tract_name]
        clusters_belong_anatomical_tracts_lst.extend(tract_cluster_mapping_dict[tract_name])
    ordered_tract_cluster_mapping_dict['Other'] = sorted(list(set(all_clusters_lst)-set(clusters_belong_anatomical_tracts_lst))) 
    
    return ordered_tract_cluster_mapping_dict



def obtain_TractFullName() -> Dict[str, str]: 
    """
        Returns a dict of the full tract names.

        This function is copied from TractCloud: 
        https://github.com/SlicerDMRI/TractCloud/blob/2c2869cde12b8d617d21be8e19fd33ae73b5f3bc/utils/funcs.py#L50

    """   
    tract_name_full_dict = {
        'AF': 'arcuate fasciculus',  # association 
        'CB': 'cingulum bundle', 
        'EC': 'external capsule',
        'EmC': 'extreme capsule',
        'ILF': 'inferior longitudinal fasciculus', 
        'IOFF': 'inferior occipito-frontal fasciculus', 
        'MdLF': 'middle longitudinal fasciculus', 
        'SLF-I': 'superior longitudinal fasciculus I', 
        'SLF-II': 'superior longitudinal fasciculus II', 
        'SLF-III': 'superior longitudinal fasciculus III', 
        'UF': 'uncinate fasciculus', 
        'CST':'corticospinal tract',  # projection 
        'CR-F':'corona-radiata-frontal (excluding the CST)', 
        'CR-P':'corona-radiata-parietal (excluding the CST)', 
        'SF':'striato-frontal', 
        'SO':'striato-occipital', 
        'SP':'striato-parietal',
        'TF':'thalamo-frontal',
        'TO':'thalamo-occipital', 
        'TT': 'thalamo-temporal', 
        'TP': 'thalamo-parietal', 
        'PLIC':'posterior limb of internal capsule',
        'CC1': 'corpus callosum 1',    # Commissural 
        'CC2': 'corpus callosum 2',  
        'CC3': 'corpus callosum 3', 
        'CC4': 'corpus callosum 4', 
        'CC5': 'corpus callosum 5', 
        'CC6': 'corpus callosum 6', 
        'CC7': 'corpus callosum 7', 
        'CPC': 'cortico-ponto-cerebellar',     # cerebellar
        'ICP': 'inferior cerebellar peduncle', 
        'Intra-CBLM-I-P':'intracerebellar input and Purkinje tract', 
        'Intra-CBLM-PaT':'intracerebellar parallel tract',
        'MCP':'middle cerebellar peduncle', 
        'Sup-F':'superficial-frontal',  # superficial
        'Sup-FP':'superficial-frontal-parietal',
        'Sup-O':'superficial-occipital',
        'Sup-OT':'superficial-occipital-temporal',
        'Sup-P':'superficial-parietal',
        'Sup-PO':'superficial-parietal-occipital',
        'Sup-PT':'superficial-parietal-temporal',
        'Sup-T':'superficial-temporal'
        }
    
    return tract_name_full_dict



if __name__ == '__main__':
    context_size = 2000
    inputPath="tractCloud_Dataset/TrainData_800clu800ol"
    train_ret= get_TractCloud_train_returner(inputPath=inputPath,
                            device=torch.device("cpu"))
    print(f"Shape of first train tractogram: {train_ret(torch.tensor(0))[0].shape}")