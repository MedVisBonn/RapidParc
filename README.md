# RapidParc
This repository provides RapidParc, a fast, accurate, and lesion-robust transformer based algorithm for registration-free parcellation of streamlines from diffusion MRI tractography. If you use our model and/or code in your research, please cite the corresponding publication:
> Bisten, J., von Bornhaupt, V., Grün, J., Bauer, T., Rüber, T., Schultz, T.  
> RapidParc: A Global-Context Transformer for Parallel, Accurate, and Lesion-Robust Tractogram Parcellation  
> Imaging Neuroscience 2026 (accepted for publication)

# Installation
1.  Clone the repository, i.e.
    ```sh
    git clone https://github.com/MedVisBonn/RapidParc.git
    cd RapidParc
    ``` 
2. Install the python environment with dependencies, i.e.
    ```sh
    python3 -m venv .venv
    python3 -m pip install -U pip
    python3 -m pip install -r requirements.txt
    ```
3. Download the [trained RapidParc Models](https://github.com/MedVisBonn/RapidParc/releases) and unzip theim into the local repository.
4. (Optional) If you want to retrain RapidParc or test RapidParc on the TractCloud test split, please download the TractCloud streamline dataset. i.e. [read the slice licence agreement](https://github.com/SlicerDMRI/TractCloud/blob/main/LICENSE) and than run the following commands: 
    ```sh
    cd TractCloud_Dataset
    source getTrainingData.sh
    cd ..
    ```

# Parcellate with RapidParc
 This repository has three main entry points: 
 - `run.py`: Parcellate Streamlines
 - `test.py`: Test a RapidParc model on the TractCloud test split 
 - `train.py`: Retrain RapidParc on the TractCloud train split

 ## Run RapidParc from Python
For each valid input tractogram, you can call
 ```python
from RapidParc.run import rapidParc
predictions = rapidParc(...)
```
The function [rapidParc](https://github.com/MedVisBonn/RapidParc/blob/main/run.py#L59) accepts multiple input-formats like `list`, `torch.Tensor` and `numpy.ndarray`. For more details take a look at the docstring of the function. 

**Example Call:**
Assuming you have installed RapidParc as described above a list `inputTractogram` of length $N$ where each entry is a streamline of shape [$n_i$, 3]
```python
from RapidParc.run import rapidParc
import torch

predictions = rapidParc(
    model_folder_path = "trained_models/rapidParc",
    inputTractogram = inputTractogram,
    eval_batch_size = 256,
    eval_context_size = 2000,
    device = torch.device("cpu")
    print_time = True
    )
predictions.shape # Shape [N], where each entry is in {0, ..., 42}
```

## Run RapidParc from CLI (for `.tck` tractograms)
If you have a tractogram stored in a `.tck` file, you can use the command line interface:

**Example Call:**
```sh
run.py --tck_path myBrain.tck --model_folder_path trained_models/rapidParc --print_time
```

# Test RapidParc on the TractCloud test split
To test RapidParc on the TractCloud test split, use the [test function in test.py](https://github.com/MedVisBonn/RapidParc/blob/main/test.py#L23). 

**Example Call:**
```python
from RapidParc.test import test

acc_43, f1_43 = test(experiment_path = "trained_models/rapidParc_v8",
     tractCloudDatasetPath = "TractCloud_Dataset/TrainData_800clu800ol",
     applyTestSetAugmentations = True,
     eval_batch_size = 256,
     evaluation_context_size = 2048,
     device = torch.device("cpu"),
     print_classification_report = True
    )
```
This prints out a classification report and creates a confusion matrix in `trained_models/rapidParc_v8`.

# Retrain RapidParc on the TractCloud train split
For further information take a look at the [argument parsing in train.py](https://github.com/MedVisBonn/RapidParc/blob/main/train.py#L25).

# Common issues
- If you get bad evaluation results, there might be a flip in the axis of the tractogram. To debug it, you might visualize a TractCloud tractogram (i.e. take one from `utils/dataset.py:getTractCloudDataset`) together with your tractogram.