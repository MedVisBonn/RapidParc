# RapidParc
This repository provides RapidParc, a fast, accurate, and lesion-robust transformer based algorithm for registration-free parcellation of streamlines from diffusion MRI tractography. If you use our model and/or code in your research, please cite the corresponding publication:
> Justus Bisten, Valentin von Bornhaupt, Johannes Grün, Tobias Bauer, Theodor Rüber, Thomas Schultz; 
> RapidParc: A Global-Context Transformer for Parallel, Accurate, and Lesion-Robust Tractogram Parcellation;   *Imaging Neuroscience* 2026; [doi.org/10.1162/IMAG.a.1168](https://doi.org/10.1162/IMAG.a.1168)

# Installation
1.  Clone the repository, i.e.
    ```sh
    git clone https://github.com/MedVisBonn/RapidParc.git
    cd RapidParc
    ``` 
2. Create a virtual environment, i.e.
    ```sh
    python3 -m venv .venv
    source .venv/bin/activate
    python3 -m pip install -U pip
    ```
3. Install PyTorch. Therefore, go to [pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) and follow the instructions for your build. For this project, we used *PyTorch 2.9.1*.
4. Install RapidParc locally
    ```python
    python3 -m pip install -e .
    ```

# Parcellate with RapidParc
 This repository has three main entry points: 
 - `run.py`: Parcellate Streamlines
 - `test.py`: Benchmark RapidParc on the TractCloud test split yourself
 - `train.py`: Retrain RapidParc on the TractCloud train split yourself

 ## Run RapidParc from Python
To parcellate a tractogram, you can call
```python
from RapidParc import rapidParc
predictions = rapidParc(...)
```
The function [rapidParc](https://github.com/MedVisBonn/RapidParc/blob/main/run.py#L59) accepts multiple input-formats like `list`, `torch.Tensor` and `numpy.ndarray`. For more details take a look at the docstring of the function. 

**Example Call:**
```python
from RapidParc import rapidParc
import torch

inputTractogram = torch.rand([5000, 15, 3])

predictions = rapidParc(
    model_name_or_path = "rapidparc",
    inputTractogram = inputTractogram,
    eval_batch_size = 128,
    eval_context_size = 2000,
    device = torch.device("cpu"),
    print_time = True,
    print_class_distributiion = True
)

print(f"Output shape: {predictions.shape}")
print(f"Output: {predictions}")
```
```sh
Time to load model: 0.05s
Time to shuffle data: 0.00s
5000 Streamlines are loaded and preprocessed.
100%|█████████████████████████████████████| 1/1 [00:00<00:00,  5.28it/s]
Prediction is done in 0.20s
Class distribution
     Class       | Count
 0 AF                (0) 
 1 CB                (0) 
 2 EC                (0) 
 3 EmC               (0) 
 4 ILF               (0) 
 5 IOFF              (0) 
 6 MdLF              (0) 
 7 SLF-I             (0) 
 8 SLF-II            (0) 
 9 SLF-III           (0) 
10 UF                (0) 
11 CST               (0) 
12 CR-F              (0) 
13 CR-P              (0) 
14 SF                (0) 
15 SO                (0) 
16 SP                (0) 
17 TF                (0) 
18 TO                (0) 
19 TT                (0) 
20 TP                (0) 
21 PLIC              (0) 
22 CC1               (0) 
23 CC2               (0) 
24 CC3               (0) 
25 CC4               (0) 
26 CC5               (0) 
27 CC6               (0) 
28 CC7               (0) 
29 CPC               (0) 
30 ICP               (0) 
31 Intra-CBLM-I-P    (0) 
32 Intra-CBLM-PaT    (0) 
33 MCP               (1) 
34 Sup-F             (0) 
35 Sup-FP            (0) 
36 Sup-O             (1) 
37 Sup-OT            (0) 
38 Sup-P             (0) 
39 Sup-PO            (0) 
40 Sup-PT            (0) 
41 Sup-T             (0) 
42 Other          (4998) ██████████████████████████████████████████████████
Total time: 0.27s

Output shape: torch.Size([5000])
Output: tensor([42, 42, 42,  ..., 42, 42, 42])
```
I.e. our random paths are getting classified as outliers.

## Run RapidParc from CLI (for `.tck` tractograms)
If you have a tractogram stored in a `.tck` file, you can use the command line interface:

**Example Call:**
```sh
rapidparc --tck_path tractogram.tck 
    --out_path parcellated_tractogram
    --model_name_or_path rapidparc 
    --eval_batch_size 128
    --eval_context_size 2000
    --device cpu
    --print_time
    --print_class_distributiion
```

# Test RapidParc on the *TractCloud test split*
To benchmark RapidParc on the TractCloud test split, use the [test function in test.py](https://github.com/MedVisBonn/RapidParc/blob/main/test.py#L23). 

**Example Call:**
```python
import RapidParc

acc_43, f1_43 = RapidParc.test(model_name_or_path = "rapidparc_v8"
     applyTestSetAugmentations = True,
     eval_batch_size = 128,
     evaluation_context_size = 2000,
     device = torch.device("cpu"),
     print_classification_report = True
    )
```
This prints out a classification report and creates a confusion matrix in `plots/rapidParc_v8`.

## Call benchmark from CLI
This test function can also been called from CLI using the same arguments:
```sh
rapidparc-test --model_name_or_path rapidparc_v8 --applyTestSetAugmentations --eval_batch_size 128 --evaluation_context_size 2000 --device cpu --print_classification_report
```

# Retrain RapidParc on the TractCloud train split
If you want to retrain RapidParc on your own, please take a look at [train.py](https://github.com/MedVisBonn/RapidParc/blob/main/train.py#L25). A minimal run can be started using:
```sh
rapidparc-train
```

# Known issues
- If you get bad evaluation results, there might be a flip in the axis of the tractogram. To debug it, you might visualize a TractCloud tractogram (i.e. take one from `utils/dataset.py:getTractCloudDataset`) together with your tractogram.