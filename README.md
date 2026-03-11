# RapidParc

This repository provides RapidParc, a fast, accurate, and lesion-robust transformer-based algorithm for registration-free parcellation of streamlines from diffusion MRI tractography. If you use our model and/or code in your research, please cite the corresponding publication:
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
from RapidParc import RapidParc
predictions = RapidParc(...)
```

We published three pre-trained models:
- `rapidparc` the general purpose model.
- `hemiaug` the hemispherotomy-stable RapidParc model.
- `rapidparc_v8` the vanilla model, but each input streamline gets resampled to 8 instead of 15 supporting points per streamline. This model was used during benchmarking.

The function [RapidParc](https://github.com/MedVisBonn/RapidParc/blob/main/run.py#L59) accepts multiple input-formats like `list`, `torch.Tensor` and `numpy.ndarray`. The streamlines do not have to have an equal number of supporting points. *RapidParc* resamples them automatically. For more details take a look at the Docstring of the function. 

**Example Call:**
```python
from RapidParc import RapidParc
import torch

# Your tractogram goes here
inputTractogram = torch.rand([5000, 15, 3])

predictions = RapidParc(
    model_name_or_path = "rapidparc",
    inputTractogram = inputTractogram,
    eval_batch_size = 128,
    eval_context_size = 2000,
    device = torch.device("cpu"),
    print_time = True,
    print_class_distribution = False
)

print(f"Output shape: {predictions.shape}")
print(f"Output: {predictions}")
classes, counts = torch.unique(predictions, return_counts=True)
print(f"Output summary: {[f'{cl}: {co}' for cl, co in zip(classes, counts)]}")
```
```
Time to load model: 0.05s
Time to shuffle data: 0.00s
5000 Streamlines are loaded and preprocessed.
100%|█████████████████████████████████████| 1/1 [00:00<00:00,  5.28it/s]
Prediction is done in 0.20s
Total time: 0.27s

Output shape: torch.Size([5000])
Output: tensor([42, 42, 42,  ..., 42, 42, 42])
Output summary: ['33: 1', '42: 4999']
```
I.e. our random paths are getting classified as outliers.

## Run RapidParc from CLI (for `.tck` tractograms)
If you have saved a tractogram in a `.tck` file, you can use the following wrapper function, which also handles file I/O:
```python
from RapidParc import RapidParcTckEval
RapidParcTckEval(...)
```

This function can be called via the CLI:

**Example Call:**
```sh
rapidparc --tck_path tractogram.tck \
 --out_path parcellated_tractogram \
 --model_name_or_path rapidparc \
 --eval_batch_size 64 \
 --eval_context_size 2000 \
 --device cpu \
 --print_time \
 --print_class_distribution
```

```
Time to load tck file with 926641 streamlines: 1.50s
Time to shorten streamlines: 5.54s
Time to load model: 0.07s
Time to shuffle data: 0.08s
926641 Streamlines are loaded and preprocessed.
100%|█████████████████████████████████████| 8/8 [00:12<00:00,  1.60s/it]
Prediction is done in 12.83s
Class distribution
      Class       | Count
 0 AF                (6254) ▇▇▇▇▇▇
 1 CB               (14252) ▇▇▇▇▇▇▇▇▇▇▇▇▇
 2 EC                (3795) ▇▇▇▇
 3 EmC               (5990) ▇▇▇▇▇▇
 4 ILF              (17420) ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇
 5 IOFF             (19687) ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇
 6 MdLF             (10216) ▇▇▇▇▇▇▇▇▇▇
 7 SLF-I            (10937) ▇▇▇▇▇▇▇▇▇▇
 8 SLF-II           (28028) ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇
 9 SLF-III           (5348) ▇▇▇▇▇
10 UF                (4185) ▇▇▇▇
11 CST              (24662) ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇
12 CR-F             (14493) ▇▇▇▇▇▇▇▇▇▇▇▇▇
13 CR-P              (2089) ▇▇
14 SF               (23246) ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇
15 SO                (5810) ▇▇▇▇▇▇
16 SP                (5294) ▇▇▇▇▇
17 TF               (40690) ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇
18 TO                (4753) ▇▇▇▇▇
19 TT                (8700) ▇▇▇▇▇▇▇▇
20 TP                (9958) ▇▇▇▇▇▇▇▇▇
21 PLIC              (4345) ▇▇▇▇
22 CC1               (1477) ▇▇
23 CC2              (18125) ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇
24 CC3               (7703) ▇▇▇▇▇▇▇
25 CC4               (9848) ▇▇▇▇▇▇▇▇▇
26 CC5               (7267) ▇▇▇▇▇▇▇
27 CC6              (18489) ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇
28 CC7               (8638) ▇▇▇▇▇▇▇▇
29 CPC                (875) ▇
30 ICP                (866) ▇
31 Intra-CBLM-I-P    (6059) ▇▇▇▇▇▇
32 Intra-CBLM-PaT   (10004) ▇▇▇▇▇▇▇▇▇
33 MCP                (929) ▇
34 Sup-F            (56703) ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇
35 Sup-FP            (5456) ▇▇▇▇▇
36 Sup-O             (7084) ▇▇▇▇▇▇▇
37 Sup-OT            (6879) ▇▇▇▇▇▇▇
38 Sup-P            (23330) ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇
39 Sup-PO            (6748) ▇▇▇▇▇▇
40 Sup-PT           (19768) ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇
41 Sup-T            (11187) ▇▇▇▇▇▇▇▇▇▇
42 Outlier 'Other' (429054) ###################################################
Total time: 13.28s
Files saved in parcellated_tractogram/
```

# Test RapidParc on the *TractCloud test split*

To benchmark RapidParc on the TractCloud test split, use the [test function in test.py](https://github.com/MedVisBonn/RapidParc/blob/main/test.py#L23). 

**Example Call:**
```python
import RapidParc
import torch

acc_43, f1_43 = RapidParc.test(
    model_name_or_path = "rapidparc_v8",
    applyTestSetAugmentations = False,
    eval_batch_size = 128,
    evaluation_context_size = 2000,
    device = torch.device("cpu"),
    print_classification_report = False
)
print(f"Accuracy: {100 * acc_43:.2f} %, Macro F1 Score: {100 * f1_43:.2f} %")
```
This prints out a classification report and creates a confusion matrix in `plots/rapidParc_v8`.
```
Testing experiment rapidparc_v8 on the test set without augmentations
Duration: 4.34 seconds
Accuracy: 94.43 %, Macro F1 Score: 93.26 %
```

This test function can also be called from CLI.
```sh
rapidparc-test -h
```

# Retrain RapidParc on the TractCloud train split

If you want to retrain RapidParc on your own, please take a look at [train.py](https://github.com/MedVisBonn/RapidParc/blob/main/train.py#L25). A minimal run can be started using:
```sh
rapidparc-train
```

# Known issues

- If you get bad evaluation results, there might be a flip in the axis of the tractogram. To debug it, you might visualize a TractCloud tractogram (i.e. take one from `utils/dataset.py:getTractCloudDataset`) together with your tractogram.