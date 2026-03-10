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
from RapidParc import RapidParc
predictions = RapidParc(...)
```
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
    print_class_distributiion = False
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
 --print_class_distributiion
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
    print_classification_report = True
)
print(f"Accuracy: {100 * acc_43:.2f} %, Macro F1 Score: {100 * f1_43:.2f} %")
```
This prints out a classification report and creates a confusion matrix in `plots/rapidParc_v8`.
```
Testing experiment rapidparc_v8 on the test set without augmentations
Duration: 4.34 seconds
                precision    recall  f1-score   support

            AF    0.92047   0.97110   0.94511      2491
            CB    0.92190   0.97370   0.94709      3346
            EC    0.84697   0.93322   0.88801       599
           EmC    0.88774   0.92722   0.90705       742
           ILF    0.92168   0.94809   0.93470      3699
          IOFF    0.94049   0.97370   0.95681      2662
          MdLF    0.93006   0.95645   0.94307      3490
         SLF-I    0.91515   0.96261   0.93828      2969
        SLF-II    0.92296   0.96228   0.94221      3075
       SLF-III    0.92074   0.96532   0.94250      1384
            UF    0.92512   0.96451   0.94441      1409
           CST    0.91655   0.96322   0.93931      2121
          CR-F    0.91050   0.97079   0.93968      2054
          CR-P    0.89286   0.97087   0.93023       412
            SF    0.89530   0.92712   0.91093      3348
            SO    0.89493   0.91481   0.90476       270
            SP    0.89474   0.88650   0.89060       326
            TF    0.94351   0.97206   0.95757      4725
            TO    0.93784   0.95724   0.94744       725
            TT    0.93639   0.95998   0.94804      2024
            TP    0.88249   0.91215   0.89707      1696
          PLIC    0.87253   0.90639   0.88914       438
           CC1    0.84858   0.88487   0.86634       304
           CC2    0.92959   0.97238   0.95050      2824
           CC3    0.94191   0.96296   0.95232      1566
           CC4    0.94189   0.96972   0.95560      1354
           CC5    0.93522   0.95441   0.94472      1316
           CC6    0.91969   0.98136   0.94952      2789
           CC7    0.91992   0.95218   0.93577       941
           CPC    0.87893   0.95778   0.91667       379
           ICP    0.91616   0.96344   0.93920       465
Intra-CBLM-I-P    0.90156   0.94009   0.92042      2270
Intra-CBLM-PaT    0.93871   0.97799   0.95795      5998
           MCP    0.93518   0.97302   0.95372      1112
         Sup-F    0.93410   0.96930   0.95137     16187
        Sup-FP    0.91000   0.94035   0.92492      2129
         Sup-O    0.91341   0.93810   0.92559      1147
        Sup-OT    0.90274   0.94622   0.92397      1599
         Sup-P    0.92816   0.94924   0.93858      7486
        Sup-PO    0.90622   0.94459   0.92501      2220
        Sup-PT    0.93277   0.95747   0.94496      5666
         Sup-T    0.90977   0.95695   0.93276      3066
         Other    0.96911   0.92743   0.94781     95177

      accuracy                        0.94427    200000
     macro avg    0.91499   0.95114   0.93260    200000
  weighted avg    0.94525   0.94427   0.94435    200000

  Accuracy: 94.43 %, Macro F1 Score: 93.26 %
```

This test function can also been called from CLI.
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