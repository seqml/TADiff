# TADiff: The implementation codes of "What if Tomorrow is the World Cup Final? Counterfactual Time Series Forecasting with Textual Conditions" 📈
<div align="center">

[![project page](https://img.shields.io/badge/Project%20page-TADiff%20-lightblue)](https://seqml.github.io/VerbalTS/)&nbsp;
[![paper link](https://img.shields.io/badge/ICML-65690-b31b1b.svg)](https://icml.cc/virtual/2026/poster/65690)&nbsp;

</div>

<p align="center" style="font-size: larger;">
  <a href="https://icml.cc/virtual/2026/poster/65690">What if Tomorrow is the World Cup Final? Counterfactual Time Series Forecasting with Textual Conditions</a>
</p>

<div>
  <p align="center" style="font-size: larger;">
    <strong>ICML 2026</strong>
  </p>
</div>

<p align="center">
<img src="https://github.com/seqml/TADiff/blob/main/asset/task_formulation.png" width=95%>
<p>
<be>

## Contribution
### 1. Counterfacutal Forecasting Method
We propose TADiff, a counterfactual time series forecasting model with text-attribution mechanism. 
<p align="center">
<img src="https://github.com/seqml/TADiff/blob/main/asset/framework.png" width=95%>
<p>
To achieve balanced forecasting considering the combined effects of history and future, we propose a text-attribution mechanism that attributes historical sequences prior to forecasting, aiming to decouple the intrinsic features of the sequence from the extrinsic conditions. The overall process consists of a two-stage inference and a joint training with
two optimization objectives.

### 2. Counterfactual Forecasting Evaluation
We propose DTTC scores, the model-based evaluation metrics for counterfactual time sries forecasting, even when the ground truth time series is absent. The DTTC scores consist of DTTC-I and DTTC-E, measuring the consistency of forecasts with intrinsic historical featues and extrinsic future condtions, respectively.

### 3. Experimental Results
We compare our method, TADiff, with the baselines on both synthetic and real-world datasets. We consider both the factual forecasting and counterfactual forecasting. As shown in the table below, TADiff achieves both superior numerical accuracy and semantic consistency. Moreover, TADiff exhibits strong adaptability and generalization for forecasting under diverse future conditions.
<p align="center">
<img src="https://github.com/seqml/TADiff/blob/main/asset/main_exp.png" width=95%>
<p>

## Installation
### 1. Environment
```
torch==2.6.0
pandas==2.0.3
pyyaml==6.0.3
linear_attention_transformer==0.19.1
tensorboard==2.20.0
scikit-learn==1.7.2
```
You can use the following command to prepare your environment.
```
pip install -r requirements.txt
```
### 2. Dataset
Download the datasets from [Google Drive](https://drive.google.com/drive/folders/1N0zxkLdvpdjkwayKA2OZIJYP4nfzhOeF?usp=drive_link).
<details>
    <summary> Assume the datasets are in `/path/to/dataset/`. It should be like:</summary>
  
    /path/to/dataset/:
        Synth/:
            F/:
                train_ts.npy
                train_attrs.npy
                train_caps.npy
                ...
            CF/:
                train_ts.npy
                train_attrs.npy
                train_caps.npy
                ...
        ETTm1/:
            ...
   **NOTE: The arg `--data_folder=/path/to/dataset/` should be passed to the training script.**
</details>

### 3. Pretrained model checkpoints
Download the [LongCLIP](https://huggingface.co/zer0int/LongCLIP-GmP-ViT-L-14) from Huggingface, and put the model weights in `/path/to/save/`.

Download the checkpoints from [Google Drive](https://drive.google.com/drive/folders/17zQJlxj5j7eWr636vmYdw1sGqi-uW1i4?usp=drive_link).
<details>
    <summary> Assume the checkpoints are in `/path/to/save/`. It should be like:</summary>

    /path/to/save/:
        [dataset_name]_cttp:
            ...
        [dataset_name]_eval:
            [run_id]:
                ckpts:
                    model_best.pth
                train_configs.yaml
                eval_configs.yaml
                model_cond_configs.yaml
                model_diff_configs.yaml
            ...
        ...
    
  **NOTE: The arg `--save_folder=/path/to/save/` should be passed to the training script.**
</details>
   
## Training
### 1. Train scripts
To pretrain the model on the factual data and finetune the model on the counterfactual data.
```
bash scripts/{dataset_name}/pretrain.sh
bash scripts/{dataset_name}/finetune.sh
```
### 2. Results
After the training, check the results at the following path.
```
{save_folder}/{run_id}/results.csv
```
### 3. Evaluate with checkpoints
To evaluate the model with the checkpoints.
```
bash scripts/{dataset_name}/eval.sh
```
### 5. Device
All codes in this repository run on GPU by default. If you need to run on the CPU, please modify the device-related parameters in the config file.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation
If our work helps you in research, please give us a star or cite us using the following:
```
@article{gu2026tadiff,
  title={VerbalTS: Generating Time Series from Texts},
  author={Gu, Shuqi and Zhao, Yongxiang and Jing, Baoyu and Ren, Kan},
  journal={What if Tomorrow is the World Cup Final? Counterfactual Time Series Forecasting with Textual Conditions},
  year={2026}
}
```
