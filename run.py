import os
import yaml
import json
import datetime
import argparse
import pandas as pd
import torch
import numpy as np
import random

from data import MixDataset
from models.conditional_forecaster import ConditionalForecaster
from train.trainer import Trainer
from evaluation.base_evaluator import BaseEvaluator

def save_configs(configs, path):
    print(json.dumps(configs, indent=4))
    with open(path, "w") as f:
        yaml.dump(configs, f, yaml.SafeDumper)

def train(train_configs, model_diff_configs, model_cond_configs, eval_configs,  output_folder):
    train_configs["train"]["output_folder"] = output_folder

    dataset = MixDataset(train_configs["data"])
    model = ConditionalForecaster(model_diff_configs, model_cond_configs)

    print("\n***** Train Configs *****")
    path = os.path.join(output_folder, "train_configs.yaml")
    save_configs(train_configs, path)

    print("\n***** Model Configs *****")
    path = os.path.join(output_folder, "model_diff_configs.yaml")
    save_configs(model_diff_configs, path)
    path = os.path.join(output_folder, "model_cond_configs.yaml")
    save_configs(model_cond_configs, path)

    pretrainer = Trainer(train_configs["train"], eval_configs, dataset, model)
    print("Begin training!")
    pretrainer.train()


def evaluate(eval_configs, model_diff_configs, model_cond_configs, output_folder):
    eval_configs["eval"]["model_path"] = os.path.join(output_folder, "ckpts/model_best_loss.pth")

    dataset = MixDataset(eval_configs["data"])
    model = ConditionalForecaster(model_diff_configs, model_cond_configs)

    print("\n***** Evaluate Configs *****")
    path = os.path.join(output_folder, "eval_configs.yaml")
    save_configs(eval_configs, path=path)

    evaluator = BaseEvaluator(eval_configs["eval"], dataset, model)

    df = _evaluate_(evaluator)
    return df


def _evaluate_(evaluator, sampler="ddim", n_samples=1):
    evaluator.n_samples = n_samples
    res_dict = evaluator.evaluate(split="test", sampler=sampler, ret_res=True)

    info = {
        "sampler": sampler,
        "n_sample": evaluator.n_samples,
    }
    info.update(res_dict["df"])    
    df = pd.DataFrame([info])
    df["steps"].astype(int)
    return df

def run(train_configs, eval_configs, model_diff_configs, model_cond_configs, output_folder, data_folder="", only_evaluate=False):
    if only_evaluate == False:
        train(train_configs, model_diff_configs, model_cond_configs, eval_configs, output_folder)

    df = evaluate(eval_configs, model_diff_configs, model_cond_configs, output_folder, only_evaluate)
    path = os.path.join(output_folder, "results.csv")
    df.to_csv(path)
    return df

parser = argparse.ArgumentParser(description="TADiff")
parser.add_argument("--model_diff_config_path", type=str, default="")
parser.add_argument("--model_cond_config_path", type=str, default="")
parser.add_argument("--forecaster_pretrain_path", type=str, default="")
parser.add_argument("--train_config_path", type=str, default="")
parser.add_argument("--evaluate_config_path", type=str, default="")
parser.add_argument("--data_folder", type=str, default="./datasets")
parser.add_argument("--save_folder", type=str, default="./save")
parser.add_argument("--dttc_folder", type=str, default="")
parser.add_argument("--start_runid", type=int, default=0)
parser.add_argument("--n_runs", type=int, default=3)
parser.add_argument("--only_evaluate", type=bool, default=False)

parser.add_argument("--hist_len", type=int, default=128)
parser.add_argument("--pred_len", type=int, default=128)
parser.add_argument("--num_layers", type=int, default=3)
parser.add_argument("--channels", type=int, default=64)
parser.add_argument("--patch_size", type=int, default=4)

parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--epochs", type=int, default=200)

parser.add_argument("--text_type", type=str, default="none")
parser.add_argument("--cond_modal", type=str, default="none")
parser.add_argument("--task_type", type=str, default="CF")
parser.add_argument("--f_loss_weight", type=float, default=1.0)
parser.add_argument("--a_loss_weight", type=float, default=1.0)

args = parser.parse_args()

save_folder = args.save_folder
os.makedirs(save_folder, exist_ok=True)
print("All files will be saved to '{}'".format(save_folder))

train_configs = yaml.safe_load(open(args.train_config_path))
eval_configs = yaml.safe_load(open(args.evaluate_config_path))
model_diff_configs = yaml.safe_load(open(args.model_diff_config_path))
model_cond_configs = yaml.safe_load(open(args.model_cond_config_path))
model_cond_configs["cond_modal"] = args.cond_modal

train_configs["train"]["lr"] = args.lr
train_configs["train"]["epochs"] = args.epochs
train_configs["train"]["batch_size"] = args.batch_size
train_configs["train"]["task_type"] = args.task_type
train_configs["train"]["f_loss_weight"] = args.f_loss_weight
train_configs["train"]["a_loss_weight"] = args.a_loss_weight

eval_configs["eval"]["batch_size"] = args.batch_size
eval_configs["eval"]["task_type"] = args.task_type

train_configs["data"]["folder"] = args.data_folder
train_configs["data"]["text_type"] = args.text_type
train_configs["data"]["hist_len"] = args.hist_len
train_configs["data"]["pred_len"] = args.pred_len
eval_configs["data"]["folder"] = args.data_folder
eval_configs["data"]["text_type"] = args.text_type
eval_configs["data"]["hist_len"] = args.hist_len
eval_configs["data"]["pred_len"] = args.pred_len

model_diff_configs["diffusion"]["num_layers"] = args.num_layers
model_diff_configs["diffusion"]["channels"] = args.channels
model_diff_configs["diffusion"]["patch_size"] = args.patch_size

if args.dttc_folder != "":
    eval_configs["eval"]["dttc_model_path"] = fr"{args.dttc_folder}/dttc_model_best.pth"
    eval_configs["eval"]["dttc_config_path"] = fr"{args.dttc_folder}/model_configs.yaml"
    

seed_list = [1, 7, 42]
df_list = []
for n in range(args.start_runid, args.n_runs):
    fix_seed = seed_list[n]
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    print(f"\nRun: {n}")
    output_folder = os.path.join(save_folder, str(n))
    os.makedirs(output_folder, exist_ok=True)
    eval_configs["eval"]["model_path"] = ""
    if args.forecaster_pretrain_path != "":
        model_diff_configs["forecaster_pretrain_path"] = f"{args.forecaster_pretrain_path}/{n}/ckpts/model_best_loss.pth"
    else:
        model_diff_configs["forecaster_pretrain_path"] = ""
    df = run(train_configs, eval_configs, model_diff_configs, model_cond_configs, output_folder, data_folder=args.data_folder, only_evaluate=args.only_evaluate)

    n_records = df.shape[0]
    df.insert(0, column="run", value=[n]*n_records)
    df_list.append(df)

df = pd.concat(df_list, ignore_index=True)
path = os.path.join(save_folder, "results.csv")
df.to_csv(path)