"""
This is an example of training a Mixture-of-Experts (MoE) model on MULTIPLE CAMELS datasets.

It uses the MultiSourceDataset to read from various configured sources.
"""

import os
from hydrodataset.hydro_dataset import StandardVariable
from torchhydro import SETTING
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate
from hydrodataset.multi_dataset_reader import MultiDatasetReader, DATASET_MAPPING
import torch
import numpy as np


def main():

    DATASETS = ["camels_us", "camels_cl", "camels_aus", "camels_col"]

    print(f"Using datasets: {DATASETS}")

    # 读取所有站点 ID
    reader = MultiDatasetReader(datasets=DATASETS)
    id_mapping = reader.collect_all_ids(force_refresh=False)

    # 展平并去重
    all_gage_ids = []
    for ds_name in DATASETS:
        ids = id_mapping.get(ds_name, [])
        all_gage_ids.extend(ids)

    # 去重（保持确定性：先 sort 再 set）
    all_gage_ids = sorted(set(all_gage_ids))
    print(f"Total unique stations: {len(all_gage_ids)}")

    # ============================================================
    # 5 折交叉验证划分
    # ============================================================

    SEED = 42
    N_FOLDS = 5

    rng = np.random.default_rng(SEED)

    all_gage_ids = np.array(all_gage_ids)
    rng.shuffle(all_gage_ids)  # 固定随机打乱（可复现）

    folds = np.array_split(all_gage_ids, N_FOLDS)

    # folds[i] 即第 i 折的 validation id
    # 训练集 = 其余 folds 拼起来
    cv_splits = []

    for fold_idx in range(N_FOLDS):
        val_ids = folds[fold_idx].tolist()
        train_ids = np.concatenate(
            [folds[i] for i in range(N_FOLDS) if i != fold_idx]
        ).tolist()

        cv_splits.append({"fold": fold_idx, "train_ids": train_ids, "val_ids": val_ids})

        print(f"Fold {fold_idx}: " f"train={len(train_ids)}, val={len(val_ids)}")

    fold = 0
    train_ids = cv_splits[fold]["train_ids"]
    val_ids = cv_splits[fold]["val_ids"]
    print(f"Fold {fold}: " f"train={train_ids}, val={val_ids}")


if __name__ == "__main__":
    main()
