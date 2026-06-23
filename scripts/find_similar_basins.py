"""Find similar basins by cosine-similarity matrices.

Library API:
    get_similar_basins(target_id, method, k) -> list[str]
        Returns top-k similar basin ids (8-digit, leading-zero, sorted asc)
        for the given method ('method1' | 'method2' | 'method3').

CLI:
    python find_similar_basins.py                # interactive (legacy)
    python find_similar_basins.py --target 9484000 --method method1 --k 100
    python find_similar_basins.py --target 9484000 --method method1 --k 100 \\
        --out gage_id_cache/9484000/method1_k100.txt
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from path_config import get_paths  # noqa: E402

random.seed(888)
np.random.seed(888)

_DF_CACHE: dict[str, pd.DataFrame] = {}


def format_ids_with_leading_zeros(ids_series: pd.Series) -> pd.Series:
    return ids_series.astype(str).str.zfill(8)


def _format_id(basin_id) -> str:
    return str(basin_id).zfill(8)


def load_similarity_matrix(method: str) -> pd.DataFrame:
    """Load (and cache) the similarity matrix for the named method.

    method: 'method1' | 'method2' | 'method3'
    """
    if method in _DF_CACHE:
        return _DF_CACHE[method]
    paths = get_paths()
    matrices = paths.get("similarity_matrices") or {}
    csv_path = matrices.get(method)
    if not csv_path:
        raise KeyError(f"similarity_matrices.{method} missing in paths.yaml")
    df = pd.read_csv(csv_path, index_col=0)
    df.index = format_ids_with_leading_zeros(df.index)
    df.columns = format_ids_with_leading_zeros(df.columns)
    _DF_CACHE[method] = df
    return df


def find_top_basins(df: pd.DataFrame, basin_id, top_n: int) -> list[str]:
    """Top-N most similar basins (excluding self), sorted by similarity desc."""
    formatted = _format_id(basin_id)
    if formatted not in df.index:
        raise ValueError(f"basin id '{formatted}' not found in similarity matrix")
    sims = df.loc[formatted]
    sims = sims[sims.index != formatted]
    sims = sims.sort_values(ascending=False)
    return sims.index.tolist()[:top_n]


def get_similar_basins(target_id, method: str, k: int) -> list[str]:
    """Top-k similar basins, leading-zero ids, sorted ascending by id."""
    df = load_similarity_matrix(method)
    return sorted(find_top_basins(df, target_id, k))


def find_similar_basins_sorted(df: pd.DataFrame, basin_id, threshold: float = 0.5) -> list[str]:
    """Legacy: ids with similarity > threshold, sorted desc."""
    formatted = _format_id(basin_id)
    if formatted not in df.index:
        print(f"Error: Basin ID '{formatted}' not found in the matrix.")
        return []
    sims = df.loc[formatted]
    series = sims[(sims > threshold) & (sims.index != formatted)]
    return series.sort_values(ascending=False).index.tolist()


def get_all_basins(df: pd.DataFrame, basin_id) -> list[str]:
    formatted = _format_id(basin_id)
    if formatted not in df.index:
        print(f"Error: Basin ID '{formatted}' not found in the matrix.")
        return []
    all_basins = df.index.tolist()
    all_basins.remove(formatted)
    return all_basins


def remove_leading_zeros(id_list):
    return [str(int(i)) for i in id_list]


def _interactive_main():
    """Original interactive workflow: prompt for an id, print top-100 per method."""
    df_cosine = load_similarity_matrix("method1")
    df_method2 = load_similarity_matrix("method2")
    df_method3 = load_similarity_matrix("method3")

    input_basin_id = input("请输入一个流域ID: ")

    print("\n--- 从所有流域中随机抽取670个流域ID ---")
    all_basins_list = get_all_basins(df_method2, input_basin_id)
    total = len(all_basins_list)
    print(f"总共有 {total} 个流域ID。")
    if total >= 300:
        random_basins = sorted(random.sample(all_basins_list, 670))
        print("随机抽取的670个流域ID (列表格式):")
        print(random_basins)
    else:
        print(f"总流域数量 ({total}) 小于670，将打印所有流域ID:")
        print(all_basins_list)

    for label, df in [("方法1（余弦相似度）", df_cosine), ("方法2", df_method2), ("方法3", df_method3)]:
        print(f"\n--- {label}按相似度排序的前100个流域ID ---")
        try:
            top = find_top_basins(df, input_basin_id, 100)
        except ValueError as e:
            print(e)
            continue
        print(f"找到 {len(top)} 个流域ID:")
        print(top)


def _cli_main(argv=None):
    p = argparse.ArgumentParser(description="Find top-k similar basins for a target")
    p.add_argument("--target", help="target basin id (with or without leading zeros)")
    p.add_argument("--method", choices=["method1", "method2", "method3"])
    p.add_argument("--k", type=int)
    p.add_argument("--out", help="optional output file (one id per line)")
    p.add_argument("--interactive", action="store_true",
                   help="run the legacy interactive workflow")
    args = p.parse_args(argv)

    if args.interactive or not (args.target and args.method and args.k):
        _interactive_main()
        return
    ids = get_similar_basins(args.target, args.method, args.k)
    text = "\n".join(ids)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")
        print(f"wrote {len(ids)} ids -> {out}")
    else:
        print(text)


if __name__ == "__main__":
    _cli_main()
