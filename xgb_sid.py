import numpy as np 
import pandas as pd 
import xgboost as xgb
from pathlib import Path
import datetime
import os
import shutil
import pyarrow.parquet as pq


###############################################
# Get all sid and their number of rows
def get_all_sids(folder_list: list[Path], save_path: Path):
    meta_file_name = "meta.parquet"   # TODO
    sid_count = {}
    for folder in folder_list:
        df = pd.read_parquet(folder / meta_file_name)
        sids, counts = np.unique(df['sid'], return_counts=True)
        for sid, count in zip(sids, counts):
            if sid in sid_count:
                sid_count[sid] += count
            else:
                sid_count[sid] = count
    df = pd.DataFrame(list(sid_count.items()), columns=['sid', 'count'])
    df.to_parquet(save_path, index=False)


def collect_all_sids(sid_path: Path, result_path: Path):
    files = list(sid_path.glob("*.parquet"))
    dfs = [pd.read_parquet(f) for f in files]
    df_all = pd.concat(dfs, ignore_index=True)
    df_all = df_all.groupby('sid', as_index=False).sum()
    df_all.to_parquet(result_path, index=False)

###############################################
# Collect all data for sid
def get_df_sid_core(folder_list: list[Path], sid: int):
    meta_file_name = "meta.parquet"   # TODO
    union_feat_file_names = [
        "feat1.parquet",
        "feat2.parquet",
        "target.parquet",
    ]   # TODO
    max_parts = 10   # TODO
    separate_feat_prefix = "large_data_part"   # TODO

    dfs = []
    for folder in folder_list:
        df = pd.read_parquet(folder / meta_file_name)
        idx = df.index[df['sid'] == sid].tolist()
        idx.sort()
        if len(idx) == 0:
            continue
        dfs_this = [df.loc[idx]] + [pd.read_parquet(folder / fname).loc[idx] for fname in union_feat_file_names]
        df_all = pd.concat(dfs_this, axis=1).reset_index(drop=True)
        right = 0
        part_i = 0
        rows = []
        pq_file = pq.ParquetFile(folder / f"{separate_feat_prefix}0.parquet")
        col_names = pq_file.schema.names
        for i in idx:
            while right <= i:
                left = right
                pq_file = pq.ParquetFile(folder / f"{separate_feat_prefix}{part_i}.parquet")
                right += pq_file.metadata.num_rows
                part_i += 1
            df_pq = pq_file.read().to_pandas()
            rows.append(df_pq.loc[i - left].values)
        df_separate = pd.DataFrame(rows, columns=col_names)
        df_all = pd.concat([df_all, df_separate], axis=1)
        dfs.append(df_all)

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        return None
    

def get_df_sid(folder_list: list[Path], sid: int, save_path: Path):
    """
    From folder_list, get all rows with sid, and save the resuling dataframe to save_path
    """
    df = get_df_sid_core(folder_list, sid)
    if df is None:
        return
    df.to_parquet(save_path, index=False)


###############################################
# Train xgboost model
def xgb_train_core(df: pd.DataFrame, features: list[str], target: str):
    dtrain = xgb.DMatrix(df[features], label=df[target])
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'tree_method': 'hist',
        'learning_rate': 0.1,
        'max_depth': 6,
        'seed': 42,
    }   # TODO
    bst = xgb.train(params, dtrain, num_boost_round=5) # TODO
    return bst
    

def xgb_train(data_folder: Path, sid: int, features: list[str], target: str, model_folder: Path, output_folder: Path):
    """
    Use all data files in data_folder, train xgboost 
    Assumes that all files in data_folder only contain sid
    """
    required_cols = ['target', 'baseline']   # TODO
    required_cols = [target]

    data_files = list(data_folder.glob(f"*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in data_files], ignore_index=True)
    shutil.rmtree(data_folder)
    bst = xgb_train_core(df, features, target)
    save_file = model_folder / f"xgb_model_sid_{sid}.json"
    bst.save_model(save_file)
    preds = bst.predict(xgb.DMatrix(df[features]))
    df1 = df[required_cols].copy()
    df1['pred'] = preds
    df1.to_parquet(output_folder / f"train_pred_sid_{sid}.parquet", index=False)
    r2 = 1 - np.sum((df1[target] - df1['pred']) ** 2) / np.sum(df1[target] ** 2)
    print(f"[Train] sid = {sid}: R^2 on training data: {r2:.4f}")


###############################################
# Fill predictions
def xgb_fill_pred(data_folder: Path, sid: int, features: list[str], target: str, model_folder: Path, output_folder: Path):
    required_cols = [target, 'baseline']   # TODO
    required_cols = [target]

    data_files = list(data_folder.glob(f"*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in data_files], ignore_index=True)
    shutil.rmtree(data_folder)
    model_path = model_folder / f"xgb_model_sid_{sid}.json"
    bst = xgb.Booster()
    bst.load_model(model_path)
    preds = bst.predict(xgb.DMatrix(df[features]))
    df1 = df[required_cols].copy()
    df1['pred'] = preds
    df1.to_parquet(output_folder / f"pred_sid_{sid}.parquet", index=False)
    r2 = 1 - np.sum((df1[target] - df1['pred']) ** 2) / np.sum(df1[target] ** 2)   # TODO
    print(f"[Eval] sid = {sid}: R^2 on prediction data: {r2:.4f}")


if __name__ == "__main__":
    date_ranges = [
        (datetime.date(2025, 1, 1), datetime.date(2025, 1, 31)),
        (datetime.date(2025, 2, 1), datetime.date(2025, 2, 28)),
        (datetime.date(2025, 3, 1), datetime.date(2025, 3, 31)),
    ]
    sid_save_folder = Path("temp/sid")
    all_sid_path = Path("temp/all_sids.parquet")
    data_save_folder = Path("temp/data")
    model_save_folder = Path("temp/model")
    train_pred_folder = Path("temp/train_pred")
    pred_folder = Path("temp/pred")
    sid_save_folder.mkdir(parents=True, exist_ok=True)
    data_save_folder.mkdir(parents=True, exist_ok=True)
    model_save_folder.mkdir(parents=True, exist_ok=True)
    train_pred_folder.mkdir(parents=True, exist_ok=True)
    pred_folder.mkdir(parents=True, exist_ok=True)


    # Program 1: Get all sids
    print('Program 1')
    for data_range in date_ranges:
        start_date, end_date = data_range
        folder_list = [Path(f"example/data_{date}") for date in pd.date_range(start_date, end_date).strftime('%Y-%m-%d')]
        sid_save_path = sid_save_folder / f"sid_{start_date}_{end_date}.parquet"
        get_all_sids(folder_list, sid_save_path)
    collect_all_sids(sid_save_folder, all_sid_path)


    # Program 2: Train XGBoost model for each sid
    print('Program 2')
    df_all_sid = pd.read_parquet(all_sid_path)
    all_sids = df_all_sid['sid'].unique().tolist()
    all_sids = all_sids[:3]  # TODO: For testing, only use first 3 sids

    feature_cols = [f'feat1_{i}' for i in range(10)] + [f'feat2_{i}' for i in range(10)] + [f'col_{i}' for i in range(100)]
    target_col = 'target'

    for sid in all_sids:
        sid_data_save_folder = data_save_folder / f"sid_{sid}"
        sid_data_save_folder.mkdir(parents=True, exist_ok=True)
        for date_range in date_ranges:
            start_date, end_date = date_range
            files = [Path(f"example/data_{date}") for date in pd.date_range(start_date, end_date).strftime('%Y-%m-%d')]
            data_save_path = sid_data_save_folder / f"{start_date}_{end_date}.parquet"
            get_df_sid(files, sid, data_save_path)

        xgb_train(sid_data_save_folder, sid, feature_cols, target_col, model_save_folder, train_pred_folder)
        print("Training completed.")


    # Program 3: Fill predictions
    print('Program 3')
    for sid in all_sids:
        sid_data_save_folder = data_save_folder / f"test_sid_{sid}"
        sid_data_save_folder.mkdir(parents=True, exist_ok=True)
        for date_range in date_ranges:
            start_date, end_date = date_range
            files = [Path(f"example/data_{date}") for date in pd.date_range(start_date, end_date).strftime('%Y-%m-%d')]
            data_save_path = sid_data_save_folder / f"{start_date}_{end_date}.parquet"
            get_df_sid(files, sid, data_save_path)

        xgb_fill_pred(sid_data_save_folder, sid, feature_cols, target_col, model_save_folder, pred_folder)
        
