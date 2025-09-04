import torch
from torch.utils.data import IterableDataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import random
import pyarrow.parquet as pq
import datetime
import time


class FastDataset(IterableDataset):
    def __init__(self, folder_list):
        self.folder_list = folder_list
        self.union_feature_files = [
            ("feat1.parquet", [f'feat1_{j}' for j in range(10)]),
            ("feat2.parquet", [f'feat2_{j}' for j in range(10)]),
        ]
        self.separate_file_prefix = "large_data_part"
        self.separate_file_columns = [f'col_{k}' for k in range(100)]
        self.max_parts = 10
        self.dataset_len = 0
        for folder in self.folder_list:
            folder_path = Path(folder)
            parquet_file = pq.ParquetFile(folder_path / self.union_feature_files[0][0])
            total_rows = parquet_file.metadata.num_rows
            self.dataset_len += total_rows

    def reset(self):
        random.shuffle(self.folder_list)

    def __len__(self):
        return self.dataset_len

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id, num_workers = (worker_info.id, worker_info.num_workers) if worker_info else (0, 1)
        for folder in self.folder_list:
            folder_path = Path(folder)
            union_index = 0
            parquet_file = pq.ParquetFile(folder_path / self.union_feature_files[0][0])
            total_rows = parquet_file.metadata.num_rows
            if num_workers > 1:
                union_left = total_rows * worker_id // num_workers
                union_right = total_rows * (worker_id + 1) // num_workers
            else:
                union_left = 0
                union_right = total_rows

            for part in range(self.max_parts):
                separate_file = folder_path / f"{self.separate_file_prefix}{part}.parquet"
                if not separate_file.exists():
                    break
                parquet_file = pq.ParquetFile(separate_file)
                n_rows = parquet_file.metadata.num_rows
                union_index_new = union_index + n_rows
                if union_index_new <= union_left:
                    union_index = union_index_new
                    continue
                separate_df = parquet_file.read(columns=self.separate_file_columns).to_pandas()
                skipped_rows = max(0, union_left - union_index)
                omitted_rows = max(0, union_index_new - union_right)
                union_dfs = []
                
                for union_file, columns in self.union_feature_files:
                    union_file_path = folder_path / union_file
                    # print(union_file_path, columns)
                    df = pd.read_parquet(union_file_path, columns=columns)
                    union_dfs.append(df.iloc[union_index + skipped_rows:union_index_new - omitted_rows].reset_index(drop=True))
                union_df = pd.concat(union_dfs, axis=1)
                df = pd.concat([separate_df.iloc[skipped_rows:-omitted_rows].reset_index(drop=True), union_df], axis=1)
                union_index = union_index_new
                for _, row in df.iterrows():
                    yield torch.tensor(row.values, dtype=torch.float32)
                if union_index >= union_right:
                    union_index -= omitted_rows
                    break
            
            assert union_index == union_right, f"union_index: {union_index}, union_right: {union_right}"

if __name__ == "__main__":
    start_date = datetime.date(2025, 1, 1)
    end_date = datetime.date(2025, 1, 31)

    folder_list = []
    for i in range((end_date - start_date).days + 1):
        current_date = start_date + datetime.timedelta(days=i)
        folder = Path(f"example/data_{current_date}")
        folder_list.append(str(folder))

    dataset = FastDataset(folder_list)
    print(f"Dataset length: {len(dataset)}")
    # shuffling is achieved by shuffling folder_list
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=False, prefetch_factor=2, persistent_workers=False, multiprocessing_context="spawn")
    print(f"len(dataloader): {len(dataloader)} batches")

    for epoch in range(5):
        t1 = time.time()
        dataset.reset()
        t2 = time.time()
        print(f"Epoch {epoch}, reset time: {t2 - t1:.6f} seconds")
        for batch_idx, batch in enumerate(dataloader):
            pass
        t3 = time.time()
        print(f"Epoch {epoch}, data loading time: {t3 - t2:.6f} seconds")