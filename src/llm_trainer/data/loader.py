import os
from typing import Tuple, Optional, Any

try:
    from datasets import load_dataset, Dataset, DatasetDict
except ImportError:
    load_dataset = None
    Dataset = Any
    DatasetDict = Any

from ..config import AppConfig

def load_raw_dataset(config: AppConfig) -> Tuple[Any, Optional[Any]]:
    if load_dataset is None:
        raise ImportError("HuggingFace 'datasets' library is required to load raw datasets. Please run `pip install datasets`.")

    ds_path = config.dataset.dataset_name_or_path
    train_split = config.dataset.train_split
    eval_split = config.dataset.eval_split
    streaming = config.dataset.streaming

    # Check if local file path
    if os.path.exists(ds_path):
        ext = os.path.splitext(ds_path)[-1].lower().strip(".")
        if ext in ("json", "jsonl"):
            ds_type = "json"
        elif ext == "csv":
            ds_type = "csv"
        elif ext in ("parquet", "pq"):
            ds_type = "parquet"
        else:
            ds_type = "text"

        full_ds = load_dataset(ds_type, data_files=ds_path, streaming=streaming)
        if isinstance(full_ds, DatasetDict):
            train_ds = full_ds["train"]
            eval_ds = None
        else:
            train_ds = full_ds
            eval_ds = None
    else:
        # HuggingFace Hub Dataset
        kwargs = {}
        if config.dataset.dataset_config_name:
            kwargs["name"] = config.dataset.dataset_config_name

        train_ds = load_dataset(
            ds_path,
            split=train_split,
            streaming=streaming,
            **kwargs
        )
        
        eval_ds = None
        if eval_split:
            try:
                eval_ds = load_dataset(
                    ds_path,
                    split=eval_split,
                    streaming=streaming,
                    **kwargs
                )
            except Exception:
                eval_ds = None

    return train_ds, eval_ds
