from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
import itertools

try:
    import torch
except ImportError:
    torch = None

def apply_prompt_template(example: Dict[str, Any], template_style: Optional[str] = None, text_field: str = "text", prompt_field: Optional[str] = None, response_field: Optional[str] = None) -> str:
    if prompt_field and response_field and prompt_field in example and response_field in example:
        instruction = example[prompt_field]
        response = example[response_field]
        if template_style == "alpaca":
            return f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
        elif template_style == "chatml":
            return f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
        elif template_style == "llama3":
            return f"<|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{response}<|eot_id|>"
        else:
            return f"User: {instruction}\nAssistant: {response}"
    
    if text_field in example:
        return str(example[text_field])
    
    # Fallback to first string key
    for k, v in example.items():
        if isinstance(v, str):
            return v
    return ""

def format_and_tokenize_dataset(dataset, tokenizer, config):
    max_len = config.dataset.max_seq_length
    text_field = config.dataset.text_field
    prompt_field = config.dataset.prompt_field
    response_field = config.dataset.response_field
    template_style = config.dataset.chat_template

    def tokenize_fn(example):
        formatted_text = apply_prompt_template(
            example,
            template_style=template_style,
            text_field=text_field,
            prompt_field=prompt_field,
            response_field=response_field
        )
        tokenized = tokenizer(
            formatted_text,
            truncation=True,
            max_length=max_len,
            padding=False,
            return_tensors=None,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    return dataset.map(tokenize_fn, remove_columns=dataset.column_names if hasattr(dataset, "column_names") else None)

@dataclass
class DataCollatorForUniversalSFT:
    tokenizer: Any
    pad_to_multiple_of: Optional[int] = 8

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if torch is None:
            raise ImportError("PyTorch (torch) is required to run DataCollatorForUniversalSFT.")

        batch_input_ids = [f["input_ids"] for f in features]
        batch_attention_mask = [f.get("attention_mask", [1] * len(f["input_ids"])) for f in features]
        batch_labels = [f.get("labels", f["input_ids"]) for f in features]

        max_len = max(len(ids) for ids in batch_input_ids)
        if self.pad_to_multiple_of and max_len % self.pad_to_multiple_of != 0:
            max_len = ((max_len // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of

        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []

        for ids, mask, labels in zip(batch_input_ids, batch_attention_mask, batch_labels):
            pad_len = max_len - len(ids)
            padded_input_ids.append(ids + [pad_id] * pad_len)
            padded_attention_mask.append(mask + [0] * pad_len)
            padded_labels.append(labels + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Sequence Packing
# ---------------------------------------------------------------------------

def pack_dataset(
    tokenized_dataset,
    max_seq_length: int,
    eos_token_id: int,
) -> List[Dict[str, List[int]]]:
    """
    Greedy bin-packing: concatenates tokenized sequences up to max_seq_length
    to eliminate padding waste, giving ~2-3x throughput improvement on
    short-sample datasets.

    Returns a list of packed samples, each with:
    - input_ids  : concatenated token IDs (length == max_seq_length)
    - labels     : same as input_ids but -100 at cross-sequence boundaries
    - position_ids: per-token position within its source sequence
    - seq_lens   : list of original sequence lengths (informational)
    """
    packs: List[Dict[str, List[int]]] = []
    current_ids: List[int] = []
    current_labels: List[int] = []
    current_pos: List[int] = []
    current_lens: List[int] = []

    for sample in tokenized_dataset:
        ids: List[int] = sample["input_ids"]
        lbl: List[int] = sample.get("labels", ids.copy())

        # If single sample already exceeds max_seq_length, truncate it
        if len(ids) > max_seq_length:
            ids = ids[:max_seq_length]
            lbl = lbl[:max_seq_length]

        seq_len = len(ids)

        if len(current_ids) + seq_len > max_seq_length:
            # Flush current pack padded to max_seq_length
            pad_len = max_seq_length - len(current_ids)
            eos = eos_token_id if eos_token_id is not None else 0
            current_ids  += [eos] * pad_len
            current_labels += [-100] * pad_len
            current_pos  += list(range(pad_len))
            packs.append({
                "input_ids":   current_ids,
                "labels":      current_labels,
                "position_ids": current_pos,
                "seq_lens":    current_lens,
            })
            current_ids, current_labels, current_pos, current_lens = [], [], [], []

        pos_offset = list(range(seq_len))
        # Mark first token of each new sequence in labels as -100 (boundary)
        boundary_labels = lbl.copy()
        if current_ids:  # not the first sequence in this pack
            boundary_labels[0] = -100

        current_ids   += ids
        current_labels += boundary_labels
        current_pos   += pos_offset
        current_lens.append(seq_len)

    # Flush final partial pack
    if current_ids:
        pad_len = max_seq_length - len(current_ids)
        eos = eos_token_id if eos_token_id is not None else 0
        current_ids  += [eos] * pad_len
        current_labels += [-100] * pad_len
        current_pos  += list(range(pad_len))
        packs.append({
            "input_ids":   current_ids,
            "labels":      current_labels,
            "position_ids": current_pos,
            "seq_lens":    current_lens,
        })

    return packs


@dataclass
class PackedDataCollator:
    """Collates pre-packed samples (output of pack_dataset) into padded batches."""
    tokenizer: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if torch is None:
            raise ImportError("PyTorch (torch) is required to run PackedDataCollator.")

        input_ids   = torch.tensor([f["input_ids"]    for f in features], dtype=torch.long)
        labels      = torch.tensor([f["labels"]       for f in features], dtype=torch.long)
        position_ids= torch.tensor([f["position_ids"] for f in features], dtype=torch.long)
        attention_mask = (input_ids != (self.tokenizer.pad_token_id or 0)).long()

        return {
            "input_ids":    input_ids,
            "labels":       labels,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }
