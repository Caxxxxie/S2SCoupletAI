from __future__ import annotations

from typing import List
import argparse
from pathlib import Path
import logging

import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm

from module import Tokenizer
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CoupletExample:
    def __init__(self, seq: List[str], tag: List[str]):
        assert len(seq) == len(tag), f"Length mismatch: len(seq)={len(seq)} len(tag)={len(tag)}"
        self.seq = seq
        self.tag = tag


class CoupletFeatures:
    def __init__(self, src_ids: List[int], tgt_in_ids: List[int], tgt_out_ids: List[int]):
        self.src_ids = src_ids
        self.tgt_in_ids = tgt_in_ids
        self.tgt_out_ids = tgt_out_ids


def read_examples(fdir: Path) -> List[CoupletExample]:
    """
    Expect:
      fdir/in.txt  : each line is tokenized by whitespace, e.g. "马 齿 草 焉 无 马 齿"
      fdir/out.txt : same format
    """
    seqs: List[List[str]] = []
    tags: List[List[str]] = []

    in_path = fdir / "in.txt"
    out_path = fdir / "out.txt"
    if not in_path.exists() or not out_path.exists():
        raise FileNotFoundError(f"Missing in.txt/out.txt under: {fdir}")

    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            seqs.append(line.split())

    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tags.append(line.split())

    if len(seqs) != len(tags):
        raise ValueError(f"#lines mismatch: in.txt={len(seqs)} out.txt={len(tags)}")

    examples = [CoupletExample(seq, tag) for seq, tag in zip(seqs, tags)]
    return examples


def _require_special_tokens(tokenizer: Tokenizer) -> None:
    missing = []
    for name in ("pad_id", "unk_id", "bos_id", "eos_id"):
        if not hasattr(tokenizer, name):
            missing.append(name)
    if missing:
        raise AttributeError(
            "Tokenizer is missing required special token ids for seq2seq: "
            f"{missing}. Please update Tokenizer.build() to include [BOS]/[EOS] "
            "and add bos_id/eos_id properties."
        )


def convert_examples_to_features(
    examples: List[CoupletExample],
    tokenizer: Tokenizer,
    max_seq_len: int
) -> List[CoupletFeatures]:
    """
    Build seq2seq training pairs:
      src_ids: truncated to length L (<= max_seq_len-1 to keep tgt room)
      tgt_in : [BOS] + tgt_ids[:L]
      tgt_out: tgt_ids[:L] + [EOS]
    """
    _require_special_tokens(tokenizer)

    features: List[CoupletFeatures] = []
    for ex in tqdm(examples, desc="creating features"):
        # enforce equal length at raw token level
        assert len(ex.seq) == len(ex.tag)

        src = tokenizer.convert_tokens_to_ids(ex.seq)
        tgt = tokenizer.convert_tokens_to_ids(ex.tag)
        L = min(len(src), len(tgt), max_seq_len - 1)
        if L <= 0:
            # extremely short / empty lines
            continue

        src = src[:L]
        tgt = tgt[:L]

        tgt_in = [tokenizer.bos_id] + tgt         # length L+1
        tgt_out = tgt + [tokenizer.eos_id]        # length L+1

        # sanity
        assert len(src) == len(tgt)
        assert len(tgt_in) == len(tgt_out)
        assert len(tgt_in) <= max_seq_len

        features.append(CoupletFeatures(src, tgt_in, tgt_out))

    return features


def convert_features_to_tensors(
    features: List[CoupletFeatures],
    tokenizer: Tokenizer,
    max_seq_len: int
):
    """
    Returns:
      src_ids      (N,T)
      src_mask     (N,T)   bool, True=PAD
      src_len      (N,)
      tgt_in_ids   (N,T)
      tgt_out_ids  (N,T)
      tgt_mask     (N,T)   bool, True=PAD
      tgt_len      (N,)
    """
    _require_special_tokens(tokenizer)

    total = len(features)
    if total == 0:
        raise ValueError("No features created. Check your input files and tokenization.")

    src_ids = torch.full((total, max_seq_len), tokenizer.pad_id, dtype=torch.long)
    src_mask = torch.ones((total, max_seq_len), dtype=torch.bool)  # True=PAD
    src_len = torch.zeros((total,), dtype=torch.long)

    tgt_in_ids = torch.full((total, max_seq_len), tokenizer.pad_id, dtype=torch.long)
    tgt_out_ids = torch.full((total, max_seq_len), tokenizer.pad_id, dtype=torch.long)
    tgt_mask = torch.ones((total, max_seq_len), dtype=torch.bool)  # True=PAD
    tgt_len = torch.zeros((total,), dtype=torch.long)

    for i, f in enumerate(tqdm(features, desc="creating tensors")):
        # src
        Ls = min(len(f.src_ids), max_seq_len)
        src_ids[i, :Ls] = torch.tensor(f.src_ids[:Ls], dtype=torch.long)
        src_mask[i, :Ls] = 0
        src_len[i] = Ls

        # tgt
        Lt = min(len(f.tgt_in_ids), max_seq_len)
        assert len(f.tgt_in_ids) == len(f.tgt_out_ids)
        tgt_in_ids[i, :Lt] = torch.tensor(f.tgt_in_ids[:Lt], dtype=torch.long)
        tgt_out_ids[i, :Lt] = torch.tensor(f.tgt_out_ids[:Lt], dtype=torch.long)
        tgt_mask[i, :Lt] = 0
        tgt_len[i] = Lt

    return src_ids, src_mask, src_len, tgt_in_ids, tgt_out_ids, tgt_mask, tgt_len


def create_dataset(fdir: Path, tokenizer: Tokenizer, max_seq_len: int) -> TensorDataset:
    examples = read_examples(fdir)
    features = convert_examples_to_features(examples, tokenizer, max_seq_len)
    tensors = convert_features_to_tensors(features, tokenizer, max_seq_len)
    dataset = TensorDataset(*tensors)
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="couplet", type=str)
    parser.add_argument("--output", default="dataset", type=str)
    parser.add_argument("--max_seq_len", default=32, type=int)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    vocab_file = input_dir / "vocabs"

    logger.info("creating tokenizer...")
    tokenizer = Tokenizer()
    tokenizer.build(vocab_file)

    # hard check
    _require_special_tokens(tokenizer)

    logger.info("creating dataset...")
    train_dataset = create_dataset(input_dir / "train", tokenizer, args.max_seq_len)
    test_dataset = create_dataset(input_dir / "test", tokenizer, args.max_seq_len)

    logger.info("saving dataset...")
    tokenizer.save_pretrained(output_dir / "vocab.pkl")
    torch.save(train_dataset, output_dir / "train.pkl")
    torch.save(test_dataset, output_dir / "test.pkl")

    logger.info("done.")
    logger.info(
        "train.pkl sample fields: (src_ids, src_mask, src_len, tgt_in_ids, tgt_out_ids, tgt_mask, tgt_len)"
    )


if __name__ == "__main__":
    main()