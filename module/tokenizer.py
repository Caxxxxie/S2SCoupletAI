from typing import List, Union
from pathlib import Path
import torch


class Tokenizer(object):
    def __init__(self):
        self.token_to_ix = {}
        self.ix_to_token = {}

    # ---- special token ids ----
    @property
    def pad_id(self):
        return self.token_to_ix['[PAD]']

    @property
    def unk_id(self):
        return self.token_to_ix['[UNK]']

    @property
    def bos_id(self):
        return self.token_to_ix['[BOS]']

    @property
    def eos_id(self):
        return self.token_to_ix['[EOS]']

    @property
    def vocab_size(self):
        return len(self.token_to_ix)

    # ---- build / load ----
    def build(self, vocab_file: Union[str, Path]):
        """
        Build tokenizer from a vocab file.
        The vocab file should contain ONE token per line.
        """
        if isinstance(vocab_file, str):
            vocab_file = Path(vocab_file)

        # fixed order for special tokens (VERY IMPORTANT)
        token_to_ix = {
            '[PAD]': 0,
            '[UNK]': 1,
            '[BOS]': 2,
            '[EOS]': 3,
        }

        with vocab_file.open('r', encoding='utf-8') as f:
            for line in f:
                token = line.rstrip('\n')
                if token and token not in token_to_ix:
                    token_to_ix[token] = len(token_to_ix)

        self.token_to_ix = token_to_ix
        self.ix_to_token = {v: k for k, v in token_to_ix.items()}

    def save_pretrained(self, filename: Union[str, Path]):
        if isinstance(filename, str):
            filename = Path(filename)
        filename.parent.mkdir(exist_ok=True, parents=True)

        torch.save(
            {
                'token_to_ix': self.token_to_ix,
                'ix_to_token': self.ix_to_token,
            },
            filename
        )

    @classmethod
    def from_pretrained(cls, filename: Union[str, Path]):
        info = torch.load(filename)
        tok = cls()
        tok.token_to_ix = info['token_to_ix']
        tok.ix_to_token = info['ix_to_token']
        return tok

    # ---- encode / decode ----
    def convert_token_to_id(self, token: str) -> int:
        return self.token_to_ix.get(token, self.unk_id)

    def convert_id_to_token(self, idx: int) -> str:
        return self.ix_to_token[idx]

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.convert_token_to_id(t) for t in tokens]

    def convert_ids_to_tokens(self, ids: List[int], ignore_pad: bool = False) -> List[str]:
        tokens = []
        for i in ids:
            if ignore_pad and i == self.pad_id:
                continue
            tokens.append(self.ix_to_token[i])
        return tokens

    def encode(self, sent: str) -> List[int]:
        # character-level encoding (unchanged behavior)
        return self.convert_tokens_to_ids(list(sent))

    def decode(self, ids: List[int], stop_at_eos: bool = True) -> str:
        tokens = []
        for i in ids:
            if i == self.pad_id:
                continue
            if stop_at_eos and i == self.eos_id:
                break
            if i == self.bos_id:
                continue
            tokens.append(self.ix_to_token[i])
        return "".join(tokens)