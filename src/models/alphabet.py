#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/7/24 11:00
@project: LucaOne
@file: alphabet
@desc: alphabet for LucaOne
'''
import sys
import itertools
from typing import Sequence, List
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../src")
try:
    from utils import gene_seq_replace
except ImportError:
    from src.utils import gene_seq_replace

ATCGU = {"A", "T", "C", "G", "U"}

gene_standard_toks = ['1', '2', '3', '4', '5', '.', '-', '*']

prot_standard_toks = ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O', 'J', '.', '-', '*']

gene_prot_standard_toks = ['1', '2', '3', '4', '5', 'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O', 'J', '.', '-', '*']

gene_prot_prepend_toks = ['[PAD]', '[UNK]']

gene_prot_append_toks = ['[CLS]', '[SEP]', '[MASK]']


class Alphabet(object):
    def __init__(
            self,
            standard_toks: Sequence[str],
            prepend_toks: Sequence[str] = gene_prot_prepend_toks,
            append_toks: Sequence[str] = gene_prot_append_toks,
            prepend_bos: bool = True,
            append_eos: bool = True
    ):
        self.standard_toks = list(standard_toks)
        self.prepend_toks = list(prepend_toks)
        self.append_toks = list(append_toks)
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos

        self.all_toks = list(self.prepend_toks)
        self.all_toks.extend(self.append_toks)
        self.all_toks.extend(self.standard_toks)

        self.tok_to_idx = {tok: i for i, tok in enumerate(self.all_toks)}

        self.unk_idx = self.tok_to_idx["[UNK]"]
        self.padding_idx = self.get_idx("[PAD]")
        self.pad_token_id = self.padding_idx
        self.cls_idx = self.get_idx("[CLS]")
        self.mask_idx = self.get_idx("[MASK]")
        self.eos_idx = self.get_idx("[SEP]")
        self.all_special_tokens = self.prepend_toks + self.append_toks
        self.all_special_token_idx_list = [self.tok_to_idx[v] for v in self.all_special_tokens]
        self.unique_no_split_tokens = self.all_toks
        self.vocab_size = self.__len__()

    def __len__(self):
        return len(self.all_toks)

    def get_idx(self, tok):
        return self.tok_to_idx.get(tok, self.unk_idx)

    def get_tok(self, ind):
        return self.all_toks[ind]

    def to_dict(self):
        return self.tok_to_idx.copy()

    @classmethod
    def from_predefined(cls, name: str):
        if name.lower() == "prot":
            standard_toks = prot_standard_toks
        elif name.lower() == "gene":
            standard_toks = gene_standard_toks
        elif name.lower() in ["gene_prot", "prot_gene"]:
            standard_toks = gene_prot_standard_toks
        else:
            raise Exception("Not support tokenizer name: %s" % name)

        prepend_toks = gene_prot_prepend_toks
        append_toks = gene_prot_append_toks
        prepend_bos = True
        append_eos = True

        return cls(standard_toks, prepend_toks, append_toks, prepend_bos, append_eos)

    @classmethod
    def from_pretrained(cls, dir_path: str) -> object:
        try:
            import os, pickle
            return pickle.load(open(os.path.join(dir_path, "alphabet.pkl"), "rb"))
        except Exception:
            return cls.from_predefined("gene_prot")

    def save_pretrained(self, save_dir):
        import os, pickle, json
        with open(os.path.join(save_dir, "alphabet.pkl"), 'wb') as wfp:
            pickle.dump(self, wfp, pickle.HIGHEST_PROTOCOL)
        all_tokens_json = {
            "vocab_size": self.vocab_size,
            "prepend": self.prepend_toks,
            "standard": self.standard_toks,
            "append": self.append_toks,
            "prepend_bos": self.prepend_bos,
            "append_eos": self.append_eos,
        }
        with open(os.path.join(save_dir, "all_tokens.json"), "w") as wfp:
            json.dump(all_tokens_json, wfp, ensure_ascii=False)

    def _tokenize(self, seq) -> List[str]:
        return seq.split()

    def tokenize(self, seq, **kwargs) -> List[str]:
        def split_on_token(tok, seq):
            result = []
            split_seq = seq.split(tok)
            for i, sub_seq in enumerate(split_seq):
                if i < len(split_seq) - 1:
                    sub_seq = sub_seq.rstrip()
                if i > 0:
                    sub_seq = sub_seq.lstrip()

                if i == 0 and not sub_seq:
                    result.append(tok)
                elif i == len(split_seq) - 1:
                    if sub_seq:
                        result.append(sub_seq)
                    else:
                        pass
                else:
                    if sub_seq:
                        result.append(sub_seq)
                    result.append(tok)
            return result

        def split_on_tokens(tok_list, seq):
            if not seq.strip():
                return []
            tokenized_seq = []
            seq_list = [seq]
            for tok in tok_list:
                tokenized_seq = []
                for sub_seq in seq_list:
                    if sub_seq not in self.unique_no_split_tokens:
                        tokenized_seq.extend(split_on_token(tok, sub_seq))
                    else:
                        tokenized_seq.append(sub_seq)
                seq_list = tokenized_seq

            return list(
                itertools.chain.from_iterable(
                    (
                        self._tokenize(token)
                        if token not in self.unique_no_split_tokens
                        else [token]
                        for token in tokenized_seq
                    )
                )
            )

        no_split_token = self.unique_no_split_tokens
        tokenized_seq = split_on_tokens(no_split_token, seq)
        return tokenized_seq

    '''
    def encode(self, seq):
        return [self.tok_to_idx[tok] for tok in self.tokenize(seq)]
    '''

    def encode(self, seq_type, seq):
        if seq_type in ["gene", "dna", "rna", "nucleic_acid", "nucleotide"]:
            if len(ATCGU & set(list(seq.upper()))) > 0:
                seq = gene_seq_replace(seq)
        return [self.tok_to_idx[tok] for tok in self.tokenize(seq)]

    '''
    def encode_for_eval_mask(self, seq):
        return [self.tok_to_idx[tok] if tok != '-' else self.tok_to_idx["[MASK]"] for tok in self.tokenize(seq)]
    '''

    def encode_for_eval_mask(self, seq_type, seq):
        if seq_type in ["gene", "dna", "rna", "nucleic_acid", "nucleotide"]:
            if len(ATCGU & set(list(seq.upper()))) > 0:
                seq = gene_seq_replace(seq)
        return [self.tok_to_idx[tok] if tok != '-' else self.tok_to_idx["[MASK]"] for tok in self.tokenize(seq)]


if __name__ == "__main__":
    import sys
    sys.path.append("./")
    sys.path.append("../")
    sys.path.append("../../")
    sys.path.append("../../src")
    from src.utils import gene_seq_replace
    alphabet = Alphabet.from_predefined("gene_prot")
    seq = "gttgtttggtagctaggagcctgactacatggcttcaaggctaaatggccacaggtgcccaggctatttggcttgctggaggcttcattcat"
    seq = gene_seq_replace(seq)
    toks = alphabet.tokenize(seq)
    print(toks)
    print(len(toks))
    input_ids = alphabet.encode(seq_type="gene", seq=seq)
    print(input_ids)
    print(len(input_ids))

