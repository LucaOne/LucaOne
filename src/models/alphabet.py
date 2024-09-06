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
        self.all_special_tokens = prepend_toks + append_toks
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
    def from_pretrained(cls, dir_path):
        import os, pickle
        return pickle.load(open(os.path.join(dir_path, "alphabet.pkl"), "rb"))

    def save_pretrained(self, save_dir):
        import os, pickle
        with open(os.path.join(save_dir, "alphabet.pkl"), 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

    def _tokenize(self, text) -> str:
        return text.split()

    def tokenize(self, text, **kwargs) -> List[str]:
        def split_on_token(tok, text):
            result = []
            split_text = text.split(tok)
            for i, sub_text in enumerate(split_text):
                if i < len(split_text) - 1:
                    sub_text = sub_text.rstrip()
                if i > 0:
                    sub_text = sub_text.lstrip()

                if i == 0 and not sub_text:
                    result.append(tok)
                elif i == len(split_text) - 1:
                    if sub_text:
                        result.append(sub_text)
                    else:
                        pass
                else:
                    if sub_text:
                        result.append(sub_text)
                    result.append(tok)
            return result

        def split_on_tokens(tok_list, text):
            if not text.strip():
                return []
            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in self.unique_no_split_tokens:
                        tokenized_text.extend(split_on_token(tok, sub_text))
                    else:
                        tokenized_text.append(sub_text)
                text_list = tokenized_text

            return list(
                itertools.chain.from_iterable(
                    (
                        self._tokenize(token)
                        if token not in self.unique_no_split_tokens
                        else [token]
                        for token in tokenized_text
                    )
                )
            )

        no_split_token = self.unique_no_split_tokens
        tokenized_text = split_on_tokens(no_split_token, text)
        return tokenized_text

    def encode(self, text):
        return [self.tok_to_idx[tok] for tok in self.tokenize(text)]


if __name__ == "__main__":
    alphabet = Alphabet.from_predefined("gene_prot")
    from src.utils import gene_seq_replace
    print(alphabet.encode(gene_seq_replace("gttgtttggtagctaggagcctgactacatggcttcaaggctaaatggccacaggtgcccaggctatttggcttgctggaggcttcattcat")))

