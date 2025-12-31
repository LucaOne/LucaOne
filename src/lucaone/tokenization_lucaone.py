#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2025, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2025/12/30 11:33
@project: lucaone
@file: tokenization_lucaone
@desc: tokenization_lucaone
'''

import os
import json
import itertools
from typing import List, Optional, Dict, Any, Tuple, Union
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

def gene_seq_replace(seq):
    """
    Gene sequence preprocessing: A->1, U/T->2, C->3, G->4, N->5
    Optimized for performance.
    """
    # 使用字典映射比 if-else 判断快
    mapping = {
        'A': '1', 'a': '1',
        'T': '2', 't': '2', 'U': '2', 'u': '2',
        'C': '3', 'c': '3',
        'G': '4', 'g': '4'
    }
    # 对于不在字典中的字符（如 N），默认返回 '5'
    return "".join([mapping.get(ch, '5') for ch in seq])

class LucaGPLMTokenizer(PreTrainedTokenizer):
    """
    HuggingFace-compatible tokenizer that performs identical tokenization 
    to the old model's Alphabet class.
    """
    
    # Vocabulary definitions matching the old model
    gene_prepend_toks = ['[PAD]', '[UNK]']
    gene_append_toks = ['[CLS]', '[SEP]', '[MASK]']
    gene_standard_toks = ['1', '2', '3', '4', '5', '.', '-', '*']
    
    prot_prepend_toks = ['[PAD]', '[UNK]']
    prot_append_toks = ['[CLS]', '[SEP]', '[MASK]']
    prot_standard_toks = ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O', 'J', '.', '-', '*']
    
    gene_prot_prepend_toks = ['[PAD]', '[UNK]']
    gene_prot_append_toks = ['[CLS]', '[SEP]', '[MASK]']
    # EXACT VOCABULARY ORDER FROM ORIGINAL ALPHABET CLASS

    gene_prot_standard_toks = [
        '1',      # 5 - gene A (after gene_seq_replace)
        '2',      # 6 - gene T/U (after gene_seq_replace) 
        '3',      # 7 - gene C (after gene_seq_replace)
        '4',      # 8 - gene G (after gene_seq_replace)
        '5',      # 9 - gene N/unknown
        'L',      # 10 - protein
        'A',      # 11 - protein
        'G',      # 12 - protein
        'V',      # 13 - protein
        'S',      # 14 - protein
        'E',      # 15 - protein
        'R',      # 16 - protein
        'T',      # 17 - protein
        'I',      # 18 - protein
        'D',      # 19 - protein
        'P',      # 20 - protein
        'K',      # 21 - protein
        'Q',      # 22 - protein
        'N',      # 23 - protein
        'F',      # 24 - protein
        'Y',      # 25 - protein
        'M',      # 26 - protein
        'H',      # 27 - protein
        'W',      # 28 - protein
        'C',      # 29 - protein
        'X',      # 30 - protein unknown
        'B',      # 31 - protein
        'U',      # 32 - protein
        'Z',      # 33 - protein
        'O',      # 34 - protein
        'J',      # 35 - protein
        '.',      # 36 - special
        '-',      # 37 - special
        '*'       # 38 - special
    ]

    def __init__(
        self,
        vocab_type: str = "gene_prot",
        prepend_bos: bool = True,
        append_eos: bool = True,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        **kwargs
    ):
        # Set vocabulary based on type
        if vocab_type.lower() == "prot":
            prepend_toks = self.prot_prepend_toks
            append_toks = self.prot_append_toks
            standard_toks = self.prot_standard_toks
        elif vocab_type.lower() == "gene":
            prepend_toks = self.gene_prepend_toks
            append_toks = self.gene_append_toks
            standard_toks = self.gene_standard_toks
        elif vocab_type.lower() in ["gene_prot", "prot_gene"]:
            prepend_toks = self.gene_prot_prepend_toks
            append_toks = self.gene_prot_append_toks
            standard_toks = self.gene_prot_standard_toks
        else:
            raise ValueError(f"Not support tokenizer vocab_type: {vocab_type}")
        
        # Build vocabulary
        self.all_toks = list(prepend_toks) + list(append_toks) + list(standard_toks)
        self.tok_to_idx = {tok: i for i, tok in enumerate(self.all_toks)}
        self.idx_to_tok = {i: tok for i, tok in enumerate(self.all_toks)}
        
        # Store configuration
        self.vocab_type = vocab_type
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos
        self.unique_no_split_tokens = self.all_toks.copy()
        
        # Special token indices
        self.unk_idx = self.tok_to_idx.get("[UNK]", 1)
        self.padding_idx = self.tok_to_idx.get("[PAD]", 0)
        self.cls_idx = self.tok_to_idx.get("[CLS]", 2)
        self.mask_idx = self.tok_to_idx.get("[MASK]", 4)
        self.eos_idx = self.tok_to_idx.get("[SEP]", 3)
        
        super().__init__(
            unk_token=unk_token,
            pad_token=pad_token,
            cls_token=cls_token,
            sep_token=sep_token,
            mask_token=mask_token,
            **kwargs
        )

    def get_vocab(self) -> Dict[str, int]:
        return self.tok_to_idx.copy()

    @property
    def vocab_size(self) -> int:
        return len(self.all_toks)

    def get_idx(self, tok):
        return self.tok_to_idx.get(tok, self.unk_idx)

    def get_tok(self, idx):
        return self.idx_to_tok.get(idx, "[UNK]")

    def _tokenize_char_level(self, text: str) -> List[str]:
        """Simple character-level tokenization (fallback)"""
        return list(text)

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text using the same logic as the old Alphabet.tokenize() method
        """
        text = text.strip()
        if not text:
            return []
            
        return list(text)

    def _convert_token_to_id(self, token: str) -> int:
        return self.get_idx(token)

    def _convert_id_to_token(self, index: int) -> str:
        return self.get_tok(index)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return "".join(tokens)

    def _convert_text_to_ids(self, text: str, seq_type: str) -> List[int]:
        """Internal helper to convert text to IDs without special tokens."""
        if seq_type == "gene":
            text = gene_seq_replace(text)
        tokens = self._tokenize(text)
        return [self._convert_token_to_id(token) for token in tokens]

    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None) -> List[int]:
        """
        Build model inputs from a sequence by adding special tokens.
        This mimics the old model's prepend_bos and append_eos behavior.
        """
        result = token_ids_0.copy()
        
        if self.prepend_bos:
            result = [self.cls_idx] + result
        if self.append_eos:
            result = result + [self.eos_idx]
            
        return result

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        result = [0] * len(token_ids_0)
        if self.prepend_bos:
            result = [1] + result
        if self.append_eos:
            result = result + [1]
        return result

    def encode(
        self,
        text: str,
        seq_type: str = "gene",
        add_special_tokens: bool = True,
        padding: Union[bool, str] = False, # 虽然 encode 通常不处理 padding，但保持 API 兼容性
        truncation: bool = False,          # <--- 关键参数
        max_length: Optional[int] = None,  # <--- 关键参数
        **kwargs
    ) -> List[int]:
        
        # 1. 基础转换
        token_ids = self._convert_text_to_ids(text, seq_type)
        
        # 2. 添加特殊 token
        if add_special_tokens:
            token_ids = self.build_inputs_with_special_tokens(token_ids)
            
        # 3. 执行截断 (修复点：之前这里缺失逻辑)
        if truncation and max_length is not None and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
            # 如果启用了 append_eos，强行把截断后的最后一位改回 SEP
            if add_special_tokens and self.append_eos:
                token_ids[-1] = self.eos_idx
                
        return token_ids

    def __call__(
        self,
        text: Union[str, List[str]],
        text_pair: Optional[Union[str, List[str]]] = None,
        seq_type: str = "gene",
        add_special_tokens: bool = True,
        padding: Union[bool, str] = False,
        max_length: Optional[int] = None,
        return_attention_mask: bool = True,
        return_token_type_ids: bool = True,
        return_tensors: Optional[str] = None,
        truncation: bool = False,
        **kwargs
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Main callable method for tokenization - HuggingFace standard interface
        """
        if isinstance(text, list):
            # Handle batch processing
            return self.batch_encode_plus(
                text,
                text_pair=text_pair,
                seq_type=seq_type,
                add_special_tokens=add_special_tokens,
                padding=padding,
                max_length=max_length,
                return_attention_mask=return_attention_mask,
                return_token_type_ids=return_token_type_ids,
                return_tensors=return_tensors,
                truncation=truncation,
                **kwargs
            )
        else:
            # Handle single text
            return self.encode_plus(
                text,
                text_pair=text_pair,
                seq_type=seq_type,
                add_special_tokens=add_special_tokens,
                padding=padding,
                max_length=max_length,
                return_attention_mask=return_attention_mask,
                return_token_type_ids=return_token_type_ids,
                return_tensors=return_tensors,
                truncation=truncation,
                **kwargs
            )

    def batch_encode_plus(self, *args, **kwargs):
        # 显式调用父类，或者保留你原有的实现，只要确保内部调用的是修复后的 encode_plus 即可
        return super().batch_encode_plus(*args, **kwargs)

    def encode_plus(
        self,
        text: str,
        text_pair: Optional[str] = None,
        seq_type: str = "gene",
        add_special_tokens: bool = True,
        padding: Union[bool, str] = False,
        max_length: Optional[int] = None,
        return_attention_mask: bool = True,
        return_token_type_ids: bool = True,
        return_tensors: Optional[str] = None,
        truncation: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        
        # 调用修复后的 encode，它现在会正确处理截断
        token_ids = self.encode(
            text, 
            seq_type=seq_type, 
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            max_length=max_length
        )
        
        # 处理 Padding
        attention_mask = [1] * len(token_ids)
        if padding == "max_length" and max_length is not None:
            if len(token_ids) < max_length:
                pad_length = max_length - len(token_ids)
                token_ids.extend([self.padding_idx] * pad_length)
                attention_mask.extend([0] * pad_length)
        # 注意：padding=True (dynamic padding) 通常由 batch_encode_plus 处理，这里单条通常不处理
        
        result = {"input_ids": token_ids}
        
        if return_attention_mask:
            result["attention_mask"] = attention_mask
        
        if return_token_type_ids:
            # 0 for gene, 1 for protein
            type_value = 0 if seq_type == "gene" else 1
            result["token_type_ids"] = [type_value] * len(token_ids)
        
        if return_tensors == "pt":
            import torch
            for key, value in result.items():
                result[key] = torch.tensor(value, dtype=torch.long).unsqueeze(0)
        
        return result

    def encode_old_model_style(
        self,
        text: str,
        seq_type: str = "gene", 
        max_length: int = None
    ) -> List[int]:
        """
        Encode using the EXACT same process as the old model's encoder function.
        This replicates the logic from src/llm/lucaone_virus/get_embedding.py:encoder()
        """
        # Preprocess gene sequences (done in get_embedding function BEFORE calling encoder)
        if seq_type == "gene":
            text = gene_seq_replace(text)
        
        # Call tokenizer.encode (which does NOT include BOS/EOS in old model)
        seq_encoded = self.encode(text, seq_type=seq_type, add_special_tokens=False)
        
        # Apply max_length truncation if specified  
        if max_length and len(seq_encoded) > max_length:
            seq_encoded = seq_encoded[:max_length]
        
        # Calculate processed_seq_len (as done in old model)
        processed_seq_len = len(seq_encoded) + int(self.prepend_bos) + int(self.append_eos)
        
        # Create input_ids tensor (as done in old model encoder function)
        input_ids = [self.padding_idx] * processed_seq_len
        
        # Add BOS token if enabled
        if self.prepend_bos:
            input_ids[0] = self.cls_idx
            
        # Place the encoded sequence
        start_idx = int(self.prepend_bos)
        for i, token_id in enumerate(seq_encoded):
            input_ids[start_idx + i] = token_id
            
        # Add EOS token if enabled  
        if self.append_eos:
            input_ids[len(seq_encoded) + int(self.prepend_bos)] = self.eos_idx
            
        return input_ids

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the tokenizer vocabulary to a JSON file.
        Required by HuggingFace tokenizer interface.
        """
        if filename_prefix is None:
            filename_prefix = ""
        else:
            filename_prefix = filename_prefix + "-"
        
        vocab_file = os.path.join(save_directory, f"{filename_prefix}vocab.json")
        vocab_dict = self.get_vocab()
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(vocab_dict, f, ensure_ascii=False, indent=2)
        
        return (vocab_file,)

    @classmethod 
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """
        Load tokenizer from pretrained model path (standard HuggingFace interface)
        """
        vocab_file = os.path.join(pretrained_model_name_or_path, "vocab.json")
        if os.path.exists(vocab_file):
            print("Load from saved vocabulary (not implemented yet, use default)")
            return cls(vocab_type="gene_prot", **kwargs)
        else:
            return cls(vocab_type="gene_prot", **kwargs)

class LucaGPLMTokenizerFast(PreTrainedTokenizerFast):
    """
    Fast tokenizer version - currently just delegates to slow tokenizer
    """
    slow_tokenizer_class = LucaGPLMTokenizer
    
    def __init__(self, **kwargs):
        # For now, this is just a placeholder
        # In a full implementation, you would use the tokenizers library
        super().__init__(**kwargs)

__all__ = ["LucaGPLMTokenizer", "LucaGPLMTokenizerFast", "gene_seq_replace"]