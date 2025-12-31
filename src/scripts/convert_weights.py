#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2025, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2025/12/30 14:29
@project: lucaone
@file: convert.py
@desc: convert pth to hf checkpoints
'''
import os
import json
import shutil
import torch
from lucaone import LucaGPLMConfig, LucaGPLMForMaskedLM, LucaGPLMTokenizer

def convert():
    # checkpoints local dir list
    ckpt_paths = [
        "../../checkpoints/lucaone/lucaone/checkpoint-step36000000/pytorch.pth",
        "../../checkpoints/lucaone/lucaone-gene/checkpoint-step36800000/pytorch.pth",
        "../../checkpoints/lucaone/lucaone-prot/checkpoint-step38200000/pytorch.pth",
        "../../checkpoints/lucaone/lucaone-mask/checkpoint-step36000000/pytorch.pth",
        "../../checkpoints/lucaone/lucaone/checkpoint-step17600000/pytorch.pth",
        "../../checkpoints/lucaone/lucaone/checkpoint-step5600000/pytorch.pth",
    ]
    save_paths = [
        "../../checkpoints/LucaGroup/LucaOne-default-step36M",
        "../../checkpoints/LucaGroup/LucaOne-gene-step36.8M",
        "../../checkpoints/LucaGroup/LucaOne-prot-step38.2M",
        "../../checkpoints/LucaGroup/LucaOne-mask-step36M",
        "../../checkpoints/LucaGroup/LucaOne-default-step17.6M",
        "../../checkpoints/LucaGroup/LucaOne-default-step5.6M",
    ]
    for ckpt_path, save_path in zip(ckpt_paths, save_paths):
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # 2. load checkpoint
        print(f"Loading original state_dict from {ckpt_path}...")
        old_state_dict = torch.load(ckpt_path, map_location="cpu")

        # shared weight
        shared_weight = old_state_dict.get("embed_tokens.weight")

        if shared_weight is not None:
            actual_vocab_size = shared_weight.shape[0]
            actual_hidden_size = shared_weight.shape[1]
            print(f"Inferred vocab_size: {actual_vocab_size}, hidden_size: {actual_hidden_size}")
        else:
            actual_vocab_size = 39 # 默认值
            actual_hidden_size = 2560
            print("Warning: embed_tokens.weight not found, using default sizes.")

        # 3. create config
        config = LucaGPLMConfig(
            vocab_size=actual_vocab_size,
            hidden_size=actual_hidden_size,
            num_hidden_layers=20,
            num_attention_heads=40,
            tie_word_embeddings=False # 默认设为 False，保证存储层的独立性
        )

        ckpt_config = json.load(open(os.path.join(os.path.dirname(ckpt_path), "config.json"), "r", encoding="utf-8"))
        for k, v in ckpt_config.items():
            # 使用 hasattr 检查对象是否有这个属性
            if hasattr(config, k) and k not in ["id2label", "label2id", "max_position_embeddings"]:
                # 使用 getattr 获取当前对象中的值
                current_val = getattr(config, k)
                if current_val != v:
                    print(f"Warning: {k} mismatch, expected {v}, got {current_val}. Updating config.")
                    # 使用 setattr 动态更新属性值
                    setattr(config, k, v)
            else:
                # 如果原始 config 对象里没有这个键，但你想从 json 里加进去
                print(f"Notice: Ignoring new config key {k}={v} from ckpt_config.")
                # setattr(config, k, v)

        # 定义 auto_map 映射，这是 AutoModel 能正常工作的核心
        auto_map = {
            "AutoConfig": "configuration_lucaone.LucaGPLMConfig",
            "AutoTokenizer": ["tokenization_lucaone.LucaGPLMTokenizer", None],
            "AutoModel": "modeling_lucaone.LucaGPLMModel",
            "AutoModelForMaskedLM": "modeling_lucaone.LucaGPLMForMaskedLM",
            "AutoModelForSequenceClassification": "modeling_lucaone.LucaGPLMForSequenceClassification",
            "AutoModelForTokenClassification": "modeling_lucaone.LucaGPLMForTokenClassification"
        }
        config.auto_map = auto_map

        # 实例化模型
        model = LucaGPLMForMaskedLM(config)
        # 实例化分词器
        tokenizer = LucaGPLMTokenizer(vocab_type="gene_prot")

        # 4. 开始权重映射
        print("Mapping weights...")
        new_state_dict = {}
        for k, v in old_state_dict.items():
            # A. 处理 LM Head 权重
            if k.startswith("lm_head"):
                # 原始可能是 lm_head.weight 或 lm_head.decoder.weight
                if k == "lm_head.weight" or k == "lm_head.decoder.weight":
                    new_state_dict["lm_head.decoder.weight"] = v
                else:
                    new_state_dict[k] = v

            # B. 忽略原始模型中与当前 HF 架构不一致的分类头
            elif any(k.startswith(prefix) for prefix in [
                "contact_head", "hidden_layer_list", "hidden_act_list",
                "classifier_dropout_list", "classifier_list", "output_list", "loss_fct_list"
            ]):
                continue

            # C. 映射 Encoder 权重 (注意路径前缀是 lucaone)
            elif k.startswith("layers."):
                # 路径: lucaone.encoder.layers.x...
                new_state_dict[f"lucaone.encoder.{k}"] = v
            elif k.startswith("last_layer_norm."):
                new_state_dict[f"lucaone.encoder.{k}"] = v
            elif k.startswith("embed_"):
                # 路径: lucaone.embeddings.embed_xxx
                new_state_dict[f"lucaone.embeddings.{k}"] = v

        # D. 核心修正：显式设置克隆的权重，确保解耦
        if shared_weight is not None:
            new_state_dict["lucaone.embeddings.embed_tokens.weight"] = shared_weight.clone()
            new_state_dict["lm_head.decoder.weight"] = shared_weight.clone()

        # 5. 加载权重到模型
        print("Loading state_dict into the model...")
        missing, unexpected = model.load_state_dict(new_state_dict, strict=True)

        if len(missing) > 0:
            print(f"Missing keys (Check if this is expected): {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected keys: {unexpected}")

        print("State dict loaded successfully!")

        # 6. 保存模型
        print(f"Saving model to {save_path}...")
        model.save_pretrained(save_path)

        # 7. 保存分词器并强制更新 tokenizer_config.json
        print(f"Saving tokenizer to {save_path}...")
        tokenizer.tokenizer_class = "LucaGPLMTokenizer"
        tokenizer.save_pretrained(save_path)

        # 手动注入 AutoTokenizer 映射到 tokenizer_config.json 确保 AutoTokenizer.from_pretrained 正常
        tokenizer_config_file = os.path.join(save_path, "tokenizer_config.json")
        if os.path.exists(tokenizer_config_file):
            with open(tokenizer_config_file, "r", encoding="utf-8") as f:
                t_config = json.load(f)

            t_config["tokenizer_class"] = "LucaGPLMTokenizer"
            t_config["auto_map"] = {
                "AutoTokenizer": ["tokenization_lucaone.LucaGPLMTokenizer", None]
            }

            with open(tokenizer_config_file, "w", encoding="utf-8") as f:
                json.dump(t_config, f, ensure_ascii=False, indent=2)

        print("-" * 50)
        print("CONVERSION COMPLETE!")
        print(f"Model Path: {save_path}")
        print("You can now load the model using:")
        print(f"model = AutoModel.from_pretrained('{save_path}', trust_remote_code=True)")

        print("Copying code files to save_path...")
        # 假设你的代码文件就在当前目录下，或者你知道它们的具体路径
        code_files = ["__init__.py", "configuration_lucaone.py", "modeling_lucaone.py", "tokenization_lucaone.py"]
        for f in code_files:
            # 如果你的文件在 src/lucaone 下，请修改路径，例如 os.path.join("src/lucaone", f)
            src_file = os.path.join("../lucaone", f)
            if os.path.exists(src_file):
                shutil.copy(src_file, save_path)
                print(f"  Copied {f}")
            else:
                print(f"  Warning: {f} not found, you may need to copy it manually.")
        print("done %s" % ckpt_path)
    print("All done!")

if __name__ == "__main__":
    convert()
