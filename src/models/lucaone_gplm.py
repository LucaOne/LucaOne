#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/7/24 10:01
@project: LucaOne
@file: lucaone_gplm
@desc: LucaOne Model
'''
import sys
sys.path.append(".")
sys.path.append("..")
sys.path.append("../../")
sys.path.append("../../src")
try:
    from common.loss import *
    from models.model_utils import AllOutput, create_output_loss_lucagplm
    from models.alphabet import Alphabet
    from models.modeling_gplm import *
except ImportError:
    from src.common.loss import *
    from src.models.model_utils import AllOutput, create_output_loss_lucagplm
    from src.models.alphabet import Alphabet
    from src.models.modeling_gplm import *


class LucaGPLM(nn.Module):
    def __init__(
            self,
            config,
            args=None
    ):
        super().__init__()
        self.config = config
        self.has_contact_head = config.has_contact_head
        self.max_position_embeddings = config.max_position_embeddings
        self.type_vocab_size = config.type_vocab_size
        self.num_layers = config.num_hidden_layers
        self.embed_dim = config.hidden_size
        self.attention_heads = config.num_attention_heads
        self.no_position_embeddings = config.no_position_embeddings
        self.no_token_type_embeddings = config.no_token_type_embeddings
        if not isinstance(config.alphabet, Alphabet):
            self.alphabet = Alphabet.from_predefined(config.alphabet)
        else:
            self.alphabet = config.alphabet
        self.alphabet_size = len(self.alphabet)
        self.padding_idx = self.alphabet.padding_idx
        self.mask_idx = self.alphabet.mask_idx
        self.cls_idx = self.alphabet.cls_idx
        self.eos_idx = self.alphabet.eos_idx
        self.prepend_bos = self.alphabet.prepend_bos
        self.append_eos = self.alphabet.append_eos
        self.token_dropout = config.token_dropout
        self.ignore_index = config.ignore_index
        self.use_embed_layer_norm = config.use_embed_layer_norm
        self.use_last_layer_norm = config.use_last_layer_norm
        self.embed_scale = config.embed_scale
        if args and hasattr(args, "pretrained_model_name"):
            self.pretrained_model_name = args.pretrained_model_name
        else:
            self.pretrained_model_name = None
        # 如果只用于embedding 推理则有些层不加载与构建
        if args and hasattr(args, "embedding_inference"):
            self.embedding_inference = args.embedding_inference
        else:
            self.embedding_inference = False
        self._init_submodules()
        if self.pretrained_model_name is not None:
            # print("Load pretrained_model_name=%s" % self.pretrained_model_name)
            self._init_submodules_new(self.pretrained_model_name)

        if not self.embedding_inference and args is not None:
            self.pretrain_tasks = args.pretrain_tasks
            self.label_size = args.label_size
            self.loss_type = args.loss_type
            self.output_mode = args.output_mode
            self.cls = {}
            self.cls_list = []
            self.classifier_dropout = {}
            self.classifier_dropout_list = []
            self.hidden_layer = {}
            self.hidden_layer_list = []
            self.hidden_act = {}
            self.hidden_act_list = []
            self.classifier = {}
            self.classifier_list = []
            self.output = {}
            self.output_list = []
            self.loss_fct = {}
            self.loss_fct_list = []

            print("Pretrain Tasks:")
            for cur_item in self.pretrain_tasks.items():
                cur_task_level_type = cur_item[0]
                if cur_task_level_type not in self.cls:
                    self.cls[cur_task_level_type] = {}
                    self.classifier_dropout[cur_task_level_type] = {}
                    self.hidden_layer[cur_task_level_type] = {}
                    self.hidden_act[cur_task_level_type] = {}
                    self.classifier[cur_task_level_type] = {}
                    self.output[cur_task_level_type] = {}
                    self.loss_fct[cur_task_level_type] = {}
                for cur_task_level_name in cur_item[1]:
                    print(cur_task_level_type + "/" + cur_task_level_name)
                    cur_classifier_dropout, cur_hidden_layer, cur_hidden_act, cur_classifier, cur_output, cur_loss_fct \
                        = create_output_loss_lucagplm(cur_task_level_type, cur_task_level_name, config, args)

                    self.classifier_dropout[cur_task_level_type][cur_task_level_name] = cur_classifier_dropout
                    if cur_classifier_dropout is not None:
                        self.classifier_dropout_list.append(cur_classifier_dropout)

                    self.hidden_layer[cur_task_level_type][cur_task_level_name] = cur_hidden_layer
                    if cur_hidden_layer is not None:
                        self.hidden_layer_list.append(cur_hidden_layer)

                    self.hidden_act[cur_task_level_type][cur_task_level_name] = cur_hidden_act
                    if cur_hidden_act is not None:
                        self.hidden_act_list.append(cur_hidden_act)

                    self.classifier[cur_task_level_type][cur_task_level_name] = cur_classifier
                    if cur_classifier is not None:
                        self.classifier_list.append(cur_classifier)

                    self.output[cur_task_level_type][cur_task_level_name] = cur_output
                    if cur_output is not None:
                        self.output_list.append(cur_output)

                    self.loss_fct[cur_task_level_type][cur_task_level_name] = cur_loss_fct
                    if cur_loss_fct is not None:
                        self.loss_fct_list.append(cur_loss_fct)
            if self.cls_list and len(self.cls_list) > 0:
                self.cls_list = nn.ModuleList(self.cls_list)
            if self.hidden_layer_list and len(self.hidden_layer_list) > 0:
                self.hidden_layer_list = nn.ModuleList(self.hidden_layer_list)
            if self.hidden_act_list and len(self.hidden_act_list) > 0:
                self.hidden_act_list = nn.ModuleList(self.hidden_act_list)
            if self.classifier_dropout_list and len(self.classifier_dropout_list) > 0:
                self.classifier_dropout_list = nn.ModuleList(self.classifier_dropout_list)
            if self.classifier_list and len(self.classifier_list) > 0:
                self.classifier_list = nn.ModuleList(self.classifier_list)
            if self.output_list and len(self.output_list) > 0:
                self.output_list = nn.ModuleList(self.output_list)
            if self.loss_fct_list and len(self.loss_fct_list) > 0:
                self.loss_fct_list = nn.ModuleList(self.loss_fct_list)

    def _init_submodules(self):
        # normal_(0, 1)
        self.embed_tokens = nn.Embedding(
            self.alphabet_size,
            self.embed_dim,
            padding_idx=self.padding_idx,
        )
        self.embed_pos = None
        if not self.no_position_embeddings:
            self.embed_pos = nn.Embedding(self.max_position_embeddings, self.embed_dim)
        self.embed_type = None
        if not self.no_token_type_embeddings:
            self.embed_type = nn.Embedding(self.type_vocab_size, self.embed_dim)
        if self.use_embed_layer_norm:
            self.embed_layer_norm = LucaGPLM1bLayerNorm(self.embed_dim)
        else:
            self.embed_layer_norm = None

        self.layers = nn.ModuleList(
            [
                LucaGPLMTransformerLayer(
                    self.embed_dim,
                    4 * self.embed_dim,
                    self.attention_heads,
                    add_bias_kv=False,
                    use_lucagplm1b_layer_norm=True,
                    use_rotary_embeddings=self.no_position_embeddings,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.layer_size = len(self.layers)

        if not self.embedding_inference and self.has_contact_head:
            self.contact_head = ContactPredictionHead(
                self.num_layers * self.attention_heads,
                self.prepend_bos,
                self.append_eos,
                eos_idx=self.eos_idx,
                )
        if self.use_last_layer_norm:
            self.last_layer_norm = LucaGPLM1bLayerNorm(self.embed_dim)
        else:
            self.last_layer_norm = None
        if not self.embedding_inference:
            self.lm_head = RobertaLMHead(
                embed_dim=self.embed_dim,
                output_dim=self.alphabet_size,
                weight=self.embed_tokens.weight,
            )

    def _init_embedding(self, pretrained_token_matrix, token_matrix):
        '''
        0->2
        1->0
        2->3
        3->1
        4->10
        ...
        28->34
        29->36
        30->37
        31->38
        32->4
        '''
        # print("Load pretrained exists embedding vectors:")
        token_matrix[2, :] = pretrained_token_matrix[0, :]
        token_matrix[0, :] = pretrained_token_matrix[1, :]
        token_matrix[3, :] = pretrained_token_matrix[2, :]
        token_matrix[1, :] = pretrained_token_matrix[3, :]
        for idx in range(10, 35):
            token_matrix[idx, :] = pretrained_token_matrix[idx - 6, :]
        token_matrix[36, :] = pretrained_token_matrix[29, :]
        token_matrix[37, :] = pretrained_token_matrix[30, :]
        token_matrix[38, :] = pretrained_token_matrix[31, :]
        token_matrix[4, :] = pretrained_token_matrix[32, :]
        return token_matrix

    def _init_submodules_new(self, pretrained_model_name):
        # print("Load pretrained model exists weights:")
        from esm import pretrained
        from collections import OrderedDict
        pretrained, _ = pretrained.load_model_and_alphabet(pretrained_model_name)
        pretrained_state_dict = pretrained.state_dict()
        new_state_dict = OrderedDict()
        our_model_state_dict = {}
        for key, value in self.state_dict().items():
            our_model_state_dict[key] = value
        for name, weight in pretrained_state_dict.items():
            if "final_layer_norm" in name:
                name = name.replace("final_layer_norm", "post_layer_norm")
            elif "self_attn_layer_norm" in name:
                name = name.replace("self_attn_layer_norm", "pre_layer_norm")
            elif "emb_layer_norm_after" in name:
                name = name.replace("emb_layer_norm_after", "last_layer_norm")
            if name.startswith("layers."):
                layer_id = name.split(".")[1]
                if int(layer_id) >= self.num_layers:
                    continue
            if name == "embed_tokens.weight":
                new_state_dict[name] = self._init_embedding(weight, our_model_state_dict[name])
                del our_model_state_dict[name]
            elif name in our_model_state_dict and our_model_state_dict[name].shape == weight.shape:
                del our_model_state_dict[name]
                new_state_dict[name] = weight
        '''
        print("Exists layer names:")
        print(new_state_dict.keys())
        print("Not exists Layer names:")
        print(our_model_state_dict.keys())
        '''
        new_state_dict.update(our_model_state_dict)
        self.load_state_dict(new_state_dict)

    def __calc_loss__(self, task_level_type, output_mode, logits, label, label_size, loss_fct, loss_reduction):
        if output_mode in ["regression"]:
            if task_level_type not in ["seq_level"] and loss_reduction == "meanmean":
                # structure-level regression
                # logits: N, seq_len, 3
                # label: N, seq_len, 3
                loss = loss_fct(logits, label)
            else:
                # structure-level regression
                # logits: N * seq_len * 3
                # label: N * seq_len * 3
                loss = loss_fct(logits.view(-1), label.view(-1))
        elif output_mode in ["multi_label", "multi-label"]:
            # only for seq-level
            if loss_reduction == "meanmean":
                # logits: N , label_size
                # label: N , label_size
                loss = loss_fct(logits, label.float())
            else:
                # logits: N , label_size
                # label: N , label_size
                loss = loss_fct(logits.view(-1, label_size), label.view(-1, label_size).float())
        elif label_size <= 2 or output_mode in ["binary_class", "binary-class"]:
            if task_level_type not in ["seq_level"] and loss_reduction == "meanmean":
                # token-level & meanmean
                # logits: N ,seq_len, 1
                # label: N, seq_len
                loss = loss_fct(logits, label.float())
            else:
                # seq-level || token-level
                # logits: N
                # label: N
                loss = loss_fct(logits.view(-1), label.view(-1).float())
        elif output_mode in ["multi_class", "multi-class"]:
            if task_level_type not in ["seq_level"] and loss_reduction == "meanmean":
                # token-level
                # logits: N ,seq_len, label_size
                # label: N , seq_len
                loss = loss_fct(logits, label)
            else:
                # token-level
                # logits: N * seq_len, label_size
                # label: N * seq_len
                # seq-level
                # logits: N, label_size
                # label: N
                loss = loss_fct(logits.view(-1, label_size), label.view(-1))
        else:
            raise Exception("Not support output_mode=%s" % output_mode)
        return loss

    def __forward__(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            output_keys: Optional[dict[str, set[str]]] = None,
            labels: Optional[dict[str, dict[str, torch.Tensor]]] = None,
            repr_layers=[-1],
            need_head_weights=False,
            return_contacts=False,
            use_last_layer_norm=True
    ):
        assert all(-(self.layer_size + 1) <= i <= self.layer_size for i in repr_layers)
        repr_layers = [(i + self.layer_size + 1) % (self.layer_size + 1) for i in repr_layers]

        if return_contacts:
            need_head_weights = True

        if need_head_weights:
            need_weights = True
        else:
            need_weights = False

        assert input_ids.ndim == 2
        # 动态求mask，(B * Seq_len) 被mask掉位置的值为True
        if attention_mask is None:
            padding_mask = input_ids.eq(self.padding_idx)
        else:
            padding_mask = attention_mask.eq(self.padding_idx)

        x = self.embed_scale * self.embed_tokens(input_ids)
        if self.embed_pos is not None and position_ids is not None:
            x += self.embed_scale * self.embed_pos(position_ids)
        if self.embed_type is not None and token_type_ids is not None:
            x += self.embed_scale * self.embed_type(token_type_ids)
        if self.embed_layer_norm is not None:
            x = self.embed_layer_norm(x)
        # Token dropout
        if self.token_dropout:
            x.masked_fill_((input_ids == self.mask_idx).unsqueeze(-1), 0.0)
            # x: B x L x C
            mask_ratio_train = 0.15 * 0.8
            src_lengths = (~padding_mask).sum(-1)
            mask_ratio_observed = (input_ids == self.mask_idx).sum(-1).to(x.dtype) / src_lengths
            x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]

        # Mask 操作
        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        # 返回值包括哪些
        repr_layers = set(repr_layers)
        hidden_representations = {}
        # 0:embedding
        if 0 in repr_layers:
            hidden_representations[0] = x

        # 是否需要返回head weights
        if need_head_weights:
            attn_weights = []

        # (B, L, E) => (L, B, E)
        x = x.transpose(0, 1)

        if not padding_mask.any():
            padding_mask = None

        for layer_idx, layer in enumerate(self.layers):
            x, attn = layer(
                x,
                self_attn_padding_mask=padding_mask,
                need_head_weights=need_head_weights,
                need_weights=need_weights
            )
            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x.transpose(0, 1)
            if need_head_weights:
                # (H, B, L, L) => (B, H, L, L)
                attn_weights.append(attn.transpose(1, 0))

        # (L, B, E)
        if self.last_layer_norm is not None and use_last_layer_norm:
            # 最后一层隐含层 加一层layernorm
            x = self.last_layer_norm(x)
        x = x.transpose(0, 1)  # (L, B, E) => (B, L,  E)

        # last hidden representation should have layer norm applied
        if (layer_idx + 1) in repr_layers:
            hidden_representations[layer_idx + 1] = x
        # 最后一层作为表征矩阵
        # (B, L, E)
        representation_matrix = hidden_representations[self.layer_size]
        # (B, E)
        representation_vector = representation_matrix[:, 0, :]
        representations = {
            "representation_matrix": representation_matrix,
            "representation_vector": representation_vector
        }
        # mask 任务
        # B * Seq_len * vocab_size
        if not self.embedding_inference:
            lm_mask_logits = self.lm_head(x)
        logits = {}
        losses = {}
        outputs = {}
        # 每一层的attention值
        if need_head_weights:
            # attentions: B x Layers x H x L x L
            attentions = torch.stack(attn_weights, 1)
            if padding_mask is not None:
                attention_mask = 1 - padding_mask.type_as(attentions)
                attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
                attentions = attentions * attention_mask[:, None, None, :, :]
            representations["attentions"] = attentions
            # 预测contact矩阵
            if return_contacts and hasattr(self, "contact_head") \
                    and not self.embedding_inference:
                contacts = self.contact_head(input_ids, attentions)
                representations["contacts"] = contacts

        if not self.embedding_inference and output_keys:
            for item in output_keys.items():
                cur_task_level_type = item[0]
                if cur_task_level_type not in logits:
                    logits[cur_task_level_type] = {}
                    outputs[cur_task_level_type] = {}
                for cur_task_level_name in item[1]:
                    if cur_task_level_type == "token_level":
                        cur_logits = lm_mask_logits
                    elif cur_task_level_type == "seq_level":
                        cur_logits = self.classifier_dropout[cur_task_level_type][cur_task_level_name](representation_vector)
                        cur_hidden_layer = self.hidden_layer[cur_task_level_type][cur_task_level_name]
                        if cur_hidden_layer is not None:
                            cur_logits = cur_hidden_layer(cur_logits)
                        cur_hidden_act = self.hidden_act[cur_task_level_type][cur_task_level_name]
                        if cur_hidden_act is not None:
                            cur_logits = cur_hidden_act(cur_logits)
                        cur_logits = self.classifier[cur_task_level_type][cur_task_level_name](cur_logits)
                    elif cur_task_level_type == "span_level":
                        cur_logits = self.classifier_dropout[cur_task_level_type][cur_task_level_name](representation_matrix)
                        cur_hidden_layer = self.hidden_layer[cur_task_level_type][cur_task_level_name]
                        if cur_hidden_layer is not None:
                            cur_logits = cur_hidden_layer(cur_logits)
                        cur_hidden_act = self.hidden_act[cur_task_level_type][cur_task_level_name]
                        if cur_hidden_act is not None:
                            cur_logits = cur_hidden_act(cur_logits)
                        cur_logits = self.classifier[cur_task_level_type][cur_task_level_name](cur_logits)
                    elif cur_task_level_type == "structure_level":
                        cur_logits = self.classifier_dropout[cur_task_level_type][cur_task_level_name](representation_matrix)
                        cur_hidden_layer = self.hidden_layer[cur_task_level_type][cur_task_level_name]
                        if cur_hidden_layer is not None:
                            cur_logits = cur_hidden_layer(cur_logits)
                        cur_hidden_act = self.hidden_act[cur_task_level_type][cur_task_level_name]
                        if cur_hidden_act is not None:
                            cur_logits = cur_hidden_act(cur_logits)
                        cur_logits = self.classifier[cur_task_level_type][cur_task_level_name](cur_logits)
                    logits[cur_task_level_type][cur_task_level_name] = cur_logits
                    if cur_task_level_type in self.output and cur_task_level_name in self.output[cur_task_level_type] \
                            and self.output[cur_task_level_type][cur_task_level_name] is not None:
                        outputs[cur_task_level_type][cur_task_level_name] = self.output[cur_task_level_type][cur_task_level_name](cur_logits)
                    else:
                        outputs[cur_task_level_type][cur_task_level_name] = cur_logits
                    if labels is not None and cur_task_level_type in labels and cur_task_level_name in labels[cur_task_level_type]:
                        if cur_task_level_type not in losses:
                            losses[cur_task_level_type] = {}
                        cur_label = labels[cur_task_level_type][cur_task_level_name]
                        cur_label_size = self.label_size[cur_task_level_type][cur_task_level_name]
                        cur_output_mode = self.output_mode[cur_task_level_type][cur_task_level_name]
                        cur_loss_fct = self.loss_fct[cur_task_level_type][cur_task_level_name]
                        cur_loss = self.__calc_loss__(
                            task_level_type=cur_task_level_type,
                            output_mode=cur_output_mode,
                            logits=cur_logits,
                            label=cur_label,
                            label_size=cur_label_size,
                            loss_fct=cur_loss_fct,
                            loss_reduction="meanmean"
                        )
                        losses[cur_task_level_type][cur_task_level_name] = cur_loss
        return representations, logits, outputs, losses

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            global_attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            output_keys: Optional[dict[str, set[str]]] = None,
            labels: Optional[dict[str, dict[str, torch.Tensor]]] = None,
            input_ids_b: Optional[torch.Tensor] = None,
            attention_mask_b: Optional[torch.Tensor] = None,
            global_attention_mask_b: Optional[torch.Tensor] = None,
            token_type_ids_b: Optional[torch.Tensor] = None,
            position_ids_b: Optional[torch.Tensor] = None,
            head_mask_b: Optional[torch.Tensor] = None,
            inputs_embeds_b: Optional[torch.Tensor] = None,
            output_keys_b: Optional[dict[str, set[str]]] = None,
            labels_b: Optional[dict[str, dict[str, torch.Tensor]]] = None,
            pair_label: Optional[dict[str, dict[str, torch.Tensor]]] = None,
            pair_output_keys: Optional[dict[str, set[str]]] = None,
            output_hidden_states: Optional[dict[str, set[str]]] = None,
            output_attentions: Optional[dict[str, set[str]]] = None,
            need_head_weights: Optional[bool] = None,
            return_contacts: Optional[bool] = None,
            repr_layers: Optional[list[int]] = None,
            return_dict: Optional[bool] = None,
            use_last_layer_norm: Optional[bool] = True
    ) -> Union[Tuple[torch.Tensor], AllOutput]:
        if return_dict is None and self.config is not None:
            return_dict = self.config.use_return_dict
        if return_dict is None:
            return_dict = False
        if repr_layers is None or len(repr_layers) == 0:
            repr_layers = [-1]
        if return_contacts is None:
            return_contacts = False
        if need_head_weights is None:
            need_head_weights = True
        has_pair = False
        has_pair_b = False
        if input_ids is not None or inputs_embeds is not None:
            encoding, logits, outputs, losses = self.__forward__(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                output_keys=output_keys,
                labels=labels,
                repr_layers=repr_layers,
                need_head_weights=need_head_weights,
                return_contacts=return_contacts,
                use_last_layer_norm=use_last_layer_norm
            )
            has_pair = True
        if input_ids_b is not None or inputs_embeds_b is not None:
            encoding_b, logits_b, outputs_b, losses_b = self.__forward__(
                input_ids=input_ids_b,
                attention_mask=attention_mask_b,
                token_type_ids=token_type_ids_b,
                position_ids=position_ids_b,
                output_keys=output_keys_b,
                labels=labels_b,
                repr_layers=repr_layers,
                need_head_weights=need_head_weights,
                return_contacts=return_contacts,
                use_last_layer_norm=use_last_layer_norm
            )
            has_pair_b = True

        if not self.embedding_inference:
            if has_pair and has_pair_b and pair_output_keys and len(pair_output_keys) > 0:
                cur_representation_vector = encoding["representation_vector"]
                cur_representation_vector_b = encoding_b["representation_vector"]

                pair_logits = {}
                pair_outputs = {}
                for item1 in pair_output_keys.items():
                    cur_task_level_type = item1[0]
                    if cur_task_level_type not in pair_outputs:
                        pair_outputs[cur_task_level_type] = {}
                        pair_logits[cur_task_level_type] = {}
                    for cur_task_level_name in item1[1]:
                        cur_logits = self.classifier_dropout[cur_task_level_type][cur_task_level_name](
                            torch.cat((cur_representation_vector, cur_representation_vector_b), dim=-1)
                        )
                        cur_hidden_layer = self.hidden_layer[cur_task_level_type][cur_task_level_name]
                        if cur_hidden_layer is not None:
                            cur_logits = cur_hidden_layer(cur_logits)
                        cur_logits = self.classifier[cur_task_level_type][cur_task_level_name](cur_logits)
                        pair_logits[cur_task_level_type][cur_task_level_name] = cur_logits
                        pair_outputs[cur_task_level_type][cur_task_level_name] = self.output[cur_task_level_type][cur_task_level_name](cur_logits)

                if pair_label is not None:
                    pair_loss = {}
                    for item1 in pair_output_keys.items():
                        cur_task_level_type = item1[0]
                        if cur_task_level_type not in pair_label:
                            continue
                        if cur_task_level_type in pair_label:
                            pair_loss[cur_task_level_type] = {}
                        for cur_task_level_name in item1[1]:
                            if cur_task_level_name not in pair_label[cur_task_level_type]:
                                continue
                            cur_label = pair_label[cur_task_level_type][cur_task_level_name]
                            cur_label_size = self.label_size[cur_task_level_type][cur_task_level_name]
                            cur_output_mode = self.output_mode[cur_task_level_type][cur_task_level_name]
                            cur_loss_fct = self.loss_fct[cur_task_level_type][cur_task_level_name]
                            cur_logits = pair_logits[cur_task_level_type][cur_task_level_name]
                            cur_loss = self.__calc_loss__(
                                task_level_type=cur_task_level_type,
                                output_mode=cur_output_mode,
                                logits=cur_logits,
                                label=cur_label,
                                label_size=cur_label_size,
                                loss_fct=cur_loss_fct,
                                loss_reduction="meanmean"
                            )
                            pair_loss[cur_task_level_type][cur_task_level_name] = cur_loss

                    if not return_dict:
                        return [[losses, losses_b, pair_loss], [outputs, outputs_b, pair_outputs]] + [[encoding, encoding_b]]
                    return AllOutput(
                        losses=losses,
                        outputs=outputs,
                        hidden_states=encoding["representation_matrix"] if "representation_matrix" in encoding else None,
                        attentions=encoding["attentions"] if "attentions" in encoding else None,
                        global_attentions=None,
                        contacts=encoding["contacts"] if "contacts" in encoding else None,
                        losses_b=losses_b,
                        outputs_b=outputs_b,
                        hidden_states_b=encoding_b["representation_matrix"] if "representation_matrix" in encoding_b else None,
                        attentions_b=encoding_b["attentions"] if "hidden_states" in encoding_b else None,
                        global_attentions_b=None,
                        contacts_b=encoding_b["contacts"] if "contacts" in encoding_b else None,
                        pair_outputs=pair_outputs,
                        pair_losses=pair_loss)
                else:
                    if not return_dict:
                        return [[losses, losses_b], [outputs, outputs_b]] + [[encoding, encoding_b]]
                    return AllOutput(
                        losses=losses,
                        outputs=outputs,
                        hidden_states=encoding["representation_matrix"] if "representation_matrix" in encoding else None,
                        attentions=encoding["attentions"] if "attentions" in encoding else None,
                        global_attentions=None,
                        contacts=encoding["contacts"] if "contacts" in encoding else None,
                        losses_b=losses_b,
                        outputs_b=outputs_b,
                        hidden_states_b=encoding_b["representation_matrix"] if "representation_matrix" in encoding_b else None,
                        attentions_b=encoding_b["attentions"] if "attentions" in encoding_b else None,
                        global_attentions_b=None,
                        contacts_b=encoding_b["contacts"] if "contacts" in encoding_b else None
                    )
            elif has_pair:
                if not return_dict:
                    return [[losses], [outputs], [encoding]]
                return AllOutput(
                    losses=losses,
                    outputs=outputs,
                    hidden_states=encoding["representation_matrix"] if "representation_matrix" in encoding else None,
                    attentions=encoding["attentions"] if "attentions" in encoding else None,
                    global_attentions=None,
                    contacts=encoding["contacts"] if "contacts" in encoding else None
                )
            else:
                if not return_dict:
                    return [[losses_b], [outputs_b], [encoding_b]]
                return AllOutput(
                    losses_b=losses_b,
                    outputs_b=outputs_b,
                    hidden_states_b=encoding_b["representation_matrix"] if "representation_matrix" in encoding_b else None,
                    attentions_b=encoding_b["attentions"] if "attentions" in encoding_b else None,
                    global_attentions_b=None,
                    contacts_b=encoding_b["contacts"] if "contacts" in encoding_b else None
                )
        else:
            if has_pair and has_pair_b:
                if not return_dict:
                    return [[None, None], [None, None]] + [[encoding, encoding_b]]
                return AllOutput(
                    losses=None,
                    outputs=None,
                    hidden_states=encoding["representation_matrix"] if "representation_matrix" in encoding else None,
                    attentions=encoding["attentions"] if "attentions" in encoding else None,
                    global_attentions=None,
                    contacts=encoding["contacts"] if "contacts" in encoding else None,
                    losses_b=None,
                    outputs_b=None,
                    hidden_states_b=encoding_b["representation_matrix"] if "representation_matrix" in encoding_b else None,
                    attentions_b=encoding_b["attentions"] if "attentions" in encoding_b else None,
                    global_attentions_b=None,
                    contacts_b=encoding_b["contacts"] if "contacts" in encoding_b else None
                )
            elif has_pair:
                if not return_dict:
                    return [[None], [None], [encoding]]
                return AllOutput(
                    losses=None,
                    outputs=None,
                    hidden_states=encoding["representation_matrix"] if "representation_matrix" in encoding else None,
                    attentions=encoding["attentions"] if "attentions" in encoding else None,
                    global_attentions=None,
                    contacts=encoding["contacts"] if "contacts" in encoding else None
                )
            else:
                if not return_dict:
                    return [[None], [None], [encoding_b]]
                return AllOutput(
                    losses_b=None,
                    outputs_b=None,
                    hidden_states_b=encoding_b["representation_matrix"] if "representation_matrix" in encoding_b else None,
                    attentions_b=encoding_b["attentions"] if "attentions" in encoding_b else None,
                    global_attentions_b=None,
                    contacts_b=encoding_b["contacts"] if "contacts" in encoding_b else None
                )

    def predict_contacts(self, input_ids, position_ids=None, token_type_ids=None):
        return self(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            return_contacts=True
        )["contacts"]
