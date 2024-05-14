#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/7/12 13:19
@project: LucaOne
@file: gene_label_fill_udf.py
@desc: gene label fill udf
100000
'''
import json
from odps.udf import annotate

@annotate("bigint,bigint,string->string")
class gene_label_fill(object):
    '''
    {"span_level": {"gene_type": [[0, 240, 0]]}, "seq_start_p": 0, "seq_end_p": 2045, "seq_level": {"gene_taxonomy": 17, "gene_taxid": "28141"}}
    '''
    def evaluate(self, seq_start_p, seq_end_p, label):
        if label is not None and len(label) > 0:
            obj = json.loads(label, encoding='UTF-8')
            if "seq_level" not in obj:
                obj["seq_level"] = {}
                obj["seq_level"]["gene_taxonomy"] = ""
                obj["seq_level"]["gene_taxid"] = ""
            else:
                if "gene_taxonomy" not in obj["seq_level"] or obj["seq_level"]["gene_taxonomy"] is None:
                    obj["seq_level"]["gene_taxonomy"] = ""
                if "gene_taxid" not in obj["seq_level"] or obj["seq_level"]["gene_taxid"] is None:
                    obj["seq_level"]["gene_taxid"] = ""
            if "span_level" not in obj:
                obj["span_level"] = {}
                obj["span_level"]["gene_type"] = []
            else:
                if "gene_type" not in obj["span_level"] or obj["span_level"]["gene_type"] is None:
                    obj["span_level"]["gene_type"] = []
            return json.dumps(obj, ensure_ascii=False)
        else:
            obj = {
                "span_level": {
                    "gene_type": []
                },
                "seq_start_p": seq_start_p,
                "seq_end_p": seq_end_p,
                "seq_level": {
                    "gene_taxonomy": "",
                    "gene_taxid": ""
                }
            }
            return json.dumps(obj, ensure_ascii=False)




