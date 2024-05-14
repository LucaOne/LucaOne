#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/7/12 13:19
@project: LucaOne
@file: prot_label_fill_udf.py
@desc: prot label fill udf
100000
'''
import json
from odps.udf import annotate

@annotate("string,string->string")
class prot_label_fill(object):
    def evaluate(self, seq_id, label):
        if label is not None and len(label) > 0:
            obj = json.loads(label, encoding='UTF-8')[seq_id]
            if "seq_level" not in obj:
                obj["seq_level"] = {}
                obj["seq_level"]["prot_taxonomy"] = ""
                obj["seq_level"]["prot_taxid"] = ""
                obj["seq_level"]["prot_keyword"] = []
            else:
                if "prot_taxonomy" not in obj["seq_level"] or obj["seq_level"]["prot_taxonomy"] is None:
                    obj["seq_level"]["prot_taxonomy"] = ""
                if "prot_taxid" not in obj["seq_level"] or obj["seq_level"]["prot_taxid"] is None:
                    obj["seq_level"]["prot_taxid"] = ""
                if "prot_keyword" not in obj["seq_level"] or obj["seq_level"]["prot_keyword"] is None:
                    obj["seq_level"]["prot_keyword"] = []
            if "span_level" not in obj:
                obj["span_level"] = {}
                obj["span_level"]["prot_homo"] = []
                obj["span_level"]["prot_site"] = []
            else:
                if "prot_homo" not in obj["span_level"] or obj["span_level"]["prot_homo"] is None:
                    obj["span_level"]["prot_homo"] = []
                if "prot_site" not in obj["span_level"] or obj["span_level"]["prot_site"] is None:
                    obj["span_level"]["prot_site"] = []
            return json.dumps(obj, ensure_ascii=False)
        else:
            obj = {
                "span_level": {
                    "prot_homo": [],
                    "prot_site": []
                },
                "seq_level": {
                    "prot_taxonomy": "",
                    "prot_taxid": "",
                    "prot_keyword": []
                }
            }
            return json.dumps(obj, ensure_ascii=False)

