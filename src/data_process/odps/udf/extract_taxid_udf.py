#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2024/2/28 10:07
@project: LucaOne
@file: extract_taxid_udf.py
@desc: extract taxid udf
RNA:
lucaone_data.tmp_lucaone_v2_data_gene2_01
DNA:
lucaone_data.tmp_lucaone_v2_data_gene2_02
prot:
uniref: lucaone_data.tmp_lucaone_v2_data_prot_01
uniprot: lucaone_data.tmp_lucaone_v2_data_prot_02
colab: lucaone_data.tmp_lucaone_v2_data_prot_03
'''

import sys, copy, json
from odps.udf import annotate


@annotate("string,string->string")
class extract_taxid(object):
    def evaluate(self, seq_type, value):
        label = json.loads(value, encoding="UTF-8")
        if seq_type == "gene":
            if "seq_level" in label \
                    and "gene_taxid" in label["seq_level"] \
                    and label["seq_level"]["gene_taxid"] \
                    and len(str(label["seq_level"]["gene_taxid"])) > 0:
                return label["seq_level"]["gene_taxid"]
        else:
            if "seq_level" in label \
                    and "prot_taxid" in label["seq_level"] \
                    and label["seq_level"]["prot_taxid"] \
                    and len(str(label["seq_level"]["prot_taxid"])) > 0:
                return label["seq_level"]["prot_taxid"]
        return None