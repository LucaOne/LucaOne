#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/10/8 15:09
@project: LucaOne
@file: structure_verify_udf.py
@desc: structure verify udf
'''
import json
from odps.udf import annotate

@annotate("string,string,string->string")
class structure_verify(object):
    def evaluate(self, seq_id, seq, coord_list):
        seq_len = len(seq)
        obj = json.loads(coord_list, encoding='UTF-8')
        if len(obj) != seq_len:
            return "seq=%d,coord=%d" %(len(obj), len(obj))
        return None