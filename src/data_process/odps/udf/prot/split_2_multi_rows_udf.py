#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/10/8 14:42
@project: LucaOne
@file: split_2_multi_rows_udf.py
@desc: split 2 multi rows udf
'''
from odps.udf import annotate
from odps.udf import BaseUDTF
import json
@annotate('string,string->string')
class split_2_multi_rows(BaseUDTF):
    def process(self, value, separator_char):
        if value:

            while True:
                idx1 = value.find("{")
                if idx1 == -1:
                    break
                idx2 = value.find("}")
                if idx2 == -1:
                    break
                value = value[0:idx1] + value[idx2+1:]

            strs = value.split(separator_char)
            for s in strs:
                self.forward(s.strip())