#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/11/18 17:27
@project: LucaOne
@file: seq_clean_udf.py
@desc: seq clean udf
'''

from odps.udf import annotate


@annotate('string->string')
class seq_clean(object):
    def evaluate(self, value):
        if value is None:
            return None
        value = value.upper()
        new_value = ""
        for ch in value:
            if "A" <= ch <= "Z":
                new_value += ch
        if len(new_value) > 0:
            return new_value
        return None
