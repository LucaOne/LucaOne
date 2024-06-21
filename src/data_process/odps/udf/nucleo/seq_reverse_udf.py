#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/11/1 11:56
@project: LucaOne
@file: seq_reverse.py
@desc: seq reverse
'''
from odps.udf import annotate
import re


# define a execfile function as it is depreciated in python3
def execfile(filepath, globals=None, locals=None):
    if globals is None:
        globals = {}
    globals.update({
        "__file__": filepath,
        "__name__": "__main__",
    })
    with open(filepath, 'rb') as file:
        exec(compile(file.read(), filepath, 'exec'), globals, locals)


def workaround_argv_issue(pyfile):
    import sys
    # set the sys.argv to be your udf py file.
    sys.argv = [pyfile]


def activate_venv(res_name):
    import os
    venv_path = 'work/' + res_name

    # source activate <your_venv>
    activate_script = os.path.join(venv_path, "bin", "activate_this.py")
    execfile(activate_script)


@annotate("string->string")
class seq_reverse(object):
    def evaluate(self, seq):
        if seq:
            reverse_seq = ""
            seq = seq[::-1].strip().upper()
            for ch in seq:
                if ch == "A":
                    reverse_seq += "T"
                elif ch in ["T", "U"]:
                    reverse_seq += "A"
                elif ch == "C":
                    reverse_seq += "G"
                elif ch == "G":
                    reverse_seq += "C"
                else:
                    reverse_seq += ch
            return reverse_seq
        return None







