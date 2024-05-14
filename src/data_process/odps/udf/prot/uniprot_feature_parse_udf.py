#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/10/26 10:07
@project: LucaOne
@file: uniprot_feature_parse_udf.py
@desc: uniprot feature parse udf
'''
from odps.udf import annotate


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


@annotate("string,string->string")
class uniprot_feature_parse(object):
    def __init__(self):
        import sys
        workaround_argv_issue("uniprot_feature_parse_udf.py")
        sys.path.insert(0, 'work/biopython-1.80-cp37.zip')

    def evaluate(self, value, keyword):
        value = eval(value)
        if keyword in value:
            return value[keyword]
        return None