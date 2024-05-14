#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/6/13 13:45
@project: LucaOne
@file: extract_position_udf_v3.py
@desc: extract position udf v3
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

@annotate("string->string")
class extract_position(object):
    def __init__(self):
        import sys
        workaround_argv_issue("extract_position_udf_v3.py")
        sys.path.insert(0, 'work/biopython-1.80-cp37.zip')

    def evaluate(self, value):
        from Bio.SeqFeature import SeqFeature, Reference, FeatureLocation, CompoundLocation, SimpleLocation, ExactPosition, UnknownPosition, UncertainPosition, WithinPosition, BetweenPosition, BeforePosition, AfterPosition, OneOfPosition, PositionGap
        try:
            start_end = eval(value)
            min_start = None
            max_end = None
            for pt in start_end.parts:
                start = int(pt.start)
                if min_start is None or min_start > start:
                    min_start = start
                end = int(pt.end)
                if max_end is None or max_end < end:
                    max_end = end
                return "%d,%d" %(min_start, max_end)
        except Exception as e:
            return None