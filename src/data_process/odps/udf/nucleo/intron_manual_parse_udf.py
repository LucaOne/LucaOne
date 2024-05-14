#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/11/2 10:33
@project: LucaOne
@file: intron_manual_parse_udf.py
@desc: intron manual parse udf
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
class intron_manual_parse(object):
    def __init__(self):
        import sys
        workaround_argv_issue("intron_manual_parse_udf.py")
        sys.path.insert(0, 'work/biopython-1.80-cp37.zip')


    def parse_one(self, loc_str):
        from Bio.SeqFeature import SeqFeature, Reference, FeatureLocation, CompoundLocation, SimpleLocation, ExactPosition, UnknownPosition, UncertainPosition, WithinPosition, BetweenPosition, BeforePosition, AfterPosition, OneOfPosition, PositionGap
        loc = eval(loc_str)
        ft_intron = []
        pe = -1
        parts = []
        for pt in loc.parts:
            s, e = int(pt.start), int(pt.end)
            parts.append([s, e])
        parts = sorted(parts, key=lambda x:x[0])
        for pt in parts:
            s, e = pt[0], pt[1]
            if pe != -1 and pe < s:
                ft_intron.append(FeatureLocation(pe, s, loc.strand))
            pe = e
        if len(ft_intron) == 1:
            ft_return = repr(ft_intron.pop())
        elif len(ft_intron) > 1:
            ft_return = repr(CompoundLocation(ft_intron))
        else:
            ft_return = None
        return ft_return

    def evaluate(self, feature_type, start_end):
        if feature_type in ["CDS", "exon"] and "CompoundLocation" in start_end:
            return self.parse_one(start_end)
