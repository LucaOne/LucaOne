#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/11/1 11:56
@project: LucaOne
@file: interpro_db_priority_2_udf.py
@desc: interpro db priority_2_udf
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


@annotate("string->bigint")
class interpro_db_priority(object):
    def __init__(self):
        workaround_argv_issue("interpro_db_priority_2_udf.py")

        db_priority_site = ["PROSITE", "Pfam", "NCBIfam", "NCBIFAM", "SMART",
                            "PRINTS", "CDD", "Gene3D", "PANTHER", "PIRSF", "HAMAP", "SFLD"]
        db_priority_sf = ["CATH", "SUPFAM"]
        self.db_priority_all = db_priority_sf + db_priority_site

        self.db_priority_dict = dict(zip(self.db_priority_all, range(len(self.db_priority_all))))

    def evaluate(self, value):
        if value is None:
            return len(self.db_priority_all)
        if value in self.db_priority_dict:
            return self.db_priority_dict[value]

        return len(self.db_priority_all)





