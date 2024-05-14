#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/11/1 11:56
@project: LucaOne
@file: interpro_db_priority.py
@desc: interpro db priority
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
class interpro_db_priority(object):
    def __init__(self):
        workaround_argv_issue("interpro_db_priority_udf.py")

        self.reg_expr_list = {
            "CDD": "cd\d+",
            "CATH": "G3DSA:\d+\.\d+\.\d+\.\d+",
            "HAMAP": "MF_\d+",
            "Pfam": "PF\d+",
            "PIRSF": "PIRSF\d+",
            "PRINTS": "PR\d+",
            "PROSITE": "PS\d+",
            "PANTHER": "PTHR\d+",
            "SFLD": "SFLD[SFG]\d+",
            "SMART": "SM\d+",
            "superfamily": "SSF\d+",
            "NCBIfam": "TIGR\d+"
        }

        db_priority_site = ["PROSITE", "Pfam", "NCBIfam", "SMART", "PRINTS", "CDD"]
        db_priority_sf = ["CATH", "superfamily"]
        self.db_priority_all = db_priority_sf + db_priority_site

        self.db_priority_dict = dict(zip(self.db_priority_all, range(len(self.db_priority_all))))

    def evaluate(self, value):
        if value is None:
            return "None;" + str(len(self.db_priority_all))
        for item in self.reg_expr_list.items():
            if re.match(item[1], value, re.I):
                if item[0] in self.db_priority_dict:
                    return item[0] + ";" + str(self.db_priority_dict[item[0]])

        return "None;" + str(len(self.db_priority_all))





