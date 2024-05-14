import sys, copy, json
from odps.udf import annotate
from odps.udf import BaseUDAF
from odps.distcache import get_cache_file
import ast

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


@annotate('string,string->string')
class prot_structure_add(object):

    def __init__(self):
        import sys
        workaround_argv_issue("prot_structure_add_udf.py")

    def evaluate(self,labels,coord_list):
        # try:
        labels_json = json.loads(labels, encoding='UTF-8')

        labels_json['structure_level'] = {}
        labels_json['structure_level']['prot_structure'] = ""

        if coord_list:

            labels_json['structure_level']['prot_structure'] = ast.literal_eval(coord_list)


        return  json.dumps(labels_json,ensure_ascii=False)


        # except Exception as e:

        #     return None 