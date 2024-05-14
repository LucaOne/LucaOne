#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2024/2/28 10:07
@project: LucaOne
@file: data_stats.py
@desc: data stats
RNA:
lucaone_data.tmp_lucaone_v2_data_gene2_01
DNA:
lucaone_data.tmp_lucaone_v2_data_gene2_02
prot:
uniref: lucaone_data.tmp_lucaone_v2_data_prot_01
uniprot: lucaone_data.tmp_lucaone_v2_data_prot_02
colab: lucaone_data.tmp_lucaone_v2_data_prot_03
'''

import sys, copy, json
from odps.udf import annotate
from odps.udf import BaseUDAF
from odps.distcache import get_cache_file

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


@annotate('string,string,string,string->string')
class data_stats(BaseUDAF):
    def new_buffer(self):
        return {
            "t": 0,
            "min_len": 100000,
            "max_len": 0,
            "avg_len": 0,
            "gene_type": 0,
            "gene_type_set": set(),
            "gene_taxonomy": 0,
            "gene_taxonomy_set": set(),
            "prot_taxonomy": 0,
            "prot_taxonomy_set": set(),
            "prot_keyword": 0,
            "prot_keyword_set": set(),
            "prot_site": 0,
            "prot_site_set": set(),
            "prot_domain": 0,
            "prot_domain_set": set(),
            "prot_homo": 0,
            "prot_homo_set": set(),
            "prot_structure": 0
        }

    def iterate(self, buffer, seq_id, seq_type, seq, label):
        # add [CLS] and [SEP]
        seq_len = len(seq) + 2
        '''
        {
            "span_level": 
            {
                "gene_type": [[0, 1277, 7]]
            }, 
            "seq_start_p": 1278, 
            "seq_end_p": 2556, 
            "seq_level": 
            {
                "gene_taxonomy": 115, 
                "gene_taxid": "31033"
            }
        }
        
        {
            "seq_level": {
                "prot_taxid": "", 
                "prot_taxonomy": "", 
                "prot_keyword": []
            }, 
            "span_level": {
                "prot_homo": [], 
                "prot_site": [], 
                "prot_domain": []
            }, 
            "structure_level": {
                "prot_structure": ""
            }
        }
        
        '''
        label = json.loads(label, encoding="UTF-8")
        buffer["t"] += 1
        if buffer["min_len"] > seq_len:
            buffer["min_len"] = seq_len
        if buffer["max_len"] < seq_len:
            buffer["max_len"] = seq_len
        buffer["avg_len"] += seq_len
        if seq_type == "gene":
            if "span_level" in label and "gene_type" in label["span_level"] and label["span_level"]["gene_type"] and len(label["span_level"]["gene_type"]) > 0:
                gene_type_sets = set()
                for item in label["span_level"]["gene_type"]:
                    gene_type_idx = item[-1]
                    buffer["gene_type_set"].add(gene_type_idx)
                    gene_type_sets.add(gene_type_idx)
                if len(gene_type_sets) > 0:
                    buffer["gene_type"] += 1
            if "seq_level" in label and "gene_taxonomy" in label["seq_level"] and label["seq_level"]["gene_taxonomy"] and len(str(label["seq_level"]["gene_taxonomy"])) > 0:
                order_idx = label["seq_level"]["gene_taxonomy"]
                if order_idx > -1:
                    buffer["gene_taxonomy"] += 1
                    buffer["gene_taxonomy_set"].add(order_idx)
        else:
            if "seq_level" in label and "prot_taxonomy" in label["seq_level"] and label["seq_level"]["prot_taxonomy"] and len(str(label["seq_level"]["prot_taxonomy"])) > 0:
                order_idx = label["seq_level"]["prot_taxonomy"]
                if order_idx > -1:
                    buffer["prot_taxonomy"] += 1
                    buffer["prot_taxonomy_set"].add(order_idx)
            if "seq_level" in label and "prot_keyword" in label["seq_level"] and label["seq_level"]["prot_keyword"] and len(label["seq_level"]["prot_keyword"]) > 0:
                keyword_set = set()
                for keyword_idx in label["seq_level"]["prot_keyword"]:
                    buffer["prot_keyword_set"].add(keyword_idx)
                    keyword_set.add(keyword_idx)
                if len(keyword_set) > 0:
                    buffer["prot_keyword"] += 1
            if "span_level" in label and "prot_homo" in label["span_level"] and label["span_level"]["prot_homo"] and len(label["span_level"]["prot_homo"]) > 0:
                homo_set = set()
                for item in label["span_level"]["prot_homo"]:
                    homo_idx = item[-1]
                    buffer["prot_homo_set"].add(homo_idx)
                    homo_set.add(homo_idx)
                if len(homo_set) > 0:
                    buffer["prot_homo"] += 1
            if "span_level" in label and "prot_site" in label["span_level"] and label["span_level"]["prot_site"] and len(label["span_level"]["prot_site"]) > 0:
                site_set = set()
                for item in label["span_level"]["prot_site"]:
                    site_idx = item[-1]
                    site_set.add(site_idx)
                    buffer["prot_site_set"].add(site_idx)
                if len(site_set) > 0:
                    buffer["prot_site"] += 1

            if "span_level" in label and "prot_domain" in label["span_level"] and label["span_level"]["prot_domain"] and len(label["span_level"]["prot_domain"]) > 0:
                domain_set = set()
                for item in label["span_level"]["prot_domain"]:
                    domain_idx = item[-1]
                    domain_set.add(domain_idx)
                    buffer["prot_domain_set"].add(domain_idx)
                if len(domain_set) > 0:
                    buffer["prot_domain"] += 1
            if "structure_level" in label and "prot_structure" in label["structure_level"] and label["structure_level"]["prot_structure"] and len(label["structure_level"]["prot_structure"]) > 0:
                local_x = [10000000, -10000000]
                local_y = [10000000, -10000000]
                local_z = [10000000, -10000000]
                structure_x_set = set()
                structure_y_set = set()
                structure_z_set = set()
                for coord in label["structure_level"]["prot_structure"]:
                    if coord == -1 or (coord[0] == -100 and coord[1] == -100 and coord[2] == -100):
                        continue
                    x, y, z = coord[0], coord[1], coord[2]
                    if local_x[0] > x:
                        local_x[0] = x
                    if local_x[1] < x:
                        local_x[1] = x
                    if local_y[0] > y:
                        local_y[0] = y
                    if local_y[1] < y:
                        local_y[1] = y
                    if local_z[0] > z:
                        local_z[0] = z
                    if local_z[1] < z:
                        local_z[1] = z
                for coord in label["structure_level"]["prot_structure"]:
                    if coord == -1 or (coord[0] == -100 and coord[1] == -100 and coord[2] == -100):
                        continue
                    new_coord = [coord[0], coord[1], coord[2]]
                    if local_x[0] == local_x[1]:
                        new_coord[0] = 1.0
                    else:
                        new_coord[0] = (new_coord[0] - local_x[0])/(local_x[1] - local_x[0])
                    if local_y[0] == local_y[1]:
                        new_coord[1] = 1.0
                    else:
                        new_coord[1] = (new_coord[1] - local_y[0])/(local_y[1] - local_y[0])
                    if local_z[0] == local_z[1]:
                        new_coord[2] = 1.0
                    else:
                        new_coord[2] = (new_coord[2] - local_z[0])/(local_z[1] - local_z[0])
                    new_coord = [str(round(v, 4)) for v in new_coord]
                    structure_x_set.add(new_coord[0])
                    structure_y_set.add(new_coord[1])
                    structure_z_set.add(new_coord[2])

                if len(structure_x_set) > 0 or len(structure_y_set) > 0 or len(structure_z_set) > 0:
                    buffer["prot_structure"] += 1

    def merge(self, buffer, pbuffer):
        for item in pbuffer.items():
            if item[0] == "max_len":
                if item[1] > buffer[item[0]]:
                    buffer[item[0]] = item[1]
            elif item[0] == "min_len":
                if item[1] < buffer[item[0]]:
                    buffer[item[0]] = item[1]
            elif "_set" in item[0]:
                buffer[item[0]] = buffer[item[0]] | item[1]
            else:
                buffer[item[0]] += item[1]

    def terminate(self, buffer):
        if buffer["t"] > 0:
            buffer["avg_len"] /= buffer["t"]
        new_buffer = {}
        for item in buffer.items():
            if "_set" in item[0]:
                new_buffer[item[0]] = len(item[1])
            else:
                new_buffer[item[0]] = item[1]

        return json.dumps(new_buffer, ensure_ascii=False)