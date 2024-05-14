#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2024/2/28 10:07
@project: LucaOne
@file: data_stats_v2.py
@desc: data stats v2
RNA:
lucaone_data.tmp_lucaone_v2_data_gene2_01
DNA:
lucaone_data.tmp_lucaone_v2_data_gene2_02
prot:
uniref: lucaone_data.tmp_lucaone_v2_data_prot_01
uniprot: lucaone_data.tmp_lucaone_v2_data_prot_02
colab: lucaone_data.tmp_lucaone_v2_data_prot_03

add table tmp_lucaone_v2_taxid_mapping as lucaone_v2_taxid_mapping_resource -f;
'''

import sys, copy, json
from odps.udf import annotate
from odps.udf import BaseUDAF
from odps.distcache import get_cache_file
from odps.distcache import get_cache_table

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


@annotate('string,string,string,string,string,string,string->string')
class data_stats_v2(BaseUDAF):
    def __init__(self):
        gene_type_span_level_label = {}
        for line in get_cache_file("gene_type_span_level_label_v2.txt"):
            gene_type_span_level_label[len(gene_type_span_level_label)] = line.strip()
        self.gene_type_span_level_label = gene_type_span_level_label

        gene_taxonomy_seq_level_label = {}
        for line in get_cache_file("gene_taxonomy_seq_level_label_v2.txt"):
            gene_taxonomy_seq_level_label[len(gene_taxonomy_seq_level_label)] = line.strip()
        self.gene_taxonomy_seq_level_label = gene_taxonomy_seq_level_label

        protein_homo_span_level_label = {}
        for line in get_cache_file("prot_homo_span_level_label_v2.txt"):
            protein_homo_span_level_label[len(protein_homo_span_level_label)] = line.strip()
        self.protein_homo_span_level_label = protein_homo_span_level_label

        protein_site_span_level_label = {}
        for line in get_cache_file("prot_site_span_level_label_v2.txt"):
            protein_site_span_level_label[len(protein_site_span_level_label)] = line.strip()
        self.protein_site_span_level_label = protein_site_span_level_label

        protein_domain_span_level_label = {}
        for line in get_cache_file("prot_domain_span_level_label_v2.txt"):
            protein_domain_span_level_label[len(protein_domain_span_level_label)] = line.strip()
        self.protein_domain_span_level_label = protein_domain_span_level_label

        protein_taxonomy_seq_level_label = {}
        for line in get_cache_file("prot_taxonomy_seq_level_label_v2.txt"):
            protein_taxonomy_seq_level_label[len(protein_taxonomy_seq_level_label)] = line.strip()
        self.protein_taxonomy_seq_level_label = protein_taxonomy_seq_level_label

        protein_keyword_seq_level_label = {}
        for line in get_cache_file("prot_keyword_seq_level_label_v2.txt"):
            protein_keyword_seq_level_label[len(protein_keyword_seq_level_label)] = line.strip()
        self.protein_keyword_seq_level_label = protein_keyword_seq_level_label

        taxid_mapping = {}
        for row in get_cache_table("lucaone_v2_taxid_mapping_resource"):
            taxid = row[0]
            superkingdom = row[1]
            phylum = row[2]
            class_name = row[3]
            order = row[4]
            family = row[5]
            genus = row[6]
            species = row[7]
            taxid_mapping[taxid] = [superkingdom, phylum, class_name, order, family, genus, species]
        self.taxid_mapping = taxid_mapping
        self.tax_level_name_list = ["superkingdom", "phylum", "class_name", "order", "family", "genus", "species"]
        self.selected_seq_type = None
        self.selected_feature_type = None
        self.selected_feature_type_name = None



    def new_buffer(self):
        return {
            "gene": {
                "tax": {}, # 物种信息
                "order": {}, # order
                "gene_type": {}, # gene_type
                "gene_type_repeat": {}, # gene_type
                "order_exists_num": 0,
                "gene_type_exists_num": 0,
                "seq_len": {}, # 序列长度
                "token": {}, # token
                "seq_num": 0,
                "token_num": 0
            },
            "prot": {
                "tax": {}, # 物种信息
                "order": {}, # order
                "keyword": {}, # keyword
                "site": {}, # site
                "site_repeat": {}, # site
                "domain": {}, # domain
                "domain_repeat": {}, # site
                "homo": {}, # homo
                "homo_repeat": {}, # site
                "structure_x": {}, # structure
                "structure_repeat_x": {}, # structure
                "structure_y": {}, # structure
                "structure_repeat_y": {}, # structure
                "structure_z": {}, # structure
                "structure_repeat_z": {}, # structure
                "order_exists_num": 0,
                "keyword_exists_num": 0,
                "site_exists_num": 0,
                "domain_exists_num": 0,
                "homo_exists_num": 0,
                "structure_exists_num": 0,
                "seq_len": {}, # 序列长度
                "token": {}, # token
                "seq_num": 0,
                "token_num": 0
            }
        }

    def iterate(self, buffer, seq_id, seq_type, seq, label, selected_seq_type=None, selected_feature_type=None, selected_feature_type_name=None):
        self.selected_seq_type = selected_seq_type
        self.selected_feature_type = selected_feature_type
        self.selected_feature_type_name = selected_feature_type_name
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
        if seq_type == "prot":
            cur_buffer = buffer["prot"]
        else:
            cur_buffer = buffer["gene"]
        # add [CLS] and [SEP]
        seq_len = len(seq) + 2
        if selected_feature_type is None or selected_feature_type == "seq":
            cur_buffer["seq_num"] += 1
            if seq_len not in cur_buffer["seq_len"]:
                cur_buffer["seq_len"][seq_len] = 0
            cur_buffer["seq_len"][seq_len] += 1
        if selected_feature_type is None or selected_feature_type == "token":
            cur_buffer["token_num"] += (seq_len - 2)
            for token in seq:
                if token not in cur_buffer["token"]:
                    cur_buffer["token"][token] = 0
                cur_buffer["token"][token] += 1
        label = json.loads(label, encoding="UTF-8")
        if seq_type == "gene" and (selected_seq_type is None or selected_seq_type == "gene"):

            if (selected_feature_type is None or selected_feature_type == "gene_type") and "span_level" in label and "gene_type" in label["span_level"] and label["span_level"]["gene_type"] and len(label["span_level"]["gene_type"]) > 0:
                gene_type_sets = set()
                for item in label["span_level"]["gene_type"]:
                    gene_type_idx = item[-1]
                    gene_type_name = self.gene_type_span_level_label[gene_type_idx]
                    gene_type_sets.add(gene_type_name)
                    if gene_type_name not in cur_buffer["gene_type_repeat"]:
                        cur_buffer["gene_type_repeat"][gene_type_name] = 0
                    cur_buffer["gene_type_repeat"][gene_type_name] += 1
                if len(gene_type_sets) > 0:
                    cur_buffer["gene_type_exists_num"] += 1
                    for v in gene_type_sets:
                        if v not in cur_buffer["gene_type"]:
                            cur_buffer["gene_type"][v] = 0
                        cur_buffer["gene_type"][v] += 1

            if (selected_feature_type is None or selected_feature_type == "taxonomy") and "seq_level" in label and "gene_taxonomy" in label["seq_level"] and label["seq_level"]["gene_taxonomy"] and len(str(label["seq_level"]["gene_taxonomy"])) > 0:
                order_name = self.gene_taxonomy_seq_level_label[label["seq_level"]["gene_taxonomy"]]
                if order_name:
                    cur_buffer["order_exists_num"] += 1
                    if order_name not in cur_buffer["order"]:
                        cur_buffer["order"][order_name] = 0
                    cur_buffer["order"][order_name] += 1

            if (selected_feature_type is None or selected_feature_type == "tax") and "seq_level" in label and "gene_taxid" in label["seq_level"] and label["seq_level"]["gene_taxid"]:
                tax_id = label["seq_level"]["gene_taxid"]
                if tax_id and tax_id in self.taxid_mapping:
                    if selected_feature_type_name is None :
                        selected_feature_type_name = "species"
                    selected_feature_type_idx = self.tax_level_name_list.index(selected_feature_type_name)
                    tax_name = self.taxid_mapping[tax_id][selected_feature_type_idx]
                    if tax_name:
                        if tax_name not in cur_buffer["tax"]:
                            cur_buffer["tax"][tax_name] = 0
                        cur_buffer["tax"][tax_name] += 1
                        k = "%s_exists" % selected_feature_type_name
                        if k not in cur_buffer:
                            cur_buffer[k] = 0
                        cur_buffer[k] += 1

        elif seq_type == "prot" and (selected_seq_type is None or selected_seq_type == "prot"):
            if (selected_feature_type is None or selected_feature_type == "taxonomy") and "seq_level" in label and "prot_taxonomy" in label["seq_level"] and label["seq_level"]["prot_taxonomy"] and len(str(label["seq_level"]["prot_taxonomy"])) > 0:
                order_name = self.protein_taxonomy_seq_level_label[label["seq_level"]["prot_taxonomy"]]
                if order_name:
                    cur_buffer["order_exists_num"] += 1
                    if order_name not in cur_buffer["order"]:
                        cur_buffer["order"][order_name] = 0
                    cur_buffer["order"][order_name] += 1

            if (selected_feature_type is None or selected_feature_type == "tax") and "seq_level" in label and "prot_taxid" in label["seq_level"] and label["seq_level"]["prot_taxid"]:
                tax_id = label["seq_level"]["prot_taxid"]
                if tax_id and tax_id in self.taxid_mapping:
                    if selected_feature_type_name is None :
                        selected_feature_type_name = "species"
                    selected_feature_type_idx = self.tax_level_name_list.index(selected_feature_type_name)
                    tax_name = self.taxid_mapping[tax_id][selected_feature_type_idx]
                    if tax_name:
                        if tax_name not in cur_buffer["tax"]:
                            cur_buffer["tax"][tax_name] = 0
                        cur_buffer["tax"][tax_name] += 1
                        k = "%s_exists" % selected_feature_type_name
                        if k not in cur_buffer:
                            cur_buffer[k] = 0
                        cur_buffer[k] += 1

            if (selected_feature_type is None or selected_feature_type == "keyword") and "seq_level" in label and "prot_keyword" in label["seq_level"] and label["seq_level"]["prot_keyword"] and len(label["seq_level"]["prot_keyword"]) > 0:
                keyword_set = set()
                for keyword_idx in label["seq_level"]["prot_keyword"]:
                    keyword_name = self.protein_keyword_seq_level_label[keyword_idx]
                    keyword_set.add(keyword_name)
                    if keyword_name not in cur_buffer["keyword"]:
                        cur_buffer["keyword"][keyword_name] = 0
                    cur_buffer["keyword"][keyword_name] += 1
                if len(keyword_set) > 0:
                    cur_buffer["keyword_exists_num"] += 1

            if (selected_feature_type is None or selected_feature_type == "homo") and "span_level" in label and "prot_homo" in label["span_level"] and label["span_level"]["prot_homo"] and len(label["span_level"]["prot_homo"]) > 0:
                homo_name_set = set()
                for item in label["span_level"]["prot_homo"]:
                    homo_idx = item[-1]
                    homo_name = self.protein_homo_span_level_label[homo_idx]
                    homo_name_set.add(homo_name)
                    if homo_name not in cur_buffer["homo_repeat"]:
                        cur_buffer["homo_repeat"][homo_name] = 0
                    cur_buffer["homo_repeat"][homo_name] += 1
                if len(homo_name_set) > 0:
                    cur_buffer["homo_exists_num"] += 1
                    for v in homo_name_set:
                        if v not in cur_buffer["homo"]:
                            cur_buffer["homo"][v] = 0
                        cur_buffer["homo"][v] += 1

            if (selected_feature_type is None or selected_feature_type == "site") and "span_level" in label and "prot_site" in label["span_level"] and label["span_level"]["prot_site"] and len(label["span_level"]["prot_site"]) > 0:
                site_name_set = set()
                for item in label["span_level"]["prot_site"]:
                    site_idx = item[-1]
                    site_name = self.protein_site_span_level_label[site_idx]
                    site_name_set.add(site_name)
                    if site_name not in cur_buffer["site_repeat"]:
                        cur_buffer["site_repeat"][site_name] = 0
                    cur_buffer["site_repeat"][site_name] += 1
                if len(site_name_set) > 0:
                    cur_buffer["site_exists_num"] += 1
                    for v in site_name_set:
                        if v not in cur_buffer["site"]:
                            cur_buffer["site"][v] = 0
                        cur_buffer["site"][v] += 1

            if (selected_feature_type is None or selected_feature_type == "domain") and "span_level" in label and "prot_domain" in label["span_level"] and label["span_level"]["prot_domain"] and len(label["span_level"]["prot_domain"]) > 0:
                domain_name_set = set()
                for item in label["span_level"]["prot_domain"]:
                    domain_idx = item[-1]
                    domain_name = self.protein_domain_span_level_label[domain_idx]
                    domain_name_set.add(domain_name)
                    if domain_name not in cur_buffer["domain_repeat"]:
                        cur_buffer["domain_repeat"][domain_name] = 0
                    cur_buffer["domain_repeat"][domain_name] += 1
                if len(domain_name_set) > 0:
                    cur_buffer["domain_exists_num"] += 1
                    for v in domain_name_set:
                        if v not in cur_buffer["domain"]:
                            cur_buffer["domain"][v] = 0
                        cur_buffer["domain"][v] += 1

            if (selected_feature_type is None or selected_feature_type == "structure") and "structure_level" in label and "prot_structure" in label["structure_level"] and label["structure_level"]["prot_structure"] and len(label["structure_level"]["prot_structure"]) > 0:

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

                    if new_coord[0] not in cur_buffer["structure_repeat_x"]:
                        cur_buffer["structure_repeat_x"][new_coord[0]] = 0
                    cur_buffer["structure_repeat_x"][new_coord[0]] += 1

                    if new_coord[1] not in cur_buffer["structure_repeat_y"]:
                        cur_buffer["structure_repeat_y"][new_coord[1]] = 0
                    cur_buffer["structure_repeat_y"][new_coord[1]] += 1

                    if new_coord[2] not in cur_buffer["structure_repeat_z"]:
                        cur_buffer["structure_repeat_z"][new_coord[2]] = 0
                    cur_buffer["structure_repeat_z"][new_coord[2]] += 1
                    structure_x_set.add(new_coord[0])
                    structure_y_set.add(new_coord[1])
                    structure_z_set.add(new_coord[2])

                if structure_x_set:
                    for v in structure_x_set:
                        if v not in cur_buffer["structure_x"]:
                            cur_buffer["structure_x"][v] = 0
                        cur_buffer["structure_x"][v] += 1

                if structure_y_set:
                    for v in structure_y_set:
                        if v not in cur_buffer["structure_y"]:
                            cur_buffer["structure_y"][v] = 0
                        cur_buffer["structure_y"][v] += 1

                if structure_z_set:
                    for v in structure_z_set:
                        if v not in cur_buffer["structure_z"]:
                            cur_buffer["structure_z"][v] = 0
                        cur_buffer["structure_z"][v] += 1

                if len(structure_x_set) > 0 or len(structure_y_set) > 0 or len(structure_z_set) > 0:
                    cur_buffer["structure_exists_num"] += 1

    def merge(self, buffer, pbuffer):
        for item in pbuffer.items():
            if item[0] == "gene":
                if "gene" not in buffer:
                    buffer["gene"] = {}
                cur_pbuffer = pbuffer["gene"]
                cur_buffer = buffer["gene"]
                keys_1 = ["tax", "order", "gene_type", "gene_type_repeat", "seq_len", "token"]
                keys_2 = ["%s_exists" % self.selected_feature_type_name, "order_exists_num", "gene_type_exists_num", "seq_num", "token_num"]
            else:
                if "prot" not in buffer:
                    buffer["prot"] = {}
                cur_pbuffer = pbuffer["prot"]
                cur_buffer = buffer["prot"]
                keys_1 = ["tax", "order", "keyword", "site", "site_repeat", "domain", "domain_repeat", "homo", "homo_repeat",
                          "structure_x", "structure_repeat_x", "structure_y", "structure_repeat_y", "structure_z", "structure_repeat_z",
                          "seq_len", "token"]
                keys_2 = ["%s_exists" % self.selected_feature_type_name, "order_exists_num", "keyword_exists_num", "site_exists_num", "domain_exists_num", "homo_exists_num", "structure_exists_num", "seq_num", "token_num"]
            for key in keys_1:
                if key in cur_pbuffer and cur_pbuffer[key]:
                    if key not in cur_buffer:
                        cur_buffer[key] = {}
                    for item in cur_pbuffer[key].items():
                        k, v = item[0], item[1]
                        if k not in cur_buffer[key]:
                            cur_buffer[key][k] = v
                        else:
                            cur_buffer[key][k] += v
            for key in keys_2:
                if key in cur_pbuffer and cur_pbuffer[key] > 0:
                    if key not in cur_buffer:
                        cur_buffer[key] = 0
                    cur_buffer[key] += cur_pbuffer[key]

    def terminate(self, buffer):
        return json.dumps(buffer, ensure_ascii=False)


'''
data_stats_v2(obj_id, obj_type, obj_seq, obj_label, "gene", "tax", "superkingdom")
data_stats_v2(obj_id, obj_type, obj_seq, obj_label, "gene", "tax", "phylum")
data_stats_v2(obj_id, obj_type, obj_seq, obj_label, "gene", "tax", "class_name")

data_stats_v2(obj_id, obj_type, obj_seq, obj_label, "gene", "seq", NULL)
data_stats_v2(obj_id, obj_type, obj_seq, obj_label, "gene", "token", NULL)
data_stats_v2(obj_id, obj_type, obj_seq, obj_label, "gene", "taxonomy", NULL)
data_stats_v2(obj_id, obj_type, obj_seq, obj_label, "gene", "gene_type", NULL)

data_stats_v2(obj_id, obj_type, obj_seq, obj_label, "prot", "tax", "superkingdom")
data_stats_v2(obj_id, obj_type, obj_seq, obj_label, "prot", "tax", "phylum")
data_stats_v2(obj_id, obj_type, obj_seq, obj_label, "prot", "tax", "class_name")

data_stats_v2(obj_id, obj_type, obj_seq, obj_label, "prot", "seq", NULL)
data_stats_v2(obj_id, obj_type, obj_seq, obj_label, "prot", "token", NULL)
data_stats_v2(obj_id, obj_type, obj_seq, obj_label, "prot", "taxonomy", NULL)
data_stats_v2(obj_id, obj_type, obj_seq, obj_label, "prot", "keyword", NULL)
data_stats_v2(obj_id, obj_type, obj_seq, obj_label, "prot", "site", NULL)
data_stats_v2(obj_id, obj_type, obj_seq, obj_label, "prot", "homo", NULL)
data_stats_v2(obj_id, obj_type, obj_seq, obj_label, "prot", "domain", NULL)
data_stats_v2(obj_id, obj_type, obj_seq, obj_label, "prot", "structure", NULL)

'''

