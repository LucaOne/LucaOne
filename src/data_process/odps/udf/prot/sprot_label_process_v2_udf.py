#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/5/16 15:19
@project: LucaOne
@file: sprot_label_process_v2_udf.py
@desc: sprot label process v2 udf
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


@annotate('string,string,string,string,string,string,bigint,bigint->string')
class sprot_label_process(BaseUDAF):
    def __init__(self):
        workaround_argv_issue("sprot_label_process_v2_udf.py")
        sys.path.insert(0, 'work/biopython-1.80-cp37.zip')

        protein_homo_span_level_label = {}
        for line in get_cache_file("prot_homo_span_level_label_v2.txt"):
            protein_homo_span_level_label[line.strip()] = len(protein_homo_span_level_label)
        self.protein_homo_span_level_label = protein_homo_span_level_label

        protein_site_span_level_label = {}
        for line in get_cache_file("prot_site_span_level_label_v2.txt"):
            protein_site_span_level_label[line.strip()] = len(protein_site_span_level_label)
        self.protein_site_span_level_label = protein_site_span_level_label

        protein_domain_span_level_label = {}
        for line in get_cache_file("prot_domain_span_level_label_v2.txt"):
            protein_domain_span_level_label[line.strip()] = len(protein_domain_span_level_label)
        self.protein_domain_span_level_label = protein_domain_span_level_label

        protein_taxonomy_seq_level_label = {}
        for line in get_cache_file("prot_taxonomy_seq_level_label_v2.txt"):
            protein_taxonomy_seq_level_label[line.strip()] = len(protein_taxonomy_seq_level_label)
        self.protein_taxonomy_seq_level_label = protein_taxonomy_seq_level_label

        protein_keyword_seq_level_label = {}
        for line in get_cache_file("prot_keyword_seq_level_label_v2.txt"):
            protein_keyword_seq_level_label[line.strip()] = len(protein_keyword_seq_level_label)
        self.protein_keyword_seq_level_label = protein_keyword_seq_level_label

    def new_buffer(self):
        '''
        {
            "seq_id_xxx": {
                "span_level": {
                    "prot_homo": [[start, end, label_idx],...],
                    "prot_site": [[start, end, label_idx],...],
                    "prot_domain": [[start, end, label_idx],...]
                },
                "seq_level": {
                    "prot_taxid": taxid,
                    "prot_taxonomy": order_bio,
                    "prot_keyword": [w1, w2, ...]
                },
            }
        }
        :return:
        '''
        return {}

    def parse_keywords(self, value, separator_char):
        if value:
            while True:
                idx1 = value.find("{")
                if idx1 == -1:
                    break
                idx2 = value.find("}")
                if idx2 == -1:
                    break
                value = value[0:idx1] + value[idx2+1:]
            strs = value.split(separator_char)
            strs = [v.strip() for v in strs]
            strs = [v for v in strs if len(v) > 0]
            return strs
        return None

    @staticmethod
    def interval_merge(intervals, start_index=0, end_index=1, merge_type="intersection"):
        '''
        区间合并，删除子区间，合并连在一起的
        :param intervals:
        :param start_index:
        :param end_index:
        :param merge_type: 合并类型，intersection：只要有交集就合并， sub: 要是子集才合并； join: 包括首尾相接的， sub-join: 子集或者首尾相接的情况
        :return:
        '''
        sorted_intervals = sorted(intervals, key=lambda x:(x[start_index], -x[end_index]))
        result = []
        for interval in sorted_intervals:
            if result:
                if merge_type == "intersection" and result[-1][end_index] > interval[start_index]:
                    # result中最后一个区间的右值> 新区间的左值，说明两个区间有重叠，这种有交集，但是交集不是首尾相接
                    # 将result中最后一个区间更新为合并之后的新区间
                    result[-1][end_index] = max(result[-1][end_index], interval[end_index])
                elif merge_type == "sub" and result[-1][end_index] >= interval[end_index]:
                    # 要是子集包含
                    result[-1][end_index] = max(result[-1][end_index], interval[end_index])
                elif merge_type == "join" and result[-1][end_index] >= interval[start_index]:
                    # 有交集或者首尾相接的情况
                    result[-1][end_index] = max(result[-1][end_index], interval[end_index])
                elif merge_type == "sub-join" and (result[-1][end_index] == interval[start_index] or result[-1][end_index] >= interval[end_index]):
                    # 子集或者首尾相接的情况
                    result[-1][end_index] = max(result[-1][end_index], interval[end_index])
                else:
                    result.append(interval)
            else:
                result.append(interval)

        return result

    def iterate(self, buffer, seq_id, taxid, order_bio, keywords, prot_feature_name,  prot_feature_type, start_p, end_p):
        if seq_id not in buffer:
            buffer[seq_id] = {}
        if order_bio and len(order_bio) > 0 and order_bio in self.protein_taxonomy_seq_level_label:
            order_bio = self.protein_taxonomy_seq_level_label[order_bio]
        else:
            order_bio = None

        if "seq_level" not in buffer[seq_id] or buffer[seq_id]["seq_level"] is None or len(buffer[seq_id]["seq_level"]) == 0:
            buffer[seq_id]["seq_level"] = {"prot_taxid": taxid, "prot_taxonomy": order_bio}
        else:
            if buffer[seq_id]["seq_level"]["prot_taxid"] is None:
                buffer[seq_id]["seq_level"]["prot_taxid"] = taxid
            if buffer[seq_id]["seq_level"]["prot_taxonomy"] is None:
                buffer[seq_id]["seq_level"]["prot_taxonomy"] = order_bio

        if keywords and len(keywords) > 0:
            keywords = self.parse_keywords(keywords, ";")
            if keywords is not None and len(keywords) > 0:
                keywords = set([self.protein_keyword_seq_level_label[v] for v in keywords])
                if "seq_level" not in buffer[seq_id] or buffer[seq_id]["seq_level"] is None \
                        or len(buffer[seq_id]["seq_level"]) == 0:
                    buffer[seq_id]["seq_level"] = {"prot_keyword": keywords}
                elif "prot_keyword" not in buffer[seq_id]["seq_level"] or buffer[seq_id]["seq_level"]["prot_keyword"] is None \
                        or len(buffer[seq_id]["seq_level"]["prot_keyword"]) == 0:
                    buffer[seq_id]["seq_level"]["prot_keyword"] = keywords
                else:
                    buffer[seq_id]["seq_level"]["prot_keyword"] = buffer[seq_id]["seq_level"]["prot_keyword"].union(keywords)

        if start_p is not None and start_p >= 0 and end_p is not None  and end_p >= 0 and prot_feature_type in ["Homologous_superfamily", "Site", "Domain"]:
            if "span_level" not in buffer[seq_id] or buffer[seq_id]["span_level"] is None:
                buffer[seq_id]["span_level"] = {}
            if prot_feature_type == "Homologous_superfamily":
                prot_feature_type = "prot_homo"
                prot_feature_label_idx = self.protein_homo_span_level_label[prot_feature_name]
            elif prot_feature_type == "Site":
                prot_feature_type = "prot_site"
                '''
                if prot_feature_name is None:
                    prot_feature_name = "Other"
                '''
                if prot_feature_name is not None:
                    prot_feature_label_idx = self.protein_site_span_level_label[prot_feature_name]
                else:
                    prot_feature_label_idx = None
            elif prot_feature_type == "Domain":
                prot_feature_type = "prot_domain"
                if prot_feature_name is not None:
                    prot_feature_label_idx = self.protein_domain_span_level_label[prot_feature_name]
                else:
                    prot_feature_label_idx = None
            else:
                prot_feature_label_idx = None
            if prot_feature_label_idx is not None:
                if prot_feature_type not in buffer[seq_id]["span_level"] or buffer[seq_id]["span_level"][prot_feature_type] is None:
                    buffer[seq_id]["span_level"][prot_feature_type] = []
                buffer[seq_id]["span_level"][prot_feature_type].append([start_p, end_p - 1, prot_feature_label_idx])

    def merge(self, buffer, pbuffer):
        '''
        {
            "seq_id_xxx": {
                "span_level": {
                    "prot_homo": [[start, end, label_idx],...],
                    "prot_site": [[start, end, label_idx],...],
                    "prot_domain": [[start, end, label_idx],...]
                },
                "seq_level": {
                    "prot_taxid": taxid,
                    "prot_taxonomy": order_bio,
                    "prot_keyword": [w1, w2, ...]
                },
            }
        }
        :param buffer:
        :param pbuffer:
        :return:
        '''
        for item1 in pbuffer.items():
            seq_id = item1[0]
            span_level = item1[1]["span_level"] if "span_level" in item1[1] else None
            seq_level = item1[1]["seq_level"] if "seq_level" in item1[1] else None
            if seq_id not in buffer:
                buffer[seq_id] = {}
            if span_level:
                if "span_level" not in buffer[seq_id]:
                    buffer[seq_id]["span_level"] = {}
                for item2 in span_level.items():
                    prot_feature_type = item2[0]
                    if prot_feature_type not in buffer[seq_id]["span_level"]:
                        buffer[seq_id]["span_level"][prot_feature_type] = []
                    buffer[seq_id]["span_level"][prot_feature_type].extend(item2[1])
            if seq_level:
                if "seq_level" not in buffer[seq_id]:
                    buffer[seq_id]["seq_level"] = {}
                for item2 in seq_level.items():
                    prot_feature_type = item2[0]
                    if prot_feature_type not in buffer[seq_id]["seq_level"] or buffer[seq_id]["seq_level"][prot_feature_type] is None:
                        buffer[seq_id]["seq_level"][prot_feature_type] = item2[1]
                    elif prot_feature_type == "prot_keyword":
                        buffer[seq_id]["seq_level"][prot_feature_type] = buffer[seq_id]["seq_level"][prot_feature_type].union(item2[1])

    def terminate(self, buffer):
        new_buffer = copy.deepcopy(buffer)
        for item1 in buffer.items():
            seq_id = item1[0]
            span_level = item1[1]["span_level"] if "span_level" in item1[1] else None
            if span_level:
                for item2 in span_level.items():
                    prot_feature_type = item2[0]
                    # 同类span，存在join的区间进行合并
                    cur_prot_feature_type_spans = {}
                    for span in item2[1]:
                        span_type = span[2]
                        if span_type not in cur_prot_feature_type_spans:
                            cur_prot_feature_type_spans[span_type] = []
                        cur_prot_feature_type_spans[span_type].append(span)

                    prot_feature_type_spans = []
                    for v in cur_prot_feature_type_spans.items():
                        prot_feature_type_spans.extend(self.interval_merge(v[1], start_index=0, end_index=1, merge_type="join"))
                    new_buffer[seq_id]["span_level"][prot_feature_type] = prot_feature_type_spans
            seq_level = item1[1]["seq_level"] if "seq_level" in item1[1] else None
            if seq_level and "prot_keyword" in seq_level:
                new_buffer[seq_id]["seq_level"]["prot_keyword"] = list(seq_level["prot_keyword"])
        return json.dumps(new_buffer, ensure_ascii=False)