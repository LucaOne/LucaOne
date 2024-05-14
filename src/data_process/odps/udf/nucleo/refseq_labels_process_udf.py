#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/11/2 12:35
@project: LucaOne
@file: refseq_labels_process_udf.py
@desc: refseq labels process udf
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


@annotate('string,bigint,bigint,bigint,string,string,string,string->string')
class refseq_labels_process(BaseUDAF):
    def __init__(self):
        workaround_argv_issue("refseq_labels_process_udf.py")
        sys.path.insert(0, 'work/biopython-1.80-cp37.zip')
        gene_type_span_level_label_name_2_id = {}
        for line in get_cache_file("gene_type_span_level_label_v2.txt"):
            gene_type_span_level_label_name_2_id[line.strip()] = len(gene_type_span_level_label_name_2_id)
        self.gene_type_span_level_label_name_2_id = gene_type_span_level_label_name_2_id
        gene_taxonomy_seq_level_label = {}
        for line in get_cache_file("gene_taxonomy_seq_level_label_v2.txt"):
            gene_taxonomy_seq_level_label[line.strip()] = len(gene_taxonomy_seq_level_label)
        self.gene_taxonomy_seq_level_label = gene_taxonomy_seq_level_label

    def new_buffer(self):
        '''
        {
            "seq_id_xxx": {
                "span_level": {
                    "gene_type": [[start, end, label_idx],...]
                },
                "seq_level": {
                    "gene_taxid": taxid,
                    "gene_taxonomy": order_bio
                }
            }
        }
        :return:
        '''
        return {}


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


    def judgy_span_intersection(self, span1, span2):
        if span1[0] > span2[0] or span1[0] == span2[0] and span1[1] > span2[1]:
            tmp = span1
            span1 = span2
            span2 = tmp
        if span1[1] >= span2[0]:
            return True
        return False


    def span_intersection(self, span, start, end):
        span_start = span[0]
        span_end  = span[1]
        span_type = span[2]
        if span_start < start:
            span_start = start
        if span_end > end:
            span_end = end
        if span_start <= span_end:
            return [span_start, span_end, span_type]
        return None


    def iterate(self, buffer, seq_id, gene_idx, seq_start_p, seq_end_p, feature_type, start_end, taxid, order_bio):
        from Bio.SeqFeature import SeqFeature, Reference, FeatureLocation, CompoundLocation, SimpleLocation, ExactPosition, \
            UnknownPosition, UncertainPosition, WithinPosition, BetweenPosition, BeforePosition, AfterPosition, OneOfPosition, PositionGap

        if feature_type == "exon":
            feature_type = "CDS"
        elif feature_type == "intron_manual":
            feature_type = "intron"
        elif feature_type in ("3'UTR", "5'UTR"):
            feature_type = "regulatory"
        seq_id = seq_id + "#" + str(gene_idx)
        if seq_id not in buffer:
            buffer[seq_id] = {}

        span_level = []
        if start_end:
            feature_type_id = self.gene_type_span_level_label_name_2_id[feature_type]
            start_end = eval(start_end)
            for pt in start_end.parts:
                start = int(pt.start)
                end = int(pt.end) - 1
                if not self.judgy_span_intersection([seq_start_p, seq_end_p - 1], [start, end]):
                    continue
                cur_start = max(start, seq_start_p)
                cur_end = min(end, seq_end_p - 1)
                span_level.append([cur_start - seq_start_p, cur_end - seq_start_p, feature_type_id])

        taxonomy_id = None
        if order_bio:
            taxonomy_id = self.gene_taxonomy_seq_level_label[order_bio]
        if len(span_level) > 1:
            # 同类span，存在join的区间进行合并
            tmp_span_level = {}
            for span in span_level:
                span_type = span[2]
                if span_type not in tmp_span_level:
                    tmp_span_level[span_type] = []
                tmp_span_level[span_type].append(span)
            span_level = []
            for v in tmp_span_level.items():
                span_level.extend(self.interval_merge(v[1], start_index=0, end_index=1, merge_type="join"))
        if "span_level" not in buffer[seq_id]:
            buffer[seq_id]["span_level"] = {"gene_type": []}
        if span_level:
            buffer[seq_id]["span_level"]["gene_type"].extend(span_level)
        buffer[seq_id]["seq_level"] = {"gene_taxonomy": taxonomy_id, "gene_taxid": taxid}
        buffer[seq_id]["seq_start_p"] = seq_start_p
        buffer[seq_id]["seq_end_p"] = seq_end_p

    def merge(self, buffer, pbuffer):
        for item in pbuffer.items():
            seq_id = item[0]
            gene_type = item[1]["span_level"]["gene_type"]
            taxonomy = item[1]["seq_level"]["gene_taxonomy"]
            taxid = item[1]["seq_level"]["gene_taxid"]
            seq_start_p = item[1]["seq_start_p"]
            seq_end_p = item[1]["seq_end_p"]
            if seq_id not in buffer:
                buffer[seq_id] = {}
                buffer[seq_id]["span_level"] = {"gene_type": []}
            if gene_type:
                if buffer[seq_id]["span_level"] is None or buffer[seq_id]["span_level"]["gene_type"] is None:
                    buffer[seq_id]["span_level"]["gene_type"] = []
                buffer[seq_id]["span_level"]["gene_type"].extend(gene_type)
            buffer[seq_id]["seq_start_p"] = seq_start_p
            buffer[seq_id]["seq_end_p"] = seq_end_p

            if "seq_level" not in buffer[seq_id] or buffer[seq_id]["seq_level"] is None:
                buffer[seq_id]["seq_level"] = {}
                buffer[seq_id]["seq_level"]["gene_taxonomy"] = taxonomy
                buffer[seq_id]["seq_level"]["gene_taxid"] = taxid
            else:
                if "gene_taxonomy" not in buffer[seq_id]["seq_level"] or buffer[seq_id]["seq_level"]["gene_taxonomy"] is None:
                    buffer[seq_id]["seq_level"]["gene_taxonomy"] = taxonomy
                if "gene_taxid" not in buffer[seq_id]["seq_level"] or buffer[seq_id]["seq_level"]["gene_taxid"] is None:
                    buffer[seq_id]["seq_level"]["gene_taxid"] = taxid

    def terminate(self, buffer):
        new_buffer = copy.deepcopy(buffer)
        for item in buffer.items():
            seq_id = item[0]
            gene_type_list = item[1]["span_level"]["gene_type"]
            seq_start_p = item[1]["seq_start_p"]
            seq_end_p = item[1]["seq_end_p"] - 1

            if gene_type_list and len(gene_type_list) > 0:
                # 同类span，存在join的区间进行合并
                tmp_gene_type_list = {}
                for span in gene_type_list:
                    span_type = span[2]
                    if span_type not in tmp_gene_type_list:
                        tmp_gene_type_list[span_type] = []
                    tmp_gene_type_list[span_type].append(span)

                gene_type_list = []
                for v in tmp_gene_type_list.items():
                    gene_type_list.extend(self.interval_merge(v[1], start_index=0, end_index=1, merge_type="join"))
                '''
                new_gene_type_list = []
                for span in gene_type_list:
                    new_span = self.span_intersection(span, seq_start_p, seq_end_p)
                    if new_span:
                        new_gene_type_list.append(new_span)
                '''
                if len(gene_type_list) > 0:
                    new_buffer[seq_id]["span_level"]["gene_type"] = gene_type_list
                else:
                    new_buffer[seq_id]["span_level"]["gene_type"] = None
            return json.dumps(new_buffer[seq_id], ensure_ascii=False)
        # return json.dumps(new_buffer, ensure_ascii=False)
