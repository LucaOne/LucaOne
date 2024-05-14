#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/5/16 15:19
@project: LucaOne
@file: refseq_nucleo_split_udaf_c2.py
@desc: refseq nucleo split udaf c2
'''

import sys, copy, json
from odps.udf import annotate
from odps.udf import BaseUDAF


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


@annotate('string,string,string,string,bigint,bigint->string')
class refseq_nucleo_split(BaseUDAF):
    def __init__(self):
        workaround_argv_issue("refseq_nucleo_split_udaf_c2.py.py")
        sys.path.insert(0, 'work/biopython-1.80-cp37.zip')

    def new_buffer(self):
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

    def iterate(self, buffer, seq_id, molecule_type, feature_type, start_end, seq_len, max_len):
        from Bio.SeqFeature import SeqFeature, Reference, FeatureLocation, CompoundLocation, SimpleLocation, ExactPosition, UnknownPosition, UncertainPosition, WithinPosition, BetweenPosition, BeforePosition, AfterPosition, OneOfPosition, PositionGap
        buffer["max_len"] = max_len
        buffer["molecule_type"] = molecule_type
        buffer["seq_len"] = seq_len

        # 正链id
        forward_seq_id = seq_id
        # 正链基因区间
        forward_gene = []
        # 正链重要区间
        forward_span_level = []
        # 来源的seq_id
        forward_ref_seq_id = seq_id

        # 负链seq_id，正链id + "_r_"
        reverse_seq_id = None
        # 负链基因区间
        reverse_gene = []
        # 负链重要区间
        reverse_span_level = []
        # 来源的seq_id
        reverse_ref_seq_id = None

        if feature_type == "exon":
            feature_type = "CDS"
        elif feature_type == "intron_manual":
            feature_type = "intron"

        feature_type_id = feature_type

        if start_end:
            start_end = eval(start_end)
            for pt in start_end.parts:
                start = int(pt.start)
                end = int(pt.end) - 1
                if pt.strand is None:
                    cur_strand = 1
                else:
                    cur_strand = int(pt.strand)
                if feature_type_id == "gene":
                    # 负链基因区间
                    if cur_strand < 0:
                        reverse_gene.append([start, end, feature_type_id])
                    else:
                        forward_gene.append([start, end, feature_type_id])
                else:
                    # 正链重要区间
                    if cur_strand < 0:
                        reverse_span_level.append([start, end, feature_type_id])
                    else:
                        forward_span_level.append([start, end, feature_type_id])

        if len(reverse_span_level) > 0 or len(reverse_gene) > 0:
            reverse_seq_id = forward_seq_id + "_r_"
            reverse_ref_seq_id = seq_id

        if forward_seq_id not in buffer:
            buffer[forward_seq_id] = {}

        if len(forward_span_level) > 0:
            # 同类span，存在join的区间进行合并
            tmp_forward_span_level = {}
            for span in forward_span_level:
                span_type = span[2]
                if span_type not in tmp_forward_span_level:
                    tmp_forward_span_level[span_type] = []
                tmp_forward_span_level[span_type].append(span)
            forward_span_level = []
            for v in tmp_forward_span_level.items():
                forward_span_level.extend(self.interval_merge(v[1], start_index=0, end_index=1, merge_type="join"))
            if forward_seq_id not in buffer:
                buffer[forward_seq_id] = {}
            if "span_level" not in buffer[forward_seq_id]:
                buffer[forward_seq_id]["span_level"] = []
            if forward_span_level:
                buffer[forward_seq_id]["span_level"].extend(forward_span_level)
            buffer[forward_seq_id]["ref_seq_id"] = forward_ref_seq_id

        if reverse_seq_id is not None:
            if len(reverse_span_level) > 0:
                # 同类span，存在join的区间进行合并
                tmp_reverse_span_level = {}
                for span in reverse_span_level:
                    span_type = span[2]
                    if span_type not in tmp_reverse_span_level:
                        tmp_reverse_span_level[span_type] = []
                    tmp_reverse_span_level[span_type].append(span)
                reverse_span_level = []
                for v in tmp_reverse_span_level.items():
                    reverse_span_level.extend(self.interval_merge(v[1], start_index=0, end_index=1, merge_type="join"))
            if reverse_seq_id not in buffer:
                buffer[reverse_seq_id] = {}
            if "span_level" not in buffer[reverse_seq_id]:
                buffer[reverse_seq_id]["span_level"] = []
            if reverse_span_level:
                buffer[reverse_seq_id]["span_level"].extend(reverse_span_level)
            buffer[reverse_seq_id]["ref_seq_id"] = reverse_ref_seq_id

        if len(forward_gene) > 0:
            # gene的sub-join的区间进行合并
            tmp_forward_gene = {}
            for span in forward_gene:
                span_type = span[2]
                if span_type not in tmp_forward_gene:
                    tmp_forward_gene[span_type] = []
                tmp_forward_gene[span_type].append(span)
            forward_gene = []
            for v in tmp_forward_gene.items():
                forward_gene.extend(self.interval_merge(v[1], start_index=0, end_index=1, merge_type="sub-join"))

            if forward_seq_id not in buffer:
                buffer[forward_seq_id] = {}
            if "gene" not in buffer[forward_seq_id]:
                buffer[forward_seq_id]["gene"] = []
            if forward_gene:
                buffer[forward_seq_id]["gene"].extend(forward_gene)

        if len(reverse_gene) > 0:
            # gene的sub-join的区间进行合并
            tmp_reverse_gene = {}
            for span in reverse_gene:
                span_type = span[2]
                if span_type not in tmp_reverse_gene:
                    tmp_reverse_gene[span_type] = []
                tmp_reverse_gene[span_type].append(span)
            reverse_gene = []
            for v in tmp_reverse_gene.items():
                reverse_gene.extend(self.interval_merge(v[1], start_index=0, end_index=1, merge_type="sub-join"))

            if reverse_seq_id not in buffer:
                buffer[reverse_seq_id] = {}
            if "gene" not in buffer[reverse_seq_id]:
                buffer[reverse_seq_id]["gene"] = []
            if reverse_gene:
                buffer[reverse_seq_id]["gene"].extend(reverse_gene)

    def merge(self, buffer, pbuffer):
        max_len = pbuffer["max_len"]
        molecule_type = pbuffer["molecule_type"]
        seq_len = pbuffer["seq_len"]

        for item in pbuffer.items():
            seq_id = item[0]
            if seq_id in ["max_len", "molecule_type", "seq_len"]:
                continue
            if seq_id not in buffer:
                buffer[seq_id] = {}
                buffer[seq_id]["span_level"] = []
                buffer[seq_id]["gene"] = []
            if "span_level" in item[1]:
                span_level = item[1]["span_level"]
                if buffer[seq_id]["span_level"] is None:
                    buffer[seq_id]["span_level"] = []
                buffer[seq_id]["span_level"].extend(span_level)
            if "gene" in item[1]:
                gene = item[1]["gene"]
                if buffer[seq_id]["gene"] is None:
                    buffer[seq_id]["gene"] = []
                buffer[seq_id]["gene"].extend(gene)
            if "ref_seq_id" in item[1]:
                ref_seq_id = item[1]["ref_seq_id"]
                buffer[seq_id]["ref_seq_id"] = ref_seq_id
        buffer["max_len"] = max_len
        buffer["molecule_type"] = molecule_type
        buffer["seq_len"] = seq_len


    def terminate(self, buffer):
        max_len = buffer["max_len"]
        seq_len = buffer["seq_len"]
        molecule_type = buffer["molecule_type"]
        results = []
        if molecule_type in ["rRNA", "mRNA"]:
            if seq_len <= max_len:
                results.append([1, 0, seq_len, "r", 0, 6])
            else:
                split_num = (seq_len + max_len - 1)//max_len
                for split_idx in range(split_num):
                    begin = split_idx * max_len
                    end = begin + max_len
                    if end > seq_len:
                        results.append([1, max(0, seq_len - max_len), seq_len, "r", split_idx + 1, 8])
                    else:
                        results.append([1, begin, end, "r", split_idx + 1, 7])
            return json.dumps(results, ensure_ascii=False)

        for item in buffer.items():
            if item[0] in ["max_len", "molecule_type", "seq_len"]:
                continue
            seq_id = item[0]
            strand = 1
            if seq_id.endswith("_r_"):
                strand = -1
            span_level = None
            if "span_level" in item[1]:
                span_level = item[1]["span_level"]
            gene = None
            if "gene" in item[1]:
                gene = item[1]["gene"]

            seq_span_type = set()
            if span_level and len(span_level) > 0:
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

                for span in span_level:
                    start, end, feature_type_id = span[0], span[1], span[2]
                    feature_type_value = True if feature_type_id in ['CDS', 'mRNA', 'tRNA', 'ncRNA', 'rRNA', 'misc_RNA', 'tmRNA', 'regulatory'] else False
                    for idx in range(start, end + 1):
                        if feature_type_value:
                            seq_span_type.add(idx)

            if gene and len(gene) > 0:
                return json.dumps(results, ensure_ascii=False)

            seq_split_idx = 0
            if seq_len <= max_len:
                results.append([strand, 0, seq_len, "o", seq_split_idx + 1, 9])
            elif len(seq_span_type) == 0:
                split_num = (seq_len + max_len - 1)//max_len
                for split_idx in range(split_num):
                    begin = split_idx * max_len
                    end = begin + max_len
                    if end > seq_len:
                        results.append([strand, max(0, seq_len - max_len), seq_len, "s", split_idx + 1, 11])
                    else:
                        results.append([strand, begin, end, "s", split_idx + 1, 10])
                return json.dumps(results, ensure_ascii=False)
            else:
                begin = 0
                print("seq_len: %d" % seq_len)
                for idx in range(0, seq_len):
                    # 达到最大长度
                    if idx - begin + 1 == max_len:
                        end = idx
                        if idx not in seq_span_type or idx + 1 >= seq_len:
                            # 该位置是内含子或者是最后一个位置
                            results.append([strand, begin, end + 1, "l", seq_split_idx + 1, 12])
                            begin = end + 1
                            end = end + 1
                            seq_split_idx += 1
                            print("ok1")
                        elif idx in seq_span_type and idx + 1 < seq_len and idx + 1 in seq_span_type:
                            print("ok2")
                            # 该位置是非内含子，并且下一个也是非内含子，说明在重要区间内部，那么需要退回到上一个内含子位置
                            tmp_idx = idx - 1
                            while tmp_idx >= begin and tmp_idx in seq_span_type:
                                print("ok21")
                                tmp_idx -= 1
                            print("ok22")
                            # 没找到上一个内行子，那么重要区间截断，
                            if tmp_idx < begin:
                                results.append([strand, begin, idx + 1, "l", seq_split_idx + 1, 13])
                                begin = idx + 1
                                end = idx + 1
                                seq_split_idx += 1
                                print("ok22")
                            else: # 找到上一个内含子
                                end = tmp_idx
                                print("ok23")
                                if end - begin + 1 < max_len:
                                    print("ok231")
                                    # 向左扩展内行子
                                    new_idx = begin - 1
                                    expand_len = max_len - (end - new_idx + 1)
                                    # 是内行子，并且还可以扩展
                                    while new_idx >= 0 and new_idx not in seq_span_type and expand_len >= 0:
                                        print("ok232")
                                        new_idx -= 1
                                        expand_len -= 1
                                    print("ok233")
                                    results.append([strand, new_idx + 1, end + 1, "l", seq_split_idx + 1, 14])
                                else:
                                    print("ok234")
                                    results.append([strand, begin, end + 1, "l", seq_split_idx + 1, 15])
                                begin = end + 1
                                seq_split_idx += 1
                                end = end + 1
                        else: # 该位置是非内含子，下一个位置是内含子
                            print("ok3")
                            results.append([strand, begin, end + 1, "l", seq_split_idx + 1, 16])
                            begin = end + 1
                            end = end + 1
                            seq_split_idx += 1
                # if idx >= begin and idx - begin + 1 < max_len:
                while begin < seq_len:
                    print("ok4")
                    # 最后一个小于max_len，进行左右扩展
                    # 向左扩展内行子
                    new_begin_idx = begin - 1
                    # new_end_idx = min(seq_len, end + 1)
                    new_end_idx = min(seq_len, begin + max_len)
                    # expand_len = max_len - (end - begin + 1)
                    # expand_len = max_len - (seq_len - begin)
                    expand_len = max_len - (new_end_idx - begin)
                    # 是内行子，并且还可以扩展
                    flag = True
                    while expand_len > 0 and flag:
                        print("ok41")
                        flag = False
                        if new_begin_idx >= 0 and new_begin_idx not in seq_span_type:
                            # 向左扩
                            new_begin_idx -= 1
                            expand_len -= 1
                            flag = True
                            print("ok42")
                        if expand_len > 0 and new_end_idx < seq_len and new_end_idx not in seq_span_type:
                            # 向右扩
                            new_end_idx += 1
                            expand_len -= 1
                            flag = True
                            print("ok43")

                    results.append([strand, new_begin_idx + 1, new_end_idx, "l", seq_split_idx + 1, 17])
                    # results.append([strand, begin, idx + 1, "l", seq_split_idx + 1, 12])
                    seq_split_idx += 1
                    begin = new_end_idx
                    end = new_end_idx

        return json.dumps(results, ensure_ascii=False)
