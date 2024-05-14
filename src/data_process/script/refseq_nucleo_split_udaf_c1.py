#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/5/16 15:19
@project: LucaOne
@file: refseq_nucleo_split_udaf_c1.py
@desc: refseq nucleo split udaf c1
'''
import random
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


@annotate('string,string,string,bigint,bigint->string')
class refseq_nucleo_split(BaseUDAF):
    def __init__(self):
        workaround_argv_issue("refseq_nucleo_split_udaf_c1.py.py")
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

    def iterate(self, buffer, seq_id, feature_type, start_end, seq_len, max_len):
        from Bio.SeqFeature import SeqFeature, Reference, FeatureLocation, CompoundLocation, SimpleLocation, ExactPosition, UnknownPosition, UncertainPosition, WithinPosition, BetweenPosition, BeforePosition, AfterPosition, OneOfPosition, PositionGap
        buffer["max_len"] = max_len
        buffer["seq_len"] = seq_len
        if feature_type in ['CDS', 'mRNA', 'tRNA', 'ncRNA', 'rRNA', 'misc_RNA', 'tmRNA', 'regulatory', 'gene', 'exon', 'intron', 'intron_manual']:
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
        seq_len = pbuffer["seq_len"]
        for item in pbuffer.items():
            seq_id = item[0]
            if seq_id in ["max_len", "seq_len"]:
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
        buffer["seq_len"] = seq_len

    def terminate(self, buffer):
        max_len = buffer["max_len"]
        seq_len = buffer["seq_len"]
        results = []
        for item in buffer.items():
            if item[0] in ["max_len", "seq_len"]:
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

            # 存在有gene片段，但是没有八种元件的信息（22个seq_id)
            '''
            select t1.seq_id, t2.seq_id
            from 
            (
                select distinct seq_id
                from t_nucleo_feature_refseq
                where feature_type in ('gene')
            ) t1
            left outer join
            (
            
                select distinct seq_id
                from t_nucleo_feature_refseq
                where feature_type in ('CDS', 'mRNA', 'tRNA', 'ncRNA', 'rRNA', 'misc_RNA', 'tmRNA', 'regulatory', 'exon', 'intron', 'intron_manual')
            ) t2 
            on t1.seq_id = t2.seq_id
            where t2.seq_id is null;
            '''
            # 存在八种元件的信息, 但是没有gene片段（268个seq_id)
            '''
            select t1.seq_id, t2.seq_id
            from 
            (
                select distinct seq_id
                from t_nucleo_feature_refseq
                where feature_type in ('CDS', 'mRNA', 'tRNA', 'ncRNA', 'rRNA', 'misc_RNA', 'tmRNA', 'regulatory', 'exon', 'intron', 'intron_manual')
            
            ) t1
            left outer join
            (
                select distinct seq_id
                from t_nucleo_feature_refseq
                where feature_type in ('gene')
                
            ) t2 
            on t1.seq_id = t2.seq_id
            where t2.seq_id is null;
            '''
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
                    feature_type_value = False if feature_type_id in ['CDS', 'mRNA', 'tRNA', 'ncRNA', 'rRNA', 'misc_RNA', 'tmRNA', 'regulatory'] else True
                    for idx in range(start, end + 1):
                        if feature_type_value:
                            seq_span_type.add(idx)

            if gene and len(gene) > 0:
                # 同类span，存在sub-join的区间进行合并
                tmp_gene = {}
                for span in gene:
                    span_type = span[2]
                    if span_type not in tmp_gene:
                        tmp_gene[span_type] = []
                    tmp_gene[span_type].append(span)

                gene = []
                for v in tmp_gene.items():
                    gene.extend(self.interval_merge(v[1], start_index=0, end_index=1, merge_type="sub-join"))
            else:
                return json.dumps(results, ensure_ascii=False)

            # 连续/重叠gene，并且合并不足max_len
            gene = self.gene_merge(gene, start_index=0, end_index=1, max_len=min(seq_len, max_len))
            for gene_idx, gene_span in enumerate(gene):
                cur_pre_idx = max(0, gene_span[0])
                cur_last_idx = min(seq_len, gene_span[1] + 1)
                gene_span_len = gene_span[1] - gene_span[0] + 1
                # 基因片段小于最大长度
                if max_len >= gene_span_len:
                    results.append([strand, gene_span[0], gene_span[1] + 1, "o", gene_idx + 1, 0])
                else: # 超过最大长度
                    begin = None
                    # 找到最开始的非内含子
                    for idx in range(gene_span[0], gene_span[1] + 1):
                        if idx not in seq_span_type:
                            begin = idx
                            break
                    if begin is not None:
                        gene_end = gene_span[1]
                        for idx in range(begin, gene_end + 1):
                            if begin in seq_span_type:
                                begin += 1
                                continue
                            # 达到最大长度
                            if idx - begin + 1 == max_len:
                                end = idx
                                if idx in seq_span_type or \
                                        idx not in seq_span_type and idx + 1 >= gene_end + 1:
                                    # 该位置是内含子或者是最后一个位置
                                    results.append([strand, begin, end + 1, "l", gene_idx + 1, 1])
                                    begin = idx + 1
                                    end = end + 1
                                elif idx not in seq_span_type and idx + 1 < gene_end + 1 and idx + 1 not in seq_span_type:
                                    # 该位置是非内含子，并且下一个也是非内含子，说明在重要区间内部，那么需要退回到上一个内含子位置
                                    tmp_idx = idx - 1
                                    while tmp_idx >= begin and tmp_idx not in seq_span_type:
                                        tmp_idx -= 1
                                    # 没找到上一个内行子，那么重要区间截断，
                                    if tmp_idx < begin:
                                        results.append([strand, begin, idx + 1, "l", gene_idx + 1, 2])
                                        begin = idx + 1
                                    else: # 找到上一个内含子
                                        end = tmp_idx
                                        if end - begin + 1 < max_len:
                                            # 向左扩展内行子
                                            new_idx = begin - 1
                                            expand_len = max_len - (end - new_idx + 1)
                                            # 是内行子，并且还可以扩展
                                            while new_idx >= 0 and new_idx in seq_span_type and expand_len >= 0:
                                                new_idx -= 1
                                                expand_len -= 1
                                            results.append([strand, new_idx + 1, end + 1, "l", gene_idx + 1, 3])
                                        else:
                                            results.append([strand, begin, end + 1, "l", gene_idx + 1, 3])
                                        begin = end + 1
                                    end = end + 1
                                else: # 该位置是非内含子，下一个位置是内含子
                                    results.append([strand, begin, end + 1, "l", gene_idx + 1, 4])
                                    begin = idx + 1
                                    end = end + 1
                        # if idx >= begin and idx - begin + 1 < max_len:
                        while begin < gene_end + 1:
                            # 最后一个小于max_len，进行左右扩展
                            # 向左扩展内行子
                            new_begin_idx = begin - 1
                            # new_end_idx = min(gene_end + 1, end + 1)
                            new_end_idx = min(gene_end + 1, begin + max_len)
                            # expand_len = max_len - (end - begin + 1)
                            # expand_len = max_len - (gene_end - begin + 1)
                            expand_len = max_len - (new_end_idx - begin)
                            # 是内行子，并且还可以扩展
                            flag = True
                            while expand_len > 0 and flag:
                                flag = False
                                if new_begin_idx >= cur_pre_idx and new_begin_idx in seq_span_type:
                                    # 向左扩
                                    new_begin_idx -= 1
                                    expand_len -= 1
                                    flag = True
                                if expand_len > 0 and new_end_idx < cur_last_idx and new_end_idx in seq_span_type:
                                    # 向右扩
                                    new_end_idx += 1
                                    expand_len -= 1
                                    flag = True
                            results.append([strand, new_begin_idx + 1, new_end_idx, "l", gene_idx + 1, 5])
                            gene_idx += 1
                            begin = new_end_idx
                            end = new_end_idx
        return json.dumps(results, ensure_ascii=False)


    def gene_merge(self, gene_list, start_index, end_index, max_len):
        sorted_intervals = sorted(gene_list, key=lambda x: (x[start_index], -x[end_index]))
        result = []
        for interval in sorted_intervals:
            if result:
                if result[-1][end_index] >= interval[start_index] - 1 and interval[end_index] - result[-1][start_index] + 1 <= max_len:
                    # 有交集或者首尾相接的情况或者连续
                    result[-1][end_index] = max(result[-1][end_index], interval[end_index])
                else:
                    result.append(interval)
            else:
                result.append(interval)

        if len(result) <= 1:
            return result
        expand_result = []
        for interval_idx, interval in enumerate(result):
            if interval_idx == 0: # 第一个区间
                expand_left_idx = 0
                expand_right_idx = result[interval_idx + 1][start_index] - 1
            elif interval_idx == len(result) - 1: # 最后一个区间
                expand_left_idx = result[interval_idx - 1][end_index] + 1
                expand_right_idx = max_len - 1
            else: # 中间区间
                expand_left_idx = result[interval_idx - 1][end_index] + 1
                expand_right_idx = result[interval_idx + 1][start_index] - 1
            new_interval = copy.deepcopy(interval)
            cur_len = new_interval[end_index] - new_interval[start_index] + 1
            # 向左/右扩充补足随机个或者最大长度
            # expand_len = random.randint(0, max_len - cur_len)
            expand_len = max_len - cur_len
            flag = True
            while expand_len > 0 and flag:
                flag = False
                if new_interval[start_index] > expand_left_idx:
                    # 向左扩
                    new_interval[start_index] -= 1
                    expand_len -= 1
                    flag = True
                if expand_len > 0 and new_interval[end_index] < expand_right_idx:
                    # 向右扩
                    new_interval[end_index] += 1
                    expand_len -= 1
                    flag = True
            expand_result.append(new_interval)

        return expand_result

