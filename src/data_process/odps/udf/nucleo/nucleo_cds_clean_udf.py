#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/10/30 21:06
@project: LucaOne
@file: nucleo_cds_clean_udf.py
@desc: nucleo cds clean udf
'''

from odps.udf import annotate
from odps.distcache import get_cache_table
from odps.udf import BaseUDTF
import random
import sys, copy, json


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


def remove_interval(intervals, remove):
    result =[]
    for interval in intervals:
        if interval[1] == remove[1]:
            if remove[2] <= interval[2] and remove[3] >= interval[3]:
                # 要去除的区间完全包含当前区间
                continue
            elif interval[2] <= remove[2] and interval[3] >= remove[3]:
                # 当前区间完全包含要去除的区间
                if interval[2] != remove[2]:
                    result.append([interval[0], interval[1], interval[2], remove[2], interval[4]])
                if interval[3] != remove[3]:
                    result.append([interval[0], interval[1], remove[3], interval[3], interval[4]])
            elif remove[3] < interval[2] or remove[2] > interval[3]:
                # 要去除的区间与当前区间没有交集
                result.append(interval)
            else:
                # 要去除的区间与当前区间有交集
                if interval[2] < remove[2]:
                    result.append([interval[0], interval[1], interval[2], remove[2], interval[4]])
                if interval[3] > remove[3]:
                    result.append([interval[0], interval[1], remove[3], interval[3], interval[4]])
        else:
            result.append(interval)

    return result


def extract_inteval(section, feature):
    result = []

    section_normal = [item for item in section if item[1] == 1]
    section_minus = [item for item in section if item[1] == -1]

    section_normal = sorted(section_normal,key=lambda x:x[2])
    section_minus = sorted(section_minus,key=lambda x:x[2])

    if len(section_normal) > 1:
        for i in range(len(section_normal) - 1):
            # 后一个区间的起始 > 前一个区间的结束
            if section_normal[i+1][2] > section_normal[i][3]:
                result.append([section_normal[i][0], section_normal[i][1], section_normal[i][3], section_normal[i+1][2], feature])

    if len(section_minus) > 1:
        for i in range(len(section_minus) - 1):
            # 后一个区间的起始 > 前一个区间的结束
            if section_minus[i + 1][2] > section_minus[i][3]:
                result.append(
                    [section_minus[i][0], section_minus[i][1], section_minus[i][3], section_minus[i + 1][2], feature])
    return result


@annotate('string->string,string,string,string')
class nucleo_cds_clean(BaseUDTF):
    def __init__(self):
        workaround_argv_issue("nucleo_cds_clean_udf.py")
        sys.path.insert(0, 'work/biopython-1.80-cp37.zip')

    def process(self, start_end_concat):
        from Bio.SeqFeature import SeqFeature, Reference, FeatureLocation, CompoundLocation, \
            SimpleLocation, ExactPosition,UnknownPosition, UncertainPosition, WithinPosition, \
            BetweenPosition, BeforePosition, AfterPosition, OneOfPosition, PositionGap

        feature_dict_input = {
            "CDS": [],
            "intron": [],
            "exon": [],
            "UTR": []
        }

        feature_dict_output = {
            "CDS": [],
            "intron": [],
            "exon": [],
            "UTR": []
        }

        start_end_list = [
            line.split('###,,,###') for line in start_end_concat.split('###;;;###')
        ]

        for i in range(len(start_end_list)):
            seq_id = start_end_list[i][0]
            feature_type = start_end_list[i][1]
            start_end_item = start_end_list[i][2]
            eval_se = eval(start_end_item)

            for pt in eval_se.parts:
                start = int(pt.start)
                end = int(pt.end)
                strand = pt.strand
                if feature_type == 'CDS':
                    feature_dict_input['CDS'].append([seq_id, strand, start, end, feature_type])

                if feature_type == 'exon':
                    feature_dict_input['exon'].append([seq_id, strand, start, end, feature_type])

                if feature_type == 'intron' or feature_type == 'intron_manual':
                    feature_dict_input['intron'].append([seq_id, strand, start, end, feature_type])

                if feature_type in ("3'UTR", "5'UTR"):
                    feature_dict_input['UTR'].append([seq_id, strand, start, end, feature_type])

        feature_dict_output['CDS'] = feature_dict_input['CDS']
        feature_dict_output['intron'] = feature_dict_input['intron']
        feature_dict_output['exon'] = feature_dict_input['exon']
        feature_dict_output['UTR'] = feature_dict_input['UTR']

        # cds overleap intron
        if len(feature_dict_input['CDS']) > 0 and len(feature_dict_input['intron']) > 0:
            for item in feature_dict_output['intron']:
                feature_dict_output['CDS'] = remove_interval(feature_dict_output['CDS'], item)

        # exon overleap UTR
        if len(feature_dict_input['exon']) > 0 and len(feature_dict_input['UTR']) > 0:
            for item in feature_dict_output['UTR']:
                feature_dict_output['exon'] = remove_interval(feature_dict_output['exon'], item)


        result = feature_dict_output['CDS'] + \
                 feature_dict_output['exon'] + \
                 feature_dict_output['intron'] + \
                 feature_dict_output['UTR']

        if len(result) > 0:
            for item in result:
                if len(item) != 5:
                    print(item)
                    continue
                if item[1] == 1:
                    strand = +1
                    strand_str = '1'
                elif item[1] == -1:
                    strand = -1
                    strand_str = '-1'
                else:
                    strand = None
                    strand_str = None
                self.forward(item[0], item[4], strand_str, repr(SimpleLocation(start=item[2], end=item[3], strand=strand)))

