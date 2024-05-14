#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/10/13 16:09
@project: LucaOne
@file: extract_position_udtf.py
@desc: extract position udtf
'''
# pyodps pip install pyodps
# step1: 上传python资源
# add py /mnt/sanyuan.hy/extract_position_udtf.py -f;
# step2: 上传第三方包（后缀名改成.zip) biopython-1.80-cp37.zip,https://help.aliyun.com/zh/maxcompute/user-guide/resource-operations
# add archive /mnt/sanyuan.hy/biopython-1.80-cp37.zip -f;
# step3: 注册函数
# create function extract_position_v1 as 'extract_position_udtf.extract_position' using 'extract_position_udtf.py,biopython-1.80-cp37.zip' -f;
# step4: 运行SQL
# set odps.sql.python.version=cp37;
# SELECT extract_position_v1(seq_id, feature_type, start_end) as (seq_id, feature_type, start_p, end_p)
# FROM  luca_data2.t_nucleo_feature_refseq
# limit 100;

'''
set odps.sql.python.version=cp37;
create table if not exists t_nucleo_feature_refseq_gene as
select seq_id, feature_type, start_p, end_p, row_number() over (partition by seq_id order by start_p, end_p) + 1 as inner_id
from(
    SELECT extract_position_v1(seq_id, feature_type, start_end) as (seq_id, feature_type, start_p, end_p)
    FROM  luca_data2.t_nucleo_feature_refseq
    where feature_type == "gene"
) tmp
'''

# SELECT seq_id, feature_type, end_p - start_p as span_len
# FROM  tablename
# limit 100;
import json
from odps.udf import annotate
from odps.udf import BaseUDTF


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


@annotate("string,string->string,bigint,bigint,bigint,string,bigint,bigint,bigint")
class refseq_extract_split_info(BaseUDTF):
    def __init__(self):
        import sys
        workaround_argv_issue("refseq_nucleo_extract_udtf.py")
        sys.path.insert(0, 'work/biopython-1.80-cp37.zip')

    def process(self, seq_id, split_info):
        from Bio.SeqFeature import SeqFeature, Reference, FeatureLocation, CompoundLocation, SimpleLocation, ExactPosition, UnknownPosition, UncertainPosition, WithinPosition, BetweenPosition, BeforePosition, AfterPosition, OneOfPosition, PositionGap

        # seq_id, strand, start_p,  end_p, fragment_type, gene_id, split_type, segment_len
        # strand, begin, idx + 1, "l", gene_idx + 1, 5
        if split_info is not None:
            split_info = json.loads(split_info, encoding="UTF-8")
            for item in split_info:
                segment_len = item[2] - item[1]
                strand = int(item[0])
                if strand < 0:
                    new_seq_id = seq_id + "_r_"
                else:
                    new_seq_id = seq_id
                self.forward(new_seq_id, * item, segment_len)
