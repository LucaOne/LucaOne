#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/10/26 13:55
@project: LucaOne
@file: interpro_parse.py
@desc: interpro parse
'''


from datetime import datetime
import csv, os
import io, textwrap, itertools
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from Bio.SeqFeature import *

import gzip
import re
from datetime import datetime
import time
import pymysql
import sys
import pandas as pd
import hashlib
import json


def tsv_reader(handle, header=True, header_filter=True):
    '''
    csv 读取器，适合大文件
    :param handle:
    :param header:
    :param header_filter: 返回结果是否去掉头
    :return:
    '''
    handle = handle if isinstance(handle, io.TextIOWrapper) else open(handle, 'r')
    try:
        reader = csv.reader(handle, delimiter="\t")
        cnt = 0
        for row in reader:
            cnt += 1
            if header and header_filter and cnt == 1:
                continue
            yield row
    except Exception as e:
        raise StopIteration
    finally:
        if not handle.closed:
            handle.close()


def csv_reader(handle, header=True, header_filter=True):
    '''
    csv 读取器，适合大文件
    :param handle:
    :param header:
    :param header_filter: 返回结果是否去掉头
    :return:
    '''
    handle = handle if isinstance(handle, io.TextIOWrapper) else open(handle, 'r')
    try:
        reader = csv.reader(handle)
        cnt = 0
        for row in reader:
            cnt += 1
            if header and header_filter and cnt == 1:
                continue
            yield row
    except Exception as e:
        raise StopIteration
    finally:
        if not handle.closed:
            handle.close()


def parse(filepath, save_dir, prefix, chunk_size=1000000):
    chunk_idx = 1
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    wfp = open(os.path.join(save_dir, "%s_%04d.csv" %(prefix, chunk_idx)), "w")
    writer = csv.writer(wfp)
    writer.writerow(["prot_seq_accession", "interpro_accession", "interpro_feature_name",
                     "original_accession", "start_p", "end_p", "insert_date"])
    insert_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cnt = 0

    for row in tsv_reader(filepath, header=False, header_filter=False):
        new_row = [*row] + [insert_date]
        writer.writerow(new_row)
        cnt += 1
        if cnt % chunk_size == 0:
            wfp.close()
            chunk_idx += 1
            wfp = open(os.path.join(save_dir, "%s_%04d.csv" %(prefix, chunk_idx)), "w")
            writer = csv.writer(wfp)
            writer.writerow(["prot_seq_accession", "interpro_accession", "interpro_feature_name",
                             "original_accession", "start_p", "end_p", "insert_date"])
            print("done %d" % cnt)

    wfp.close()


parse("/mnt3/sanyuan.hy/data/interpro/protein2ipr.dat",
      "/mnt3/sanyuan.hy/data/interpro/interpro_parsed/",
      "tmp_prot_feature_interpro",
      1000000
      )