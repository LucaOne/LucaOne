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


def dict_to_row(item, cols):
    row = []
    for col in cols:
        row.append(item[col])
    return row


def csv_writer(dataset, handle, header):
    '''
    csv 写，适合大文件
    :param dataset: 数据
    :param handle: 文件
    :param header: 头
    :return:
    '''
    handle = handle if isinstance(handle, io.TextIOWrapper) else open(handle, 'w')
    try:
        writer = csv.writer(handle)
        if header:
            writer.writerow(header)
        for row in dataset:
            if isinstance(row, dict):
                row = dict_to_row(row, header)
            writer.writerow(row)
    except Exception as e:
        raise e
    finally:
        if not handle.closed:
            handle.close()


def parse_one(filepath):
    t_nucleo_fasta_all = []
    with gzip.open(filepath, "rt") as rfp:
        for rec in SeqIO.parse(rfp, "fasta"):
            t_nucleo_fasta = [rec.id, rec.id, len(rec), str(rec.seq).upper()]
            t_nucleo_fasta_all.append(t_nucleo_fasta)
    return t_nucleo_fasta_all


def write_chunk(chunk_t_fasta, chunk_idx, save_dir, prefix=""):
    for dirname in ["fasta"]:
        dirpath = os.path.join(save_dir, dirname)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

    csv_writer(
        chunk_t_fasta,
        os.path.join(save_dir, "fasta", f"{prefix}_fasta_{chunk_idx}.csv"),
        header=["seq_id", "seq_header", "seq_len", "seq", "seq_type"]
    )


def parse(dirpath, save_dir, prefix, chunk_size=1000000):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    err_wfp = open(os.path.join(save_dir, "err_parse_fasta.txt"), "w")

    chunk_t_nucleo_fasta = []
    dir_num = 0
    seq_cnt = 0
    total_cnt = 0
    chunk_idx = 0
    total_fasta_cnt = 0
    for dirname in os.listdir(dirpath):
        file_dir = os.path.join(dirpath, dirname)
        filepath_list = set()
        for filename in os.listdir(file_dir):
            if "genomic.fna" in filename or "rna.fna" in filename:
                filepath = os.path.join(file_dir, filename)
                filepath_list.add(filepath)
        if len(filepath_list) == 0:
            err_wfp.write("%s not contains fna file\n" % dirname)
            err_wfp.flush()
            continue
        print("process filenames: %s" % filepath_list)
        dir_num += 1
        for filepath in filepath_list:
            try:
                if "genomic.fna" in filepath:
                    seq_type = "genomic"
                else:
                    seq_type = "rna"

                cur_nucleo_fasta_all = parse_one(filepath)
                new_cur_nucleo_fasta_all = []
                for item in cur_nucleo_fasta_all:
                    new_cur_nucleo_fasta_all.append([*item, seq_type])
                cur_nucleo_fasta_all = new_cur_nucleo_fasta_all
                chunk_t_nucleo_fasta.extend(cur_nucleo_fasta_all)

                cur_size = len(cur_nucleo_fasta_all)
                seq_cnt += cur_size
                total_cnt += cur_size
                total_fasta_cnt += cur_size
                # print(seq_cnt)
                if seq_cnt >= chunk_size:
                    chunk_idx += 1
                    write_chunk(chunk_t_nucleo_fasta, chunk_idx, save_dir, prefix=prefix)
                    chunk_t_nucleo_fasta = []
                    seq_cnt = 0
            except Exception as e:
                print(e)
                print(filepath, seq_cnt, e)
                err_wfp.write("%s,%d,%s\n" % (filepath, seq_cnt, e))
                err_wfp.flush()
    print("total_fasta_cnt: %d" % total_fasta_cnt)
    print("dir_num: %d" % dir_num)
    print("total: %d" % total_cnt)
    err_wfp.close()
    if len(chunk_t_nucleo_fasta) > 0:
        chunk_idx += 1
        write_chunk(chunk_t_nucleo_fasta,chunk_idx, save_dir, prefix=prefix)


parse("/mnt2/yanyan/refseq_data/",
      "/mnt3/sanyuan.hy/data/refseq/fasta_parsed/",
      "refseq",
      chunk_size=100000
      )

'''
total_fasta_cnt: 1,357,390,996
dir_num: 297780
total: 1,357,390,996
ossutil cp -r -u /mnt2/yanyan/refseq_data/ oss://lucaone-data2/refseq_data/
'''