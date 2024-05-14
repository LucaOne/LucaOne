#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/11/18 20:01
@project: LucaOne
@file: refseq_nucleo_fasta_span_extract.py
@desc: 根据提供的fasta区间进行seq 片段获取
'''

import csv, sys
import io, textwrap, itertools
import os

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
csv.field_size_limit(sys.maxsize)


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
            if isinstance(header, str):
                header = [header]
            writer.writerow(header)
        for row in dataset:
            if isinstance(row, str):
                row = [row]
            writer.writerow(row)
    except Exception as e:
        raise e
    finally:
        if not handle.closed:
            handle.close()


def seq_reverse(seq):
    seq = seq.upper().strip()
    reverse_seq = ""
    for v in seq:
        if v == "A":
            reverse_seq += "T"
        elif v == "T" or v == "U":
            reverse_seq += "A"
        elif v == "G":
            reverse_seq += "C"
        elif v == "C":
            reverse_seq += "G"
        else:
            reverse_seq += v
    return reverse_seq


def write_chunk(chunk_t_fasta,  chunk_idx, save_dir, prefix=""):
    csv_writer(
        chunk_t_fasta,
        os.path.join(save_dir, f"{prefix}_fasta_seg_{chunk_idx}.csv"),
        header=["seq_id", "gene_idx", "seq_start_p", "seq_end_p", "segment"]
    )



'''
-- tunnel download tmp_lucaone_v2_refseq_fasta_span_v2 /mnt3/sanyuan.hy/data/refseq/tmp_lucaone_v2_refseq_fasta_span_v2.csv -cf true -fd "," -rd "\n" -h true;

tunnel download tmp_lucaone_v2_refseq_fasta_span_rna /mnt3/sanyuan.hy/data/refseq/tmp_lucaone_v2_refseq_fasta_span_rna.csv -cf true -fd "," -rd "\n" -h true;

tunnel download tmp_lucaone_v2_refseq_dna_no_segments_v2 /mnt3/sanyuan.hy/data/refseq/tmp_lucaone_v2_refseq_dna_no_segments_v2.csv -cf true -fd "," -rd "\n" -h true;
'''


def load_done(save_dir, prefix):
    if not os.path.exists(save_dir):
        return None
    target_s = f"{prefix}_fasta_seg_"
    max_idx = 0
    done_set = set()
    done_seq_id_set = set()
    done_num = 0
    for filename in os.listdir(save_dir):
        if target_s in filename:
            cur_idx = int(filename.replace(target_s, "").replace(".csv", ""))
            if max_idx < cur_idx:
                max_idx = cur_idx
            for item in csv_reader(os.path.join(save_dir, filename), header_filter=True, header=True):
                seq_id = item[0]
                gene_idx = int(item[1])
                seq_start_p = int(item[2])
                seq_end_p = int(item[3])
                reverse = False
                if seq_id.endswith("_r_"):
                    ref_seq_id = seq_id.replace("_r_", "")
                    reverse = True
                else:
                    ref_seq_id = seq_id
                done_seq_id_set.add(ref_seq_id)
                done_num += 1
                done_set.add("%s_%d_%d_%d_%d" %(ref_seq_id, gene_idx, seq_start_p, seq_end_p, int(reverse)))
    done_seq_num = len(done_seq_id_set)
    print("done_set size: %d, max_idx: %d, done_seq_num: %d done_num: %d" %(len(done_set), max_idx, done_seq_num, done_num))
    return done_set, max_idx, done_seq_num, done_num


def extract(fasta_dir, span_info_filepath, span_info_dir, save_dir, chunk_size=1000000, prefix="refseq_segs", part=None, total_part=None, header=True):

    if span_info_filepath is not None and os.path.exists(span_info_filepath):
        span_info_filepath_list = [span_info_filepath]
    elif span_info_dir is not None and os.path.exists(span_info_dir):
        span_info_filepath_list = os.listdir(span_info_dir)
        span_info_filepath_list = [os.path.join(span_info_dir, filename) for filename in span_info_filepath_list if filename.endswith(".csv")]
        span_info_filepath_list = sorted(span_info_filepath_list)
        if part and total_part and 0 < part <= total_part:
            size = len(span_info_filepath_list)
            per_num = (size + total_part - 1) // total_part
            cur_span_info_filepath_list = span_info_filepath_list[(part - 1) * per_num: min(part * per_num, size)]
            span_info_filepath_list = cur_span_info_filepath_list
            print("part/total_part: %d/%d, files size %d:" % (part, total_part, len(span_info_filepath_list)))
    else:
        raise Exception("input error")
    # seq_id, ref_seq_id, gene_idx, seq_start_p, seq_end_p
    done_set, max_chunk_idx, done_seq_num, done_num = load_done(save_dir, prefix)
    targets = {}
    total_seq_num = 0
    for span_info_filepath in span_info_filepath_list:
        for item in csv_reader(span_info_filepath, header_filter=header, header=header):
            seq_id, ref_seq_id, gene_idx, seq_start_p, seq_end_p = item[0], item[1], int(item[2]), int(item[3]), int(item[4])
            reverse = False
            if seq_id.endswith("_r_"):
                reverse = True
            if ref_seq_id not in targets:
                targets[ref_seq_id] = []
            unique_id = "%s_%d_%d_%d_%d" %(ref_seq_id, gene_idx, seq_start_p, seq_end_p, int(reverse))
            if unique_id not in done_set:
                targets[ref_seq_id].append([gene_idx, seq_start_p, seq_end_p, reverse])
                total_seq_num += 1
    print("need seq id cnt: %d, need total: %d" % (len(targets), total_seq_num))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fasta_segments_data = []
    chunk_idx = max_chunk_idx
    total_seq_num = 0
    total = 0
    for filename in os.listdir(fasta_dir):
        if not filename.endswith(".csv"):
            continue
        filepath = os.path.join(fasta_dir, filename)
        for item in csv_reader(filepath, header_filter=True, header=True):
            ref_seq_id, seq_header, seq_len, seq, seq_type = item[0], item[1], item[2], item[3], item[4]
            reverse_fasta = None
            if ref_seq_id in targets:
                total_seq_num += 1
                for target in targets[ref_seq_id]:
                    total += 1
                    gene_idx, seq_start_p, seq_end_p, reverse = target
                    if reverse:
                        if reverse_fasta is None:
                            reverse_fasta = seq_reverse(seq)
                        segment = reverse_fasta[seq_start_p:seq_end_p]
                        seq_id = ref_seq_id + "_r_"
                    else:
                        segment = seq[seq_start_p:seq_end_p]
                        seq_id = ref_seq_id
                    fasta_segments_data.append([seq_id, gene_idx, seq_start_p, seq_end_p, segment])
                    if len(fasta_segments_data) % chunk_size == 0:
                        chunk_idx += 1
                        write_chunk(fasta_segments_data,  chunk_idx=chunk_idx, save_dir=save_dir, prefix=prefix)
                        fasta_segments_data = []
                    if total % 1000000 == 0:
                        print("seq id cnt: %d, total: %d" % (total_seq_num, total))

    if len(fasta_segments_data) > 0:
        chunk_idx += 1
        write_chunk(fasta_segments_data,  chunk_idx=chunk_idx, save_dir=save_dir, prefix=prefix)
        fasta_segments_data = []
    print("over. seq id cnt: %d, total: %d" % (total_seq_num, total))


import argparse
parser = argparse.ArgumentParser(description='lucaone_v2_refseq_fasta_span')
# for logging
parser.add_argument("--part", type=int, default=None, required=True, help="part")
parser.add_argument("--total_part", type=int, default=None, help="total_part")
args = parser.parse_args()


if __name__ == "__main__":
    extract(
        fasta_dir="/mnt3/sanyuan.hy/data/refseq/fasta_parsed/fasta/",
        span_info_filepath=None,
        span_info_dir="/mnt3/sanyuan.hy/data/refseq/tmp_lucaone_v2_refseq_dna_no_segments_v2/",
        save_dir="/mnt3/sanyuan.hy/data/refseq/fasta_segments/refseq_dna_remain/",
        part=args.part,
        total_part=args.total_part,
        chunk_size=1000000,
        prefix="refseq_dna_remain_part_%02d_%02d" % (args.part, args.total_part),
        header=False
    )
    '''
    python refseq_nucleo_fasta_span_extract.py --part 6 --total_part 20
    done 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20
    中断 6, 9, 12, 16, 18, 19
    
    ossutil cp -r -u /mnt3/sanyuan.hy/data/refseq/fasta_segments/refseq_dna_remain/ oss://lucaone-data2/refseq_parsed/fasta_segments/refseq_dna_remain/
    
    
    # oss to odps    "seq_id", "gene_idx", "seq_start_p", "seq_end_p", "segment"]
    create external table if not exists lucaone_data2.tmp_lucaone_v2_refseq_fasta_segments_c2_v2
    (
        seq_id string,
        gene_idx bigint,
        seq_start_p bigint,
        seq_end_p bigint,
        segment string 
    )
    stored by 'com.aliyun.odps.CsvStorageHandler'
    location 'oss://lucaone-data2/refseq_parsed/fasta_segments/refseq_dna_remain/';
    
    create  table if not exists luca_data2.tmp_lucaone_v2_refseq_fasta_segments_c2_v2
    as
    select *
    from lucaone_data2.tmp_lucaone_v2_refseq_fasta_segments_c2_v2
    where seq_id != 'seq_id';
    
    extract(fasta_dir="/mnt3/sanyuan.hy/data/refseq/fasta_parsed/fasta/",
            span_info_filepath="/mnt3/sanyuan.hy/data/refseq/tmp_lucaone_v2_refseq_fasta_span_v2/tmp_lucaone_v2_refseq_fasta_span_v2_part_%02d.csv" % args.part,
            save_dir="/mnt3/sanyuan.hy/data/refseq/fasta_segments/refseq/part_%02d" % args.part,
            chunk_size=1000000,
            prefix="refseq")
    extract(fasta_dir="/mnt3/sanyuan.hy/data/refseq/fasta_parsed/fasta/", 
            span_info_filepath="/mnt3/sanyuan.hy/data/refseq/tmp_lucaone_v2_refseq_fasta_span_rna.csv", 
            save_dir="/mnt3/sanyuan.hy/data/refseq/fasta_segments/refseq_rna/", 
            chunk_size=1000000, 
            prefix="refseq")
    extract(fasta_dir="/mnt3/sanyuan.hy/data/refseq/fasta_parsed/fasta/",
            span_info_filepath="/mnt3/sanyuan.hy/data/refseq/tmp_lucaone_v2_refseq_fasta_span_rna_part_01.csv",
            save_dir="/mnt3/sanyuan.hy/data/refseq/fasta_segments/refseq_rna/part_01/",
            chunk_size=1000000,
            prefix="refseq")
    extract(fasta_dir="/mnt3/sanyuan.hy/data/refseq/fasta_parsed/fasta/",
            span_info_filepath="/mnt3/sanyuan.hy/data/refseq/tmp_lucaone_v2_refseq_fasta_span_v2.csv",
            save_dir="/mnt3/sanyuan.hy/data/refseq/fasta_segments/refseq/",
            chunk_size=1000000,
            prefix="refseq")
            
            
    create table if not exists tmp_lucaone_v2_refseq_fasta_span_rna_fasta_segments(
        seq_id string,
        gene_idx bigint,
        seq_start_p bigint,
        seq_end_p bigint,
        segment string
    );
    
    tunnel upload /mnt2/workspace/data/refseq_parsed/fasta_segments/refseq_rna/part_01 luca_data2.tmp_lucaone_v2_refseq_fasta_span_rna_fasta_segments -cf true -fd "," -rd "\n" -h true;
    9732289
    tunnel upload /mnt2/workspace/data/refseq_parsed/fasta_segments/refseq_rna/part_02 luca_data2.tmp_lucaone_v2_refseq_fasta_span_rna_fasta_segments -cf true -fd "," -rd "\n" -h true;
    
    tunnel upload /mnt2/workspace/data/refseq_parsed/fasta_segments/refseq_rna/part_02 luca_data2.tmp_lucaone_v2_refseq_fasta_span_rna_fasta_segments -cf true -fd "," -rd "\n" -h true;
    
    tunnel upload /mnt2/workspace/data/refseq_parsed/fasta_segments/refseq_rna/part_02 luca_data2.tmp_lucaone_v2_refseq_fasta_span_rna_fasta_segments -cf true -fd "," -rd "\n" -h true;
    
    tunnel upload /mnt2/workspace/data/refseq_parsed/fasta_segments/refseq_rna/part_02 luca_data2.tmp_lucaone_v2_refseq_fasta_span_rna_fasta_segments -cf true -fd "," -rd "\n" -h true;
    
    tunnel upload /mnt2/workspace/data/refseq_parsed/fasta_segments/refseq_rna/part_02 luca_data2.tmp_lucaone_v2_refseq_fasta_span_rna_fasta_segments -cf true -fd "," -rd "\n" -h true;
    
    tunnel upload /mnt2/workspace/data/refseq_parsed/fasta_segments/refseq_rna/part_02 luca_data2.tmp_lucaone_v2_refseq_fasta_span_rna_fasta_segments -cf true -fd "," -rd "\n" -h true;
    
    tunnel upload /mnt2/workspace/data/refseq_parsed/fasta_segments/refseq_rna/part_02 luca_data2.tmp_lucaone_v2_refseq_fasta_span_rna_fasta_segments -cf true -fd "," -rd "\n" -h true;
    
    tunnel upload /mnt2/workspace/data/refseq_parsed/fasta_segments/refseq_rna/part_02 luca_data2.tmp_lucaone_v2_refseq_fasta_span_rna_fasta_segments -cf true -fd "," -rd "\n" -h true;
    
    tunnel upload /mnt2/workspace/data/refseq_parsed/fasta_segments/refseq_rna/part_02 luca_data2.tmp_lucaone_v2_refseq_fasta_span_rna_fasta_segments -cf true -fd "," -rd "\n" -h true;
    
    create table if not exists tmp_lucaone_v2_refseq_fasta_info_100w(
        seq_id string,
        seq string,
        reverse bigint
    );
    tunnel upload /mnt2/workspace/data/refseq_parsed/csv_split_100w/_rna/part_02 luca_data2.tmp_lucaone_v2_refseq_fasta_info_100w -cf true -fd "," -rd "\n" -h true;
    
    
    '''





