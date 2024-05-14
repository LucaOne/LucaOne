#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/11/24 17:22
@project: LucaOne
@file: verify_dataset
@desc: verify dataset
'''
import csv, os, io, sys
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


import argparse
parser = argparse.ArgumentParser(description='dataset_verfiy')
parser.add_argument("--dir_path", type=str, default=None, required=True, help="dir_path")
parser.add_argument("--part", type=int, default=None, required=True, help="part")
parser.add_argument("--total_part", type=int, default=None, required=True, help="total_part")
args = parser.parse_args()

'''
python verify_dataset.py --dir_path /mnt/sanyuan.hy/workspace/LucaOne/dataset/lucagplm/v2.0/test/ --part 1 --total_part 3
python verify_dataset.py --dir_path /mnt/sanyuan.hy/workspace/LucaOne/dataset/lucagplm/v2.0/test/ --part 2 --total_part 3
python verify_dataset.py --dir_path /mnt/sanyuan.hy/workspace/LucaOne/dataset/lucagplm/v2.0/test/ --part 3 --total_part 3

python verify_dataset.py --dir_path /mnt/sanyuan.hy/workspace/LucaOne/dataset/lucagplm/v2.0/dev/ --part 1 --total_part 3
python verify_dataset.py --dir_path /mnt/sanyuan.hy/workspace/LucaOne/dataset/lucagplm/v2.0/dev/ --part 2 --total_part 3
python verify_dataset.py --dir_path /mnt/sanyuan.hy/workspace/LucaOne/dataset/lucagplm/v2.0/dev/ --part 3 --total_part 3

cd /mnt/sanyuan.hy/workspace/
python verify_dataset.py --dir_path /mnt/sanyuan.hy/workspace/LucaOne/dataset/lucagplm/v2.0/train/ --part 1 --total_part 20
python verify_dataset.py --dir_path /mnt/sanyuan.hy/workspace/LucaOne/dataset/lucagplm/v2.0/train/ --part 2 --total_part 20
python verify_dataset.py --dir_path /mnt/sanyuan.hy/workspace/LucaOne/dataset/lucagplm/v2.0/train/ --part 3 --total_part 20
python verify_dataset.py --dir_path /mnt/sanyuan.hy/workspace/LucaOne/dataset/lucagplm/v2.0/train/ --part 4 --total_part 20
python verify_dataset.py --dir_path /mnt/sanyuan.hy/workspace/LucaOne/dataset/lucagplm/v2.0/train/ --part 5 --total_part 20
python verify_dataset.py --dir_path /mnt/sanyuan.hy/workspace/LucaOne/dataset/lucagplm/v2.0/train/ --part 6 --total_part 20
python verify_dataset.py --dir_path /mnt/sanyuan.hy/workspace/LucaOne/dataset/lucagplm/v2.0/train/ --part 7 --total_part 20
python verify_dataset.py --dir_path /mnt/sanyuan.hy/workspace/LucaOne/dataset/lucagplm/v2.0/train/ --part 8 --total_part 20
python verify_dataset.py --dir_path /mnt/sanyuan.hy/workspace/LucaOne/dataset/lucagplm/v2.0/train/ --part 9 --total_part 20
python verify_dataset.py --dir_path /mnt/sanyuan.hy/workspace/LucaOne/dataset/lucagplm/v2.0/train/ --part 10 --total_part 20
python verify_dataset.py --dir_path /mnt/sanyuan.hy/workspace/LucaOne/dataset/lucagplm/v2.0/train/ --part 11 --total_part 20
python verify_dataset.py --dir_path /mnt/sanyuan.hy/workspace/LucaOne/dataset/lucagplm/v2.0/train/ --part 12 --total_part 20
python verify_dataset.py --dir_path /mnt/sanyuan.hy/workspace/LucaOne/dataset/lucagplm/v2.0/train/ --part 13 --total_part 20
python verify_dataset.py --dir_path /mnt/sanyuan.hy/workspace/LucaOne/dataset/lucagplm/v2.0/train/ --part 14 --total_part 20
python verify_dataset.py --dir_path /mnt/sanyuan.hy/workspace/LucaOne/dataset/lucagplm/v2.0/train/ --part 15 --total_part 20
python verify_dataset.py --dir_path /mnt/sanyuan.hy/workspace/LucaOne/dataset/lucagplm/v2.0/train/ --part 16 --total_part 20
python verify_dataset.py --dir_path /mnt/sanyuan.hy/workspace/LucaOne/dataset/lucagplm/v2.0/train/ --part 17 --total_part 20
python verify_dataset.py --dir_path /mnt/sanyuan.hy/workspace/LucaOne/dataset/lucagplm/v2.0/train/ --part 18 --total_part 20
python verify_dataset.py --dir_path /mnt/sanyuan.hy/workspace/LucaOne/dataset/lucagplm/v2.0/train/ --part 19 --total_part 20
python verify_dataset.py --dir_path /mnt/sanyuan.hy/workspace/LucaOne/dataset/lucagplm/v2.0/train/ --part 20 --total_part 20


'''

error_fp = open("verify_dataset_err_info_%d_%d" % (args.part, args.total_part), "w")
dir_name = args.dir_path

filename_list = []
for filename in os.listdir(dir_name):
    if not filename.endswith(".csv"):
        print(filename)
        continue
    filename_list.append(filename)
filename_list = sorted(filename_list)
print("total file cnt: %d" % len(filename_list))
per_num = (len(filename_list) + args.total_part - 1) // args.total_part
cur_filename_list = filename_list[(args.part - 1) * per_num : min(args.part * per_num, len(filename_list))]
file_cnt = 0
ok = True
print("need to verify file cnt: %d" % len(cur_filename_list))
for filename in cur_filename_list:
    file_cnt += 1
    cnt = 0
    try:
        for row in csv_reader(os.path.join(dir_name, filename), header=False, header_filter=False):
            cnt += 1
            if cnt == 1:
                if row[0] != "obj_id" or row[1] != "obj_type" or row[2] != "obj_seq" or row[3] != "obj_label" or row[4] != "obj_source":
                    error_fp.write(filename + " header error\n")
                    ok = False
                    error_fp.flush()
            elif len(row) <= 4:
                error_fp.write(filename + " row " + str(cnt) + " len error\n")
                error_fp.flush()
                ok = False
            else:
                try:
                    v = eval(row[3])
                except Exception as e:
                    error_fp.write(filename + " row " + str(cnt) + " paser error\n")
                    error_fp.flush()
                    ok = False
    except Exception as e:
        error_fp.write(filename + " row " + str(cnt) + " paser error %s\n" % str(e))
        error_fp.flush()
    if file_cnt % 100 == 0:
        print("done file cnt: %d" % file_cnt)
error_fp.flush()
error_fp.close()
print("verify file_cnt: %d" % file_cnt, ", ok=" + str(ok))