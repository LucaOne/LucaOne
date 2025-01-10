#!/usr/bin/env python
# encoding: utf-8
"""
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2024/12/26 16:47
@project: LucaOne
@file: pseudogene_only_mismatch
@desc: xxxx
"""
import json
import os.path
import sys
import random
import torch
import numpy as np
sys.path.append(".")
sys.path.append("../")
sys.path.append("../..")
sys.path.append("../../src")
try:
    from file_operator import *
except ImportError:
    from src.file_operator import *


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(1221)

pseudogene_data_filepath = "../../data/pseudogene/aligned_pairs.csv"
if not os.path.exists("../../data/pseudogene/only_mismatch"):
    os.makedirs("../../data/pseudogene/only_mismatch")

nature_selected_num = 0
total_num = 0
nature_selected_recover_data = []
nature_selected_retain_data = []
nature_seq_len_list = []
nature_mlm_mask_num_list = []
nature_mlm_probability_list = []
for row in csv_reader(pseudogene_data_filepath):
    seq_id_a, seq_id_b, raw_seq_a, raw_seq_b, seq_a_aligned, seq_b_aligned, insertion_mask, deletion_mask, mismatch_mask = row
    seq_a_aligned = seq_a_aligned.upper()
    seq_b_aligned = seq_b_aligned.upper()
    insertion_mask = eval(insertion_mask)
    deletion_mask = eval(deletion_mask)
    mismatch_mask = eval(mismatch_mask)

    if sum(mismatch_mask) > 0:
        # 根据insertion_mask 从 seq_b_aligned 中去除对应位置的-， seq_a_aligned去除对应位置的ch
        # 根据deletion_mask 那么seq_a_aligned的-变成seq_b_aligned的ch
        assert len(seq_a_aligned) == len(seq_b_aligned)
        assert len(mismatch_mask) == len(seq_a_aligned)
        assert len(insertion_mask) == len(seq_a_aligned)
        assert len(deletion_mask) == len(seq_a_aligned)
        new_seq_a_aligned = ""
        new_seq_b_aligned = ""
        for idx, v in enumerate(deletion_mask):
            if v == 1:
                assert seq_a_aligned[idx] == "-"
                assert seq_b_aligned[idx] in ["A", "T", "C", "G"]
                new_seq_a_aligned += seq_b_aligned[idx]
            else:
                new_seq_a_aligned += seq_a_aligned[idx]
            new_seq_b_aligned += seq_b_aligned[idx]

        nature_masked_seq_a = ""
        nature_masked_num = 0
        for ch_idx, ch in enumerate(new_seq_a_aligned):
            mis_flag = mismatch_mask[ch_idx]
            if mis_flag > 0:
                print("mutation idx: %d, truth: %s -> mutation: %s" % (ch_idx, new_seq_a_aligned[ch_idx], ch))
                nature_masked_seq_a += "-"
                nature_masked_num += 1
            else:
                nature_masked_seq_a += ch

        new_new_seq_a_aligned = ""
        new_new_seq_b_aligned = ""
        new_nature_masked_seq_a = ""
        for idx, v in enumerate(insertion_mask):
            if v == 1:
                assert new_seq_a_aligned[idx] in ["A", "T", "C", "G"]
                assert new_seq_b_aligned[idx] == "-"
            else:
                new_new_seq_a_aligned += new_seq_a_aligned[idx]
                new_new_seq_b_aligned += new_seq_b_aligned[idx]
                new_nature_masked_seq_a += nature_masked_seq_a[idx]

        assert len(set(list(new_new_seq_b_aligned))) == 4
        assert "-" not in new_new_seq_b_aligned
        assert len(set(list(new_new_seq_a_aligned))) == 4
        assert "-" not in new_new_seq_a_aligned
        nature_mutation_rate = nature_masked_num/len(new_nature_masked_seq_a)

        assert len(new_nature_masked_seq_a) == len(new_new_seq_a_aligned)
        assert len(new_nature_masked_seq_a) == len(new_new_seq_b_aligned)

        print("masked num: %d, mutation rate: %f" % (nature_masked_num, nature_mutation_rate))
        print("-" * 100)
        if nature_mutation_rate <= 0.3:
            seq_len = len(new_nature_masked_seq_a)
            if seq_len > 4096:
                continue
            nature_selected_num += 1
            obj_label_recover = {"token_level": {"gene_mask": new_new_seq_b_aligned}}
            nature_selected_recover_data.append([
                seq_id_a,
                "gene",
                new_nature_masked_seq_a,
                json.dumps(obj_label_recover, ensure_ascii=False),
                raw_seq_a,
                raw_seq_b,
                seq_a_aligned,
                seq_b_aligned,
                nature_masked_num,
                nature_mutation_rate
            ])
            obj_label_retain = {"token_level": {"gene_mask": new_new_seq_a_aligned}}
            nature_selected_retain_data.append([
                seq_id_a,
                "gene",
                new_nature_masked_seq_a,
                json.dumps(obj_label_retain, ensure_ascii=False),
                raw_seq_a,
                raw_seq_b,
                seq_a_aligned,
                seq_b_aligned,
                nature_masked_num,
                nature_mutation_rate
            ])
            nature_seq_len_list.append(seq_len)
            nature_mlm_mask_num_list.append(nature_masked_num)
            nature_mlm_probability_list.append(nature_mutation_rate)

    total_num += 1

csv_writer(
    dataset=[v[:4] for v in nature_selected_recover_data],
    handle="../../data/pseudogene/only_mismatch/pseudogene_nature_selected_only_mismatch_recover.csv",
    header=["obj_id", "obj_type", "obj_seq", "obj_label"]
)


csv_writer(
    dataset=nature_selected_recover_data,
    handle="../../data/pseudogene/only_mismatch/pseudogene_nature_selected_details_only_mismatch_recover.csv",
    header=["obj_id", "obj_type", "obj_seq", "obj_label", "raw_seq_a", "raw_seq_b", "seq_a_aligned", "seq_b_aligned", "mask_num:",  "mlm_probability"]
)

csv_writer(
    dataset=[v[:4] for v in nature_selected_retain_data],
    handle="../../data/pseudogene/only_mismatch/pseudogene_nature_selected_only_mismatch_retain.csv",
    header=["obj_id", "obj_type", "obj_seq", "obj_label"]
)

csv_writer(
    dataset=nature_selected_retain_data,
    handle="../../data/pseudogene/only_mismatch/pseudogene_nature_selected_details_only_mismatch_retain.csv",
    header=["obj_id", "obj_type", "obj_seq", "obj_label", "raw_seq_a", "raw_seq_b", "seq_a_aligned", "seq_b_aligned", "mask_num:",  "mlm_probability"]
)
print("nature_selected_num: %d, total_num: %d" % (nature_selected_num, total_num))
print("nature_selected_recover_data: %d, nature_selected_retain_data: %d" % (len(nature_selected_recover_data), len(nature_selected_retain_data)))

print("nature_seq_len_list:")
print(nature_seq_len_list)
print("sum nature_seq_len_list: %d" % (sum(nature_seq_len_list)))

print("nature_mlm_mask_num_list:")
print(nature_mlm_mask_num_list)
print("sum nature_mlm_mask_num_list: %d" % (sum(nature_mlm_mask_num_list)))

print("nature_mlm_probability_list:")
print(nature_mlm_probability_list)
print("sum nature_mlm_probability_list: %f" % (sum(nature_mlm_probability_list)))

def bernoulli_sampling(seq_len, mask_num):
    bernoulli_samples = np.random.binomial(1, mask_num/seq_len, size=seq_len)
    selected_indices = np.where(bernoulli_samples == 1)[0]
    while len(set(selected_indices)) != mask_num:
        bernoulli_samples = np.random.binomial(1, mask_num/seq_len, size=seq_len)
        selected_indices = np.where(bernoulli_samples == 1)[0]
    return set(selected_indices)


def uniform_sampling(seq_len, mask_num):
    l = list(range(0, seq_len))
    for _ in range(10):
        random.shuffle(l)
    selected_indices = l[0:mask_num]
    return selected_indices

# manual
manual_random_recover_data = []
for row in nature_selected_recover_data:
    obj_label = json.loads(row[3])
    seq_b_aligned = obj_label["token_level"]["gene_mask"]
    seq_len = len(seq_b_aligned)

    mask_num = row[-2]
    masked_indices = uniform_sampling(seq_len, mask_num)
    mask_seq = ['-' if ch_idx in masked_indices else ch for ch_idx, ch in enumerate(seq_b_aligned)]
    mask_seq = "".join(mask_seq)
    testing_mask_num = sum([1 for ch in mask_seq if ch == "-"])
    assert testing_mask_num == mask_num
    assert len(mask_seq) == len(obj_label["token_level"]["gene_mask"])
    manual_random_recover_data.append([row[0], row[1], mask_seq, json.dumps(obj_label, ensure_ascii=False), *row[4:]])

csv_writer(
    dataset=[v[:4] for v in manual_random_recover_data],
    handle="../../data/pseudogene/only_mismatch/pseudogene_manual_random_only_mismatch_recover.csv",
    header=["obj_id", "obj_type", "obj_seq", "obj_label"]
)
csv_writer(
    dataset=manual_random_recover_data,
    handle="../../data/pseudogene/only_mismatch/pseudogene_manual_random_details_only_mismatch_recover.csv",
    header=["obj_id", "obj_type", "obj_seq", "obj_label", "raw_seq_a", "raw_seq_b", "seq_a_aligned", "seq_b_aligned", "mask_num", "mlm_probability"]
)

manual_random_retain_data = []
for row in nature_selected_retain_data:
    obj_label = json.loads(row[3])
    seq_b_aligned = obj_label["token_level"]["gene_mask"]
    seq_len = len(seq_b_aligned)

    mask_num = row[-2]
    masked_indices = uniform_sampling(seq_len, mask_num)
    mask_seq = ['-' if ch_idx in masked_indices else ch for ch_idx, ch in enumerate(seq_b_aligned)]
    mask_seq = "".join(mask_seq)
    testing_mask_num = sum([1 for ch in mask_seq if ch == "-"])
    assert testing_mask_num == mask_num
    assert len(mask_seq) == len(obj_label["token_level"]["gene_mask"])
    manual_random_retain_data.append([row[0], row[1], mask_seq, json.dumps(obj_label, ensure_ascii=False), *row[4:]])
csv_writer(
    dataset=[v[:4] for v in manual_random_retain_data],
    handle="../../data/pseudogene/only_mismatch/pseudogene_manual_random_only_mismatch_retain.csv",
    header=["obj_id", "obj_type", "obj_seq", "obj_label"]
)
csv_writer(
    dataset=manual_random_retain_data,
    handle="../../data/pseudogene/only_mismatch/pseudogene_manual_random_details_only_mismatch_retain.csv",
    header=["obj_id", "obj_type", "obj_seq", "obj_label", "raw_seq_a", "raw_seq_b", "seq_a_aligned", "seq_b_aligned", "mask_num", "mlm_probability"]
)

print("manual_random_recover_data: %d, manual_random_retain_data: %d" % (len(manual_random_recover_data), len(manual_random_retain_data)))

print("seq len stats:")
seq_len_list = np.array(nature_seq_len_list, dtype=int)

print("min: %d, max: %d, mean: %f, median: %d, 25: %d, 45: %d, 60: %d, 75: %d, 80: %d, 85: %d, 90: %d, 95: %d, 99: %d" %(
    np.min(seq_len_list),
    np.max(seq_len_list),
    np.mean(seq_len_list),
    np.median(seq_len_list),
    np.percentile(seq_len_list, 25),
    np.percentile(seq_len_list, 45),
    np.percentile(seq_len_list, 60),
    np.percentile(seq_len_list, 75),
    np.percentile(seq_len_list, 80),
    np.percentile(seq_len_list, 85),
    np.percentile(seq_len_list, 90),
    np.percentile(seq_len_list, 95),
    np.percentile(seq_len_list, 99)
))

print("mlm probability:")
mlm_probability_list = np.array(nature_mlm_probability_list, dtype=float)
print("min: %f, max: %f, mean: %f, median: %f, 25: %f, 45: %f, 60: %f, 75: %f, 80: %f, 85: %f, 90: %f, 95: %f, 99: %f" %(
    np.min(mlm_probability_list),
    np.max(mlm_probability_list),
    np.mean(mlm_probability_list),
    np.median(mlm_probability_list),
    np.percentile(mlm_probability_list, 25),
    np.percentile(mlm_probability_list, 45),
    np.percentile(mlm_probability_list, 60),
    np.percentile(mlm_probability_list, 75),
    np.percentile(mlm_probability_list, 80),
    np.percentile(mlm_probability_list, 85),
    np.percentile(mlm_probability_list, 90),
    np.percentile(mlm_probability_list, 95),
    np.percentile(mlm_probability_list, 99)
))

print("mask num:")
mask_num_list = np.array(nature_mlm_mask_num_list, dtype=float)
print("min: %f, max: %f, mean: %f, median: %f, 25: %f, 45: %f, 60: %f, 75: %f, 80: %f, 85: %f, 90: %f, 95: %f, 99: %f" %(
    np.min(mask_num_list),
    np.max(mask_num_list),
    np.mean(mask_num_list),
    np.median(mask_num_list),
    np.percentile(mask_num_list, 25),
    np.percentile(mask_num_list, 45),
    np.percentile(mask_num_list, 60),
    np.percentile(mask_num_list, 75),
    np.percentile(mask_num_list, 80),
    np.percentile(mask_num_list, 85),
    np.percentile(mask_num_list, 90),
    np.percentile(mask_num_list, 95),
    np.percentile(mask_num_list, 99)
))

"""
----------------------------------------------------------------------------------------------------
nature_selected_num: 160, total_num: 167
nature_selected_recover_data: 160, nature_selected_retain_data: 160

nature_seq_len_list:
[1042, 1542, 684, 207, 681, 469, 500, 940, 879, 939, 502, 736, 500, 781, 758, 506, 506, 506, 736, 781, 500, 500, 500, 438,1182, 291, 879, 1593, 1593, 1593, 1141, 879, 879, 525, 1141, 525, 1812, 1378, 784, 681, 684, 525, 1812, 1378, 297, 939, 939, 462, 957, 3011, 3013, 3011, 3011, 513, 507, 513, 505, 1047, 1047, 500, 500, 957, 945, 939, 500, 939, 954, 936, 978, 930,1063, 930, 1182, 981, 930, 981, 945, 945, 924, 933, 936, 936, 1036, 930, 1037, 879, 1182, 942, 930, 981, 879, 879, 535, 496, 542, 486, 535, 467, 535, 489, 518, 535, 535, 458, 533, 467, 436, 489, 535, 534, 436, 496, 535, 542, 528, 436, 494, 496, 480, 438, 480, 438, 436, 939, 939, 502, 458, 526, 540, 502, 458, 502, 540, 526, 496, 444, 939, 939, 784, 879, 462, 975, 939, 3399, 975, 500, 736, 736, 477, 500, 582, 498, 474, 535, 496, 496, 879, 180, 2803, 500]
sum nature_seq_len_list: 131034

nature_mlm_mask_num_list:
[17, 20, 12, 5, 10, 4, 41, 119, 97, 53, 2, 160, 57, 10, 11, 80, 4, 57, 162, 122, 40, 43, 43, 48, 66, 12, 34, 11, 9, 5, 249, 64, 63, 57, 159, 46, 222, 141, 95, 80, 67, 26, 127, 59, 15, 8, 7, 42, 93, 5, 7, 105, 108, 43, 35, 29, 30, 122, 132, 40, 52, 269, 78, 8, 35, 8, 174, 109, 97, 141, 56, 22, 26, 46, 23, 239, 54, 53, 247, 79, 87, 97, 300, 50, 102, 93, 7, 185, 229, 17, 82, 75, 63, 62, 53, 36, 93, 93, 84, 47, 60, 71, 50, 30, 44, 81, 26, 54, 62, 45, 33, 53, 32, 53, 79, 37, 20, 43, 24, 51, 24, 51, 93, 178, 5, 82, 1, 28, 16, 70, 3, 70, 18, 26, 65, 4, 6, 89, 127, 64, 52, 30, 8, 551, 29, 43, 100, 93, 71, 110, 89, 25, 55, 126, 109, 124, 58, 19, 4, 44]
sum nature_mlm_mask_num_list: 11009

nature_mlm_probability_list:
[0.016314779270633396, 0.01297016861219196, 0.017543859649122806, 0.024154589371980676, 0.014684287812041116, 0.008528784648187633, 0.082, 0.12659574468085105, 0.11035267349260523, 0.05644302449414271, 0.00398406374501992, 0.21739130434782608, 0.114, 0.012804097311139564, 0.014511873350923483, 0.15810276679841898, 0.007905138339920948, 0.11264822134387352, 0.22010869565217392, 0.15620998719590268, 0.08, 0.086, 0.086, 0.1095890410958904, 0.05583756345177665, 0.041237113402061855, 0.038680318543799774, 0.006905210295040804, 0.005649717514124294, 0.003138731952291274, 0.2182296231375986, 0.07281001137656427, 0.07167235494880546, 0.10857142857142857, 0.13935144609991235, 0.08761904761904762, 0.12251655629139073, 0.102322206095791, 0.1211734693877551, 0.11747430249632893, 0.097953216374269, 0.049523809523809526, 0.07008830022075055, 0.04281567489114659,0.050505050505050504, 0.008519701810436636, 0.007454739084132056, 0.09090909090909091, 0.09717868338557993, 0.0016605778811026237, 0.0023232658479920344, 0.0348721355031551, 0.03586848223181667, 0.08382066276803118, 0.06903353057199212, 0.056530214424951264, 0.0594059405940594, 0.11652340019102196, 0.12607449856733524, 0.08, 0.104, 0.28108672936259144, 0.08253968253968254, 0.008519701810436636, 0.07, 0.008519701810436636, 0.18238993710691823, 0.11645299145299146, 0.09918200408997956, 0.15161290322580645, 0.05268109125117592, 0.023655913978494623, 0.021996615905245348, 0.046890927624872576, 0.024731182795698924, 0.2436289500509684, 0.05714285714285714, 0.056084656084656084, 0.26731601731601734, 0.08467309753483387, 0.09294871794871795, 0.10363247863247864, 0.28957528957528955, 0.053763440860215055, 0.09836065573770492, 0.10580204778156997, 0.005922165820642978, 0.19639065817409768, 0.24623655913978496, 0.017329255861365953, 0.09328782707622298, 0.08532423208191127, 0.11775700934579439, 0.125, 0.09778597785977859, 0.07407407407407407, 0.17383177570093458, 0.19914346895074947, 0.15700934579439252, 0.09611451942740286, 0.11583011583011583, 0.13271028037383178, 0.09345794392523364, 0.06550218340611354, 0.0825515947467167, 0.1734475374732334, 0.05963302752293578, 0.11042944785276074, 0.11588785046728972, 0.08426966292134831, 0.07568807339449542, 0.10685483870967742, 0.059813084112149535, 0.09778597785977859, 0.14962121212121213, 0.08486238532110092, 0.04048582995951417, 0.08669354838709678, 0.05, 0.11643835616438356, 0.05, 0.11643835616438356, 0.21330275229357798, 0.18956336528221512, 0.005324813631522897, 0.16334661354581673, 0.002183406113537118, 0.053231939163498096, 0.02962962962962963, 0.1394422310756972, 0.006550218340611353, 0.1394422310756972, 0.03333333333333333, 0.049429657794676805, 0.1310483870967742, 0.009009009009009009, 0.006389776357827476, 0.09478168264110756, 0.16198979591836735, 0.07281001137656427, 0.11255411255411256, 0.03076923076923077, 0.008519701810436636, 0.16210650191232714, 0.029743589743589743, 0.086, 0.1358695652173913, 0.12635869565217392, 0.1488469601677149, 0.22, 0.15292096219931273, 0.050200803212851405, 0.1160337552742616, 0.23551401869158878, 0.21975806451612903, 0.25, 0.06598407281001138, 0.10555555555555556, 0.0014270424545130217, 0.088]
sum nature_mlm_probability_list: 14.67086043154312

manual_random_recover_data: 160, manual_random_retain_data: 160

seq len stats:
min: 180, max: 3399, mean: 818.962500, median: 681, 25: 500, 45: 535, 60: 879, 75: 939, 80: 957, 85: 1036, 90: 1182, 95: 1603, 99: 3011

mlm probability:
min: 0.001427, max: 0.289575, mean: 0.091693, median: 0.086000, 25: 0.042421, 45: 0.082297, 60: 0.098689, 75: 0.121509, 80: 0.136566, 85: 0.156330, 90: 0.183107, 95: 0.220005, 99: 0.272962

mask num:
min: 1.000000, max: 551.000000, mean: 68.806250, median: 53.000000, 25: 26.000000, 45: 49.100000, 60: 63.000000, 75: 93.000000, 80: 97.000000, 85: 111.350000, 90: 132.900000, 95: 186.850000, 99: 281.710000
"""
