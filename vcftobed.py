#!/usr/bin/env python
"""
VCF到BEDPE转换和SV注释
"""
import os
import re
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy import signal as scisig
from scipy.stats import binom

import plotsv as sv_plot

warnings.filterwarnings("ignore")


class SVClusterDetector:
    
    def __init__(self, genome_size=3e9, min_bps=10, peak_factor=10):
        self.genome_size = genome_size
        self.min_bps = min_bps
        self.peak_factor = peak_factor
    
    def unique_py(self, seqlist):
        seen = set()
        seen_add = seen.add
        return [x for x in seqlist if not (x in seen or seen_add(x))]
    
    def calc_intermut_dist(self, subs_type, first_chrom_na=False):
        subs_type_processed = subs_type.copy()
        chr_list = self.unique_py(subs_type["chr"])
        pos_array_im = subs_type["position"].values
        index_orig_df = np.arange(len(subs_type_processed))
        distPrev_list = []
        prevPos_list = []

        for c in chr_list:
            inds_chr = np.where(subs_type["chr"] == c)
            pos_array_im_c = np.sort(pos_array_im[inds_chr])
            index_orig_df[inds_chr] = index_orig_df[inds_chr][
                np.argsort(pos_array_im[inds_chr])
            ]

            if first_chrom_na:
                prevPos_arr_c = np.hstack((np.nan, pos_array_im_c.flatten()[:-1]))
            else:
                prevPos_arr_c = np.hstack((0, pos_array_im_c.flatten()[:-1]))
            distPrev_arr_c = pos_array_im_c - prevPos_arr_c
            distPrev_arr_c[distPrev_arr_c == 0] = 1
            distPrev_list = np.append(distPrev_list, distPrev_arr_c.astype(int)).flatten()
            prevPos_list = np.append(prevPos_list, prevPos_arr_c.astype(int)).flatten()
            
        subs_type_processed = subs_type_processed.reindex(index_orig_df).reset_index(drop=True)
        subs_type_processed["prevPos"] = prevPos_list
        subs_type_processed["distPrev"] = distPrev_list
        return subs_type_processed
    
    def compute_mad(self, v):
        mad = np.median(np.abs(v - np.median(v)))
        return mad
    
    def get_mad(self, x, k=25):
        x = x[x != 0]
        run_median = scisig.medfilt(x, k)
        dif = x - run_median
        mad = self.compute_mad(dif)
        return mad
    
    def exact_pcf(self, y, kmin, gamma, flag=True):
        if flag:
            yest = np.random.rand(len(y))
        else:
            yest = flag
        N = len(y)
        yhat = np.zeros(N)
        
        if N < 2 * kmin:
            if flag:
                results = {
                    "Lengde": N,
                    "sta": 1,
                    "mean": np.mean(y),
                    "nIntervals": 1,
                    "yhat": np.repeat(np.mean(y), N, axis=0),
                }
                return results
            else:
                results = {"Lengde": N, "sta": 1, "mean": np.mean(y), "nIntervals": 1}
                return results

        initSum = sum(y[0:kmin])
        initKvad = sum(y[0:kmin] ** 2)
        initAve = initSum / kmin
        bestCost = np.zeros(N)
        bestCost[kmin - 1] = initKvad - initSum * initAve
        bestSplit = np.zeros(N)
        bestAver = np.zeros(N)
        bestAver[kmin - 1] = initAve
        Sum = np.zeros(N)
        Kvad = np.zeros(N)
        Aver = np.zeros(N)
        Cost = np.zeros(N)
        kminP1 = kmin + 1
        
        for k in range(kminP1, 2 * kmin):
            Sum[kminP1 - 1 : k] = Sum[kminP1 - 1 : k] + y[k - 1]
            Aver[kminP1 - 1 : k] = Sum[kminP1 - 1 : k] / (range((k - kmin), 0, -1))
            Kvad[kminP1 - 1 : k] = Kvad[kminP1 - 1 : k] + (y[k - 1] ** 2)
            bestAver[k - 1] = (initSum + Sum[kminP1 - 1]) / k
            bestCost[k - 1] = (initKvad + Kvad[kminP1 - 1]) - (k * bestAver[k - 1] ** 2)

        for n in range(2 * kmin, N + 1):
            yn = y[n - 1]
            yn2 = y[n - 1] ** 2
            Sum[kminP1 - 1 : n] = Sum[kminP1 - 1 : n] + yn
            Aver[kminP1 - 1 : n] = Sum[kminP1 - 1 : n] / (range((n - kmin), 0, -1))
            Kvad[kminP1 - 1 : n] = Kvad[kminP1 - 1 : n] + yn2
            nMkminP1 = n - kmin + 1
            Cost[kminP1 - 1 : nMkminP1] = (
                bestCost[kmin - 1 : (n - kmin)]
                + Kvad[kminP1 - 1 : nMkminP1]
                - Sum[kminP1 - 1 : nMkminP1] * Aver[kminP1 - 1 : nMkminP1]
                + gamma
            )
            Pos = np.argmin(Cost[kminP1 - 1 : nMkminP1]) + kmin
            cost = Cost[Pos]
            aver = Aver[Pos]
            totAver = (Sum[kminP1 - 1] + initSum) / n
            totCost = (Kvad[kminP1 - 1] + initKvad) - n * totAver * totAver
            if totCost < cost:
                Pos = 1
                cost = totCost
                aver = totAver
            bestCost[n - 1] = cost
            bestAver[n - 1] = aver
            bestSplit[n - 1] = Pos
        
        n = N
        antInt = 1
        
        if yest.any():
            while n > 0:
                yhat[(bestSplit[n - 1]) : n] = bestAver[n - 1]
                n = bestSplit[n - 1]
                antInt = antInt + 1
        else:
            while n > 0:
                n = bestSplit[n - 1]
                antInt = antInt + 1
        
        antInt = antInt - 1
        n = N
        lengde = np.repeat(0, antInt, axis=0)
        start = np.repeat(0, antInt, axis=0)
        verdi = np.repeat(0, antInt, axis=0)
        oldSplit = n
        antall = antInt
        while n > 0:
            start[antall - 1] = bestSplit[n - 1] + 1
            lengde[antall - 1] = oldSplit - bestSplit[n - 1]
            verdi[antall - 1] = bestAver[n - 1]
            n = bestSplit[n - 1]
            oldSplit = n
            antall = antall - 1
        
        if yest.any():
            results = {
                "Lengde": lengde,
                "sta": start,
                "mean": verdi,
                "nIntervals": antInt,
                "yhat": yhat,
            }
            return results
        else:
            results = {"Lengde": lengde, "sta": start, "mean": verdi, "nIntervals": antInt}
            return results
    
    def pbinom(self, q, size, prob=0.5):
        result = binom.cdf(k=q, n=size, p=prob, loc=0)
        return result
    
    def assign_pvalues(self, kat_regions, chrom_bps, bp_rate=np.nan):
        if len(kat_regions) > 0:
            if np.isnan(bp_rate):
                bp_vals = chrom_bps["pos"].values
                left_bp = np.min(bp_vals)
                right_bp = np.max(bp_vals)
                bp_rate = len(bp_vals) / (right_bp - left_bp)
            kat_regions["pvalue"] = 1 - self.pbinom(
                kat_regions["number_bps"].values,
                kat_regions["end_bp"].values - kat_regions["start_bp"].values,
                bp_rate,
            )
            kat_regions["d_seg"] = kat_regions["number_bps"].values / (
                kat_regions["end_bp"].values - kat_regions["start_bp"].values
            )
            kat_regions["rate_factor"] = kat_regions["d_seg"] / bp_rate
        return kat_regions
    
    def hotspot_info(self, kat_regions_all, subs, seg_inter_dist):
        if len(kat_regions_all) > 0:
            pos_arr = subs["pos"].values
            kat_firstBp = kat_regions_all["firstBp"].values
            kat_lastBp = kat_regions_all["lastBp"].values
            
            for index in range(len(kat_regions_all)):
                subs_hotspot = pos_arr[int(kat_firstBp[index]) : int(kat_lastBp[index]) + 1]
                kat_regions_all.loc[index, "start_bp"] = np.min(subs_hotspot)
                kat_regions_all.loc[index, "end_bp"] = np.max(subs_hotspot)
                kat_regions_all.loc[index, "length_bp"] = (
                    kat_regions_all.loc[index, "end_bp"] - kat_regions_all.loc[index, "start_bp"]
                )
                kat_regions_all.loc[index, "number_bps"] = len(subs_hotspot)
                
                if len(seg_inter_dist) > 0 and np.isnan(kat_regions_all.loc[index, "avgDist_bp"]):
                    kat_regions_all.loc[index, "avgDist_bp"] = np.mean(
                        seg_inter_dist[int(kat_firstBp[index]) : (int(kat_lastBp[index]) + 1)]
                    )
                
                kat_regions_all.loc[index, "no_samples"] = len(
                    self.unique_py(
                        [
                            subs["sample"].iloc[val]
                            for val in range(
                                int(kat_firstBp[index]), int(kat_lastBp[index]) + 1
                            )
                        ]
                    )
                )
        return kat_regions_all
    
    def extract_kat_regions(self, res, imd_thresh, subs, kmin_samples=1, 
                           pvalue_thresh=0.05, rate_factor_thresh=1,
                           kmin_filter=None, bp_rate=np.nan, do_merging=True):
        seg_inter_dist = res["yhat"]
        kataegis_threshold = imd_thresh
        kat_regions_all = pd.DataFrame()
        positions = subs["pos"]
        katLoci = seg_inter_dist <= kataegis_threshold
        
        if sum(katLoci) > 0:
            start_regions = (
                np.asarray(
                    np.where(
                        katLoci[1:] & ~(katLoci[:-1])
                        | (
                            (katLoci[1:] & (katLoci[:-1]))
                            & (seg_inter_dist[1:] != seg_inter_dist[:len(katLoci) - 1])
                        )
                    )
                )[0]
                + 1
            )
            if katLoci[0]:
                start_regions = np.hstack((0, start_regions))
            end_regions = np.asarray(
                np.where(
                    ~katLoci[1:] & (katLoci[:-1])
                    | (
                        (katLoci[1:] & (katLoci[:-1]))
                        & (seg_inter_dist[1:] != seg_inter_dist[:-1])
                    )
                )
            )[0]
            if katLoci[-1]:
                end_regions = np.hstack((end_regions, len(katLoci) - 1))

            if len(end_regions) + len(start_regions) > 0:
                if (len(end_regions) == 1) & (len(start_regions) == 0):
                    start_regions = 0
                elif (len(end_regions) == 0) & (len(start_regions) == 1):
                    end_regions = len(positions) - 1
                elif (end_regions[0] < start_regions[0]) & (start_regions[-1] > end_regions[-1]):
                    start_regions = np.hstack((0, start_regions))
                    end_regions = np.hstack((end_regions, len(positions) - 1))
                elif end_regions[0] < start_regions[0]:
                    start_regions = np.hstack((0, start_regions))
                elif start_regions[-1] > end_regions[-1]:
                    end_regions = np.hstack((end_regions, len(positions) - 1))
            
            columnslist = [
                "chr", "start_bp", "end_bp", "length_bp", "number_bps",
                "number_bps_clustered", "avgDist_bp", "no_samples",
                "no_del", "no_dup", "no_inv", "np_trn", "firstBp", "lastBp"
            ]
            temp = np.full((len(start_regions), len(columnslist)), np.nan)
            kat_regions_all = pd.DataFrame(temp, columns=columnslist)
            kat_regions_all["chr"] = subs["chr"].iloc[0]
            kat_regions_all["firstBp"] = start_regions
            kat_regions_all["lastBp"] = end_regions
            
            kat_regions_all = self.hotspot_info(kat_regions_all, subs, seg_inter_dist)
            
            if (not kat_regions_all.empty) & (len(kat_regions_all) > 0):
                kat_regions_all = kat_regions_all[kat_regions_all["no_samples"] >= kmin_samples]
            
            if kmin_filter is not None and not np.isnan(kmin_filter):
                kat_regions_all = kat_regions_all[kat_regions_all["number_bps"] >= kmin_filter]
            
            if (not kat_regions_all.empty) & (len(kat_regions_all) > 0):
                kat_regions_all = self.assign_pvalues(kat_regions_all, subs, bp_rate)
                kat_regions_all = kat_regions_all[kat_regions_all["pvalue"] <= pvalue_thresh]
                kat_regions_all = kat_regions_all[kat_regions_all["rate_factor"] >= rate_factor_thresh]
            
            if do_merging and len(kat_regions_all) > 1:
                kat_regions_all = kat_regions_all.reset_index(drop=True)
                for r in range(1, len(kat_regions_all)):
                    if kat_regions_all.loc[r-1, "lastBp"] == kat_regions_all.loc[r, "firstBp"] - 1:
                        kat_regions_all.loc[r, "firstBp"] = kat_regions_all.loc[r-1, "firstBp"]
                        kat_regions_all.loc[r-1, "firstBp"] = np.nan
                        kat_regions_all.loc[r-1, "lastBp"] = np.nan
                        kat_regions_all.loc[r, "avgDist_bp"] = np.nan
                
                kat_regions_all = kat_regions_all.dropna(subset=['firstBp']).reset_index(drop=True)
                
                if not kat_regions_all.empty:
                    kat_regions_all = self.hotspot_info(kat_regions_all, subs, seg_inter_dist)
                    kat_regions_all = self.assign_pvalues(kat_regions_all, subs, bp_rate)
        
        return kat_regions_all
    
    def annotate_bedpe(self, sv_bedpe):
        sv_bedpe = sv_bedpe.reset_index(drop=True)
        sv_bedpe["id"] = sv_bedpe.index
        
        left = pd.DataFrame({
            "chrom1": sv_bedpe["chrom1"],
            "start1": sv_bedpe["start1"],
            "sample": sv_bedpe["sample"],
            "id": sv_bedpe["id"]
        })
        right = pd.DataFrame({
            "chrom2": sv_bedpe["chrom2"],
            "start2": sv_bedpe["start2"],
            "sample": sv_bedpe["sample"],
            "id": sv_bedpe["id"]
        })
        
        cncd = pd.DataFrame(
            np.concatenate([left.values, right.values]),
            columns=("chr", "position", "sample", "id"),
        )
        cncd["isLeft"] = [True] * len(left) + [False] * len(right)
        
        sample_bps = pd.DataFrame(columns=cncd.columns)
        for chromi in self.unique_py(cncd["chr"]):
            sample_bps = pd.concat(
                [sample_bps, cncd[cncd["chr"] == chromi].sort_values("position")],
                ignore_index=True,
            )
        
        sample_bps.index = pd.RangeIndex(len(sample_bps.index))
        
        sample_bps["intermut_dist"] = self.calc_intermut_dist(
            sample_bps, first_chrom_na=False
        )["distPrev"].values
        
        exp_dist = self.genome_size / len(sample_bps)
        imd_thresh = exp_dist / self.peak_factor
        
        sdev = self.get_mad(sample_bps["intermut_dist"].values)
        gamma = 25 * sdev
        
        sample_bps["is_clustered_single"] = False
        sample_bps["mean_intermut_dist"] = np.nan
        all_kat_regions = pd.DataFrame()
        
        for chrom in self.unique_py(sample_bps["chr"]):
            chrom_mask = sample_bps["chr"] == chrom
            if sum(chrom_mask) >= self.min_bps:
                data_points = sample_bps.loc[chrom_mask, "intermut_dist"].values
                kmin = 10
                if kmin > len(data_points) // 2:
                    kmin = max(2, len(data_points) // 3)
                
                try:
                    res = self.exact_pcf(data_points, kmin, gamma, True)
                    sample_bps.loc[chrom_mask, "mean_intermut_dist"] = res["yhat"]
                    
                    subs = pd.DataFrame({
                        "chr": sample_bps.loc[chrom_mask, "chr"].values,
                        "pos": sample_bps.loc[chrom_mask, "position"].values,
                        "sample": sample_bps.loc[chrom_mask, "sample"].values
                    })
                    
                    kat_regions = self.extract_kat_regions(
                        res, imd_thresh, subs,
                        kmin_samples=1, pvalue_thresh=1, rate_factor_thresh=1,
                        kmin_filter=10, bp_rate=np.nan, do_merging=True
                    )
                    
                    if not kat_regions.empty:
                        all_kat_regions = pd.concat([all_kat_regions, kat_regions], ignore_index=True)
                        
                        for _, region in kat_regions.iterrows():
                            first_idx = int(region["firstBp"])
                            last_idx = int(region["lastBp"])
                            indices = np.where(chrom_mask)[0][first_idx:last_idx+1]
                            sample_bps.loc[sample_bps.index[indices], "is_clustered_single"] = True
                            
                except Exception as e:
                    sample_bps.loc[chrom_mask, "mean_intermut_dist"] = np.mean(data_points)
            else:
                if sum(chrom_mask) > 0:
                    sample_bps.loc[chrom_mask, "mean_intermut_dist"] = np.mean(
                        sample_bps.loc[chrom_mask, "intermut_dist"].values
                    )
        
        sample_bps["is_clustered"] = sample_bps["is_clustered_single"]
        sv_bedpe["is_clustered"] = False
        
        clustered_ids = set(sample_bps[sample_bps["is_clustered"]]["id"].values)
        sv_bedpe.loc[sv_bedpe["id"].isin(clustered_ids), "is_clustered"] = True
        
        return sv_bedpe, all_kat_regions


class VCFToBEDPEConverter:
    """VCF转BEDPE和注释TSV转换器"""
    
    def __init__(self, input_dir, output_dir=None):
        self.input_dir = input_dir.rstrip('/') + '/'
        # 使用输入文件夹名称作为项目名称
        self.project_name = os.path.basename(os.path.normpath(input_dir))
        
        if output_dir is None:
            self.output_dir = self.input_dir
        else:
            self.output_dir = output_dir.rstrip('/') + '/'
        
        self.all_svs = []
        self.dropped_svs = []
        self.all_kat_regions = pd.DataFrame()
        self.cluster_detector = SVClusterDetector()
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def parse_vcf_to_bedpe(self, vcf_path, sample_name):
        svs = []
        dropped = []
        
        svtype_mapping = {
            'DEL': 'deletion',
            'DUP': 'tandem-duplication',
            'INV': 'inversion',
            'BND': 'translocation',
        }
        
        with open(vcf_path, 'r') as vcf_file:
            for line_num, line in enumerate(vcf_file):
                if line.startswith('#'):
                    continue
                
                fields = line.strip().split('\t')
                if len(fields) < 8:
                    continue
                
                chrom = fields[0]
                pos = int(fields[1])
                alt = fields[4]
                qual = fields[5]
                filter_val = fields[6]
                info = fields[7]
                
                # 解析INFO
                info_dict = {}
                for item in info.split(';'):
                    if '=' in item:
                        key, value = item.split('=', 1)
                        info_dict[key] = value
                    else:
                        info_dict[item] = True
                
                svtype = info_dict.get('SVTYPE', 'UNKNOWN')
                svclass = svtype_mapping.get(svtype)
                
                if svclass is None:
                    continue
                
                if svclass == 'translocation':
                    # 易位解析逻辑
                    if 'SECONDARY' in info_dict:
                        continue
                    
                    sep = '['
                    if sep not in alt:
                        sep = ']'
                    
                    pattern = r'\%s(.+?)\%s' % (sep, sep)
                    matches = re.findall(pattern, alt)
                    
                    if not matches:
                        dropped.append({
                            'sample': sample_name,
                            'chrom1': chrom,
                            'start1': pos,
                            'end1': pos,
                            'chrom2': chrom,
                            'start2': pos,
                            'end2': pos,
                            'svclass': svclass,
                            'length': 0,
                            'size_bin': 'translocation',
                            'reason': '无法解析易位断点位置'
                        })
                        continue
                    
                    chrom2_pos = matches[0].split(':')
                    if len(chrom2_pos) != 2:
                        dropped.append({
                            'sample': sample_name,
                            'chrom1': chrom,
                            'start1': pos,
                            'end1': pos,
                            'chrom2': chrom,
                            'start2': pos,
                            'end2': pos,
                            'svclass': svclass,
                            'length': 0,
                            'size_bin': 'translocation',
                            'reason': '易位断点格式错误'
                        })
                        continue
                    
                    chrom2 = chrom2_pos[0]
                    pos2 = int(chrom2_pos[1])
                    
                    cipos = info_dict.get('CIPOS', '0,0')
                    try:
                        cipos_start, cipos_end = map(int, cipos.split(','))
                    except:
                        cipos_start, cipos_end = 0, 0
                    
                    ciend = info_dict.get('CIEND', '0,0')
                    try:
                        ciend_start, ciend_end = map(int, ciend.split(','))
                    except:
                        ciend_start, ciend_end = 0, 0
                    
                    start1 = pos + cipos_start - 1
                    end1 = pos + cipos_end
                    start2 = pos2 + ciend_start - 1
                    end2 = pos2 + ciend_end
                    
                    length = abs(pos2 - pos)
                    size_bin = 'translocation'
                    
                else:
                    # 非易位SV
                    end_pos = info_dict.get('END')
                    if end_pos is None:
                        dropped.append({
                            'sample': sample_name,
                            'chrom1': chrom,
                            'start1': pos,
                            'end1': pos,
                            'chrom2': chrom,
                            'start2': pos,
                            'end2': pos,
                            'svclass': svclass,
                            'length': 0,
                            'size_bin': '',
                            'reason': '缺少END信息'
                        })
                        continue
                    
                    end_pos = int(end_pos)
                    
                    cipos = info_dict.get('CIPOS', '0,0')
                    ciend = info_dict.get('CIEND', '0,0')
                    
                    try:
                        cipos_start, cipos_end = map(int, cipos.split(','))
                        ciend_start, ciend_end = map(int, ciend.split(','))
                    except:
                        cipos_start, cipos_end = 0, 0
                        ciend_start, ciend_end = 0, 0
                    
                    if pos <= end_pos:
                        start1 = pos + cipos_start - 1
                        end1 = pos + cipos_end
                        start2 = end_pos + ciend_start - 1
                        end2 = end_pos + ciend_end
                    else:
                        start1 = end_pos + ciend_start - 1
                        end1 = end_pos + ciend_end
                        start2 = pos + cipos_start - 1
                        end2 = pos + cipos_end
                    
                    chrom2 = chrom
                    length = abs(end_pos - pos)
                    
                    # 过滤小于1KB的非易位SV
                    if length < 1000:
                        dropped.append({
                            'sample': sample_name,
                            'chrom1': chrom,
                            'start1': start1,
                            'end1': end1,
                            'chrom2': chrom2,
                            'start2': start2,
                            'end2': end2,
                            'svclass': svclass,
                            'length': length,
                            'size_bin': '',
                            'reason': f'长度小于1KB ({length} bp)'
                        })
                        continue
                    
                    length_mb = length / 1000000.0
                    if length_mb <= 0.01:
                        size_bin = '1-10Kb'
                    elif length_mb <= 0.1:
                        size_bin = '10-100Kb'
                    elif length_mb <= 1:
                        size_bin = '100Kb-1Mb'
                    elif length_mb <= 10:
                        size_bin = '1Mb-10Mb'
                    else:
                        size_bin = '>10Mb'
                
                sv_info = {
                    'chrom1': chrom,
                    'start1': start1,
                    'end1': end1,
                    'chrom2': chrom2,
                    'start2': start2,
                    'end2': end2,
                    'sample': sample_name,
                    'svclass': svclass,
                    'size_bin': size_bin,
                    'length': length,
                    'is_clustered': False
                }
                
                svs.append(sv_info)
        
        df_kept = pd.DataFrame(svs)
        df_dropped = pd.DataFrame(dropped) if dropped else pd.DataFrame()
        
        return df_kept, df_dropped
    
    def save_bedpe_file(self, sv_df, output_path):
        if not sv_df.empty:
            bedpe_df = sv_df[['chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2', 
                              'sample', 'svclass', 'size_bin', 'length', 'is_clustered']]
            bedpe_df.columns = ['CHROM_A', 'START_A', 'END_A', 'CHROM_B', 'START_B', 'END_B', 
                                'sample', 'svclass', 'size_bin', 'length', 'is_clustered']
            bedpe_df.to_csv(output_path, sep='\t', index=False)
    
    def process_vcf_files(self):
        vcf_files = [f for f in os.listdir(self.input_dir) if f.endswith('.vcf')]
        
        if not vcf_files:
            print(f"错误: 在 {self.input_dir} 中没有找到VCF文件")
            return False
        
        print(f"找到 {len(vcf_files)} 个VCF文件\n")
        
        for vcf_file in vcf_files:
            sample_name = vcf_file.split('.')[0]
            vcf_path = os.path.join(self.input_dir, vcf_file)
            
            sv_df, dropped_df = self.parse_vcf_to_bedpe(vcf_path, sample_name)
            
            if not sv_df.empty:
                self.all_svs.append(sv_df)
                bedpe_output = os.path.join(self.output_dir, f"{sample_name}.bedpe")
                self.save_bedpe_file(sv_df, bedpe_output)
            
            if not dropped_df.empty:
                self.dropped_svs.append(dropped_df)
        
        return True
    
    def run_clustering(self):
        
        all_sv_df = pd.concat(self.all_svs, ignore_index=True)
        clustered_svs = []
        kat_regions_list = []
        
        for sample in all_sv_df['sample'].unique():
            sample_svs = all_sv_df[all_sv_df['sample'] == sample].copy()
            sample_svs = sample_svs.reset_index(drop=True)
            
            if len(sample_svs) < 5:
                sample_svs['is_clustered'] = False
                clustered_svs.append(sample_svs)
                continue
            
            try:
                clustered_sample, kat_regions = self.cluster_detector.annotate_bedpe(sample_svs)
                clustered_svs.append(clustered_sample)
                
                if not kat_regions.empty:
                    kat_regions['sample'] = sample
                    kat_regions_list.append(kat_regions)
                    
                n_clustered = clustered_sample['is_clustered'].sum()
                
            except Exception as e:
                print(f"  样本 {sample} 聚类分析出错: {e}")
                sample_svs['is_clustered'] = False
                clustered_svs.append(sample_svs)
        
        self.all_svs = clustered_svs
        
        if kat_regions_list:
            self.all_kat_regions = pd.concat(kat_regions_list, ignore_index=True)
            kat_regions_file = os.path.join(self.output_dir, f"{self.project_name}.SV_annotated.cluster_regions.tsv")
            self.all_kat_regions.to_csv(kat_regions_file, sep='\t', index=False)
    
    def save_annotated_tsv(self):
        if not self.all_svs:
            print("没有SV数据，无法保存注释文件")
            return None
        
        all_sv_df = pd.concat(self.all_svs, ignore_index=True)
        
        annotated_file = os.path.join(self.output_dir, f"{self.project_name}.SV_annotated.tsv")
        all_sv_df.to_csv(annotated_file, sep='\t', index=False)
        
        if self.dropped_svs:
            all_dropped_df = pd.concat(self.dropped_svs, ignore_index=True)
            dropped_file = os.path.join(self.output_dir, f"{self.project_name}.SV_annotated.dropped.tsv")
            all_dropped_df.to_csv(dropped_file, sep='\t', index=False)
        
        return all_sv_df
    
    def run(self):
        print("开始处理VCF文件...")
        
        success = self.process_vcf_files()
        
        if not success:
            return None
        
        self.run_clustering()
        
        annotated_df = self.save_annotated_tsv()
        
        if annotated_df is not None and not annotated_df.empty:
            try:
                from plotsv import plot_tmb_from_annotated_tsv
                
                # 计算每个样本的 SV 数量并绘制 TMB 图
                output_file = os.path.join(self.output_dir, f"{self.project_name}.SV_annotated.tsv")
                
                plot_tmb_from_annotated_tsv(
                    annotated_tsv=output_file,
                    output_dir=self.output_dir,
                    file_basename=self.project_name,
                    genome_size=2800,
                    cutoff=0
                )
            except ImportError:
                print("[WARNING]无法导入plotsv模块")
            except Exception as e:
                print(f"[WARNING]TMB图生成失败: {e}")
        
        print("\n完成!")
        
        return annotated_df


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='将VCF中的SV转换为BEDPE和注释TSV')
    parser.add_argument('-i', '--input_dir', required=True, help='包含VCF文件的输入目录')
    parser.add_argument('-o', '--output_dir', help='输出目录', default=None)
    
    args = parser.parse_args()
    
    converter = VCFToBEDPEConverter(args.input_dir, args.output_dir)
    annotated_df = converter.run()
    
    if annotated_df is not None:
        print(f"\n")

if __name__ == "__main__":
    main()