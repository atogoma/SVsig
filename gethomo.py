#!/usr/bin/env python
"""
获取断点序列特征
"""
import pysam
import csv
import sys
import multiprocessing as mp
from functools import partial
from collections import defaultdict
import argparse
import os

DEFAULT_WINDOW = 100
DEFAULT_CPU_COUNT = mp.cpu_count()

def _normalize_chrom_name(chrom: str, bam_chroms: list) -> str:
    if chrom in bam_chroms:
        return chrom
    elif f"chr{chrom}" in bam_chroms:
        return f"chr{chrom}"
    elif chrom.replace("chr", "") in bam_chroms:
        return chrom.replace("chr", "")
    else:
        raise ValueError(f"染色体名称'{chrom}'在BAM文件中不存在")

def _fetch_split_reads(bam, chrom, start, end):
    fetched_reads = []
    for read in bam.fetch(chrom, start, end):
        if not read.is_unmapped and read.mapping_quality >= 20:
            fetched_reads.append(read)
    return fetched_reads

def extract_sa_positions(read, target_chr):
    sa_tags = read.get_tag("SA").split(";") if read.has_tag("SA") else []
    sa_info = []
    for sa in sa_tags:
        if sa:
            parts = sa.split(",")
            if len(parts) >= 4:
                chrom, pos_str, cigar = parts[0], parts[1], parts[3]
                op_type, op_len = None, 0
                if cigar:
                    i = 0
                    while i < len(cigar) and cigar[i].isdigit():
                        i += 1
                    if i > 0:
                        op_len = int(cigar[:i])
                        op_type = cigar[i] if i < len(cigar) else None
                if chrom == target_chr:
                    sa_info.append((chrom, int(pos_str), op_type, op_len))
    return sa_info

def get_breakpoint_from_sa(reads, way, target_chr, flag):
    pos_counts = defaultdict(int)
    if flag == 0:
        for read in reads:
            sa_info = extract_sa_positions(read, target_chr)
            for chrom, pos, op_type, op_len in sa_info:
                if chrom == target_chr:
                    if op_type == 'S':
                        adjusted_pos = pos
                        pos_counts[adjusted_pos] += 1
                    if op_type == 'M':
                        adjusted_pos = pos + op_len
                        pos_counts[adjusted_pos] += 1
    else:
        for read in reads:
            sa_info = extract_sa_positions(read, target_chr)
            for chrom, pos, op_type, op_len in sa_info:
                if chrom == target_chr:
                    if op_type == 'S' and way == -1:
                        adjusted_pos = pos
                        pos_counts[adjusted_pos] += 1
                    if op_type == 'M' and way == 1:
                        adjusted_pos = pos + op_len
                        pos_counts[adjusted_pos] += 1
                        
    if pos_counts:
        return max(pos_counts, key=pos_counts.get)
    return 0

def generate_consensus(reads, target_chr, target_pos=None, way=None):
    homolen = 0
    if target_pos is None or not reads:
        return homolen
    if way == 'left':
        for read in reads:
            if read.has_tag("SA") and len(read.cigartuples) == 2:
                sa_info = extract_sa_positions(read, target_chr)
                should_break = False
                for chrom, sa_pos, op_type, op_len in sa_info:
                    if sa_pos == target_pos:
                        if read.cigartuples[1][0] == 4:
                            homolen = read.cigartuples[0][1] - op_len
                            should_break = True
                            break 
                        
                        if read.cigartuples[0][0] == 4:
                            homolen = read.cigartuples[1][1] - op_len
                            should_break = True
                            break 
                if should_break:
                    break
    else:
        for read in reads:
            if read.has_tag("SA") and len(read.cigartuples) == 2:
                sa_info = extract_sa_positions(read, target_chr)
                should_break = False
                for chrom, sa_pos, op_type, op_len in sa_info:
                    if sa_pos + op_len == target_pos:
                        if read.cigartuples[0][0] == 4:
                            homolen = op_len - read.cigartuples[0][1]
                            should_break = True
                            break
                        
                        if read.cigartuples[0][0] == 0:
                            homolen = op_len - read.cigartuples[1][1]
                            should_break = True
                            break
                if should_break:
                    break

    return homolen

def process_single_row(row, bam_path, header_indices, window):
    try:
        sv_type = row[header_indices["sv_type"]]
        
        if sv_type not in {"deletion", "tandem-duplication"}:
            return None
            
        chr_start = row[header_indices["chr_start"]]
        chr_end = row[header_indices["chr_end"]]
        pos_start = int(row[header_indices["pos_start"]])
        pos_end = int(row[header_indices["pos_end"]])
        
        if sv_type == "deletion":
            or_start = 1
            or_end = 1
            flag = 0
        elif sv_type == "tandem-duplication":
            or_start = 1
            or_end = -1
            flag = 0
        else:
            or_start = 1
            or_end = 1
            flag = 0
            
        with pysam.AlignmentFile(bam_path, "rb") as bam:
            target_chr1 = _normalize_chrom_name(chr_start, bam.references)
            target_chr2 = _normalize_chrom_name(chr_end, bam.references)
            
            reads_bp1 = _fetch_split_reads(bam, target_chr1, 
                                          start=max(0, pos_start - window),
                                          end=pos_start + window)
            reads_bp2 = _fetch_split_reads(bam, target_chr2, 
                                          start=max(0, pos_end - window),
                                          end=pos_end + window)
            
            bp1_np = get_breakpoint_from_sa(reads_bp1, or_start, chr_end, flag)
            homolen1 = generate_consensus(reads_bp1, chr_end, bp1_np, 'left')
            
            bp2_np = get_breakpoint_from_sa(reads_bp2, or_end, chr_start, flag)
            homolen2 = generate_consensus(reads_bp2, chr_start, bp2_np, 'right')
            
            homolen = mergeh(homolen1, homolen2)
            mechanism = getmechanism(homolen)
            
            insertlen = abs(homolen) if homolen < 0 else 0
            
            return row + [str(homolen), str(insertlen), mechanism]
            
    except Exception as e:
        print(f"处理行时发生错误: {str(e)}", flush=True)
        return row + ["er", "er", "er"]

def mergeh(h1: int, h2: int) -> int:
    if h1 == 0:
        return h2
    elif h1 < -10 or h2 < -10:
        return min(h1, h2)
    elif h1 > 100 or h2 > 100:
        return max(h1, h2)
    elif h1 > 1 or h2 > 1:
        return max(h1, h2)
    return h1

def getmechanism(h: int) -> str:
    if h > 100:
        return "NAHR"
    elif h < -10:
        return "FoSTeS/MMBIR" 
    elif h > 1:
        return "alt-EJ"
    return "NHEJ"

def process_sv_from_tsv(bam_path: str, input_tsv: str, output_txt: str, 
                       window: int = DEFAULT_WINDOW) -> None:
    try:
        with open(input_tsv, 'r') as infile:
            reader = csv.reader(infile, delimiter='\t')
            header = next(reader, None)
            rows = list(reader)
            
        if not header:
            raise ValueError("输入文件缺少标题行")

        try:
            header_indices = {
                "chr_start": header.index("chrom1"),
                "chr_end": header.index("chrom2"),
                "pos_start": header.index("start1"),
                "pos_end": header.index("start2"),
                "sv_type": header.index("svclass")
            }
        except ValueError as e:
            print(f"错误：输入文件缺少必要的列。请确保包含以下列：chrom1, chrom2, start1, start2, svclass")
            raise e
        
        max_workers = min(mp.cpu_count(), 8)
        
        pool = mp.Pool(processes=max_workers)

        worker = partial(
            process_single_row,
            bam_path=bam_path,
            header_indices=header_indices,
            window=window
        )

        results = []
        chunk_size = max(1, len(rows) // (max_workers * 2))
        for result in pool.imap(worker, rows, chunksize=chunk_size):
            results.append(result)

        pool.close()
        pool.join()

        with open(output_txt, 'w', newline='') as outfile:
            writer = csv.writer(outfile, delimiter='\t')
            output_header = header + ["homolen", "insertlen", "mechanism"]
            writer.writerow(output_header)
            for res in results:
                if res is not None:
                    writer.writerow(res)
                    
        print(f"处理完成！共处理 {len([r for r in results if r is not None])} 条记录")
        print(f"结果已保存至: {output_txt}")

    except Exception as e:
        print(f"文件处理错误: {str(e)}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='SV断点的同源序列')
    parser.add_argument('-b', '--bam', required=True, help='BAM文件路径')
    parser.add_argument('-i', '--input', required=True, help='输入TSV文件')
    parser.add_argument('-o', '--output', help='输出文件路径')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.bam_file):
        print(f"错误：BAM文件不存在：{args.bam_file}")
        sys.exit(1)
    
    if not os.path.exists(args.input):
        print(f"错误：输入TSV文件不存在：{args.input}")
        sys.exit(1)
    
    if args.output:
        output_file = args.output
    else:
        base_name = os.path.basename(args.input)
        prefix = base_name.split('.')[0]
        output_file = f"{prefix}.annotated.tsv"
        print(f"未指定输出文件，自动生成: {output_file}")
    
    if not os.path.exists(args.bam_file + ".bai"):
        print(f"警告：BAM索引文件不存在，正在尝试创建...")
        pysam.index(args.bam_file)
    
    process_sv_from_tsv(args.bam_file, args.input, output_file)
    print(f"\n分析完成！")

if __name__ == "__main__":
    main()