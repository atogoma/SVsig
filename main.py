#!/usr/bin/env python3

import sys
import os
import argparse
import shutil
import torch

def get_script_dir():
    return os.path.dirname(os.path.abspath(__file__))


def run_breakpoint(args):
    script_dir = get_script_dir()
    sys.path.insert(0, script_dir)
    
    import vcftobed
    converter = vcftobed.VCFToBEDPEConverter(args.vcf_dir, args.output)
    converter.run()
    
    import gethomo
    for f in os.listdir(args.output):
        if f.endswith('.SV_annotated.tsv'):
            sample = f.replace('.SV_annotated.tsv', '')
            input_path = os.path.join(args.output, f)
            output_path = os.path.join(args.output, f"{sample}.breakpoint.tsv")
            bam_file = os.path.join(args.bam_dir, f"{sample}.bam")
            
            gethomo.process_sv_from_tsv(
                bam_path=bam_file,
                input_tsv=input_path,
                output_txt=output_path
            )


def run_matrix(args):
    script_dir = get_script_dir()
    sys.path.insert(0, script_dir)
    
    import matrixgenerator
    
    generator = matrixgenerator.SVMatrixGenerator(
        annotated_file=args.breakpoint_tsv,
        output_dir=args.output,
        cnv_file=args.cnv_tsv
    )
    generator.generate_matrix()


def run_extractor(args):
    script_dir = get_script_dir()
    sys.path.insert(0, script_dir)
    
    import mvnmf
    original_argv = sys.argv
    sys.argv = ['mvnmf.py', args.count_matrix, '-o', args.output]
    mvnmf.main()
    sys.argv = original_argv
    
    import matchCOS
    file_basename = os.path.splitext(os.path.basename(args.count_matrix))[0]
    for f in os.listdir(args.output):
        if f.startswith(file_basename) and '_W_matrix_optimal_n' in f:
            w_matrix = os.path.join(args.output, f)
            matchCOS.sv_cosmic_matching(
                denovo_file=w_matrix,
                output_file=os.path.join(args.output, "cosmic_match_results.txt")
            )
            break


def run_runall(args):
    print("SVSig - runall")
    
    temp_dir = os.path.join(args.output, ".temp_svsig")
    os.makedirs(temp_dir, exist_ok=True)
    
    print("\n[Step 1/3] 执行SV断点分析...")
    print(f"VCF目录: {args.vcf_dir}")
    print(f"BAM目录: {args.bam_dir}")
    print(f"输出目录: {temp_dir}")
    
    class BreakpointArgs:
        pass
    
    bp_args = BreakpointArgs()
    bp_args.vcf_dir = args.vcf_dir
    bp_args.bam_dir = args.bam_dir
    bp_args.output = temp_dir
    
    try:
        run_breakpoint(bp_args)
        print("断点分析完成")
    except Exception as e:
        print(f"断点分析失败: {e}")
        sys.exit(1)
    
    breakpoint_files = []
    for f in os.listdir(temp_dir):
        if f.endswith('.breakpoint.tsv'):
            breakpoint_files.append(os.path.join(temp_dir, f))
    
    if not breakpoint_files:
        print("未找到breakpoint.tsv文件")
        sys.exit(1)
    
    merged_breakpoint = os.path.join(temp_dir, "all_samples.breakpoint.tsv")
    if len(breakpoint_files) == 1:
        shutil.copy(breakpoint_files[0], merged_breakpoint)
        print(f"使用单个breakpoint文件: {os.path.basename(breakpoint_files[0])}")
    else:
        print(f"合并 {len(breakpoint_files)} 个breakpoint文件...")
        with open(merged_breakpoint, 'w') as outfile:
            for i, bf in enumerate(breakpoint_files):
                with open(bf, 'r') as infile:
                    lines = infile.readlines()
                    if i == 0:
                        outfile.writelines(lines)
                    else:
                        outfile.writelines(lines[1:])
        print(f"合并完成: all_samples.breakpoint.tsv")
    
    print("\n[Step 2/3] 生成计数矩阵...")
    print(f"  Breakpoint文件: {merged_breakpoint}")
    if args.cnv_tsv:
        print(f"CNV文件: {args.cnv_tsv}")
    else:
        print(f"CNV文件: 未提供")
    print(f"  输出目录: {temp_dir}")
    
    class MatrixArgs:
        pass
    
    m_args = MatrixArgs()
    m_args.breakpoint_tsv = merged_breakpoint
    m_args.cnv_tsv = args.cnv_tsv
    m_args.output = temp_dir
    
    try:
        run_matrix(m_args)
        print("矩阵生成完成")
    except Exception as e:
        print(f"矩阵生成失败: {e}")
        sys.exit(1)
    
    count_matrix = None
    for f in os.listdir(temp_dir):
        if f.endswith('_sv_count_matrix.csv') or f.endswith('_count_matrix.csv'):
            count_matrix = os.path.join(temp_dir, f)
            break
    
    if not count_matrix:
        print("未找到计数矩阵文件")
        sys.exit(1)
    
    print(f"  找到计数矩阵: {os.path.basename(count_matrix)}")
    
    print("\n[Step 3/3] 特征提取...")
    print(f"  计数矩阵: {count_matrix}")
    print(f"  输出目录: {args.output}")
    
    class ExtractorArgs:
        pass
    
    e_args = ExtractorArgs()
    e_args.count_matrix = count_matrix
    e_args.output = args.output
    
    try:
        run_extractor(e_args)
        print("特征提取完成")
    except Exception as e:
        print(f"特征提取失败: {e}")
        sys.exit(1)
    
    if not args.keep_temp:
        print("\n清理临时文件...")
        shutil.rmtree(temp_dir)
        print(f"  已删除临时目录: {temp_dir}")
    else:
        print(f"\n临时文件保留在: {temp_dir}")
    
    print("分析完成！结果文件：")
    
    output_files = [f for f in os.listdir(args.output) if not f.startswith('.')]
    for f in sorted(output_files):
        fpath = os.path.join(args.output, f)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath)
            print(f"  • {f} ({size:,} bytes)")
    
    print("\n最终COSMIC匹配结果: " + os.path.join(args.output, "cosmic_match_results.txt"))


def main():
    parser = argparse.ArgumentParser(description='SVSig - SV特征分析工具包')
    subparsers = parser.add_subparsers(dest='command', help='子命令')
    
    p_break = subparsers.add_parser('breakpoint', help='SV断点分析')
    p_break.add_argument('-i', '--vcf_dir', required=True, help='VCF文件目录')
    p_break.add_argument('-b', '--bam_dir', required=True, help='BAM文件目录')
    p_break.add_argument('-o', '--output', required=True, help='输出目录')
    
    p_matrix = subparsers.add_parser('matrix', help='生成计数矩阵')
    p_matrix.add_argument('-p', '--breakpoint_tsv', required=True, help='输出的注释文件')
    p_matrix.add_argument('-c', '--cnv_tsv', help='CNV文件')
    p_matrix.add_argument('-o', '--output', required=True, help='输出目录')
    
    p_extract = subparsers.add_parser('extractor', help='特征提取')
    p_extract.add_argument('-m', '--count_matrix', required=True, help='计数矩阵文件')
    p_extract.add_argument('-o', '--output', required=True, help='输出目录')
    
    p_runall = subparsers.add_parser('runall', help='一键执行所有分析步骤')
    p_runall.add_argument('-i', '--vcf_dir', required=True, help='VCF文件目录')
    p_runall.add_argument('-bam', '--bam_dir', required=True, help='BAM文件目录')
    p_runall.add_argument('-c', '--cnv_tsv', help='CNV文件（可选）')
    p_runall.add_argument('-o', '--output', required=True, help='最终输出目录')
    p_runall.add_argument('--keep_temp', action='store_true', help='保留临时文件')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    if args.command != 'runall':
        os.makedirs(args.output, exist_ok=True)
    
    if args.command == 'breakpoint':
        run_breakpoint(args)
    elif args.command == 'matrix':
        run_matrix(args)
    elif args.command == 'extractor':
        run_extractor(args)
    elif args.command == 'runall':
        run_runall(args)


if __name__ == "__main__":
    main()