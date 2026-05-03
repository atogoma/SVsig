#!/usr/bin/env python
"""
cosmic匹配
"""
import sys
import os
import argparse
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')


def get_cosmic_file_path():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cosmic_file = os.path.join(script_dir, "cosmic_sv32.txt")
    
    return cosmic_file


def load_local_cosmic(cosmic_file_path):
    cosmic_data = pd.read_csv(cosmic_file_path, sep='\t', index_col=0)
    return cosmic_data.values, cosmic_data.columns.tolist()


def decompose_to_cosmic(denovo_matrix, cosmic_matrix, cosmic_signature_names,
                        nnls_add_penalty=0.05, nnls_remove_penalty=0.01):
    n_denovo = denovo_matrix.shape[1]
    
    contribution_matrix = []
    similarity_scores = []
    reconstruction_errors = []
    significant_contributions_list = []
    
    for i in range(n_denovo):
        denovo_sig = denovo_matrix[:, i:i+1]
        
        coeffs, residual = nnls(cosmic_matrix, denovo_sig.flatten())
        
        reconstructed = cosmic_matrix @ coeffs
        
        similarity = cosine_similarity(
            denovo_sig.T, 
            reconstructed.reshape(1, -1)
        )[0, 0]
        
        significant_signatures = []
        for j, coeff in enumerate(coeffs):
            if coeff > 0:
                if coeff > nnls_add_penalty:
                    status = 'strong'
                elif coeff > nnls_remove_penalty:
                    status = 'weak'
                else:
                    status = 'low'
                
                significant_signatures.append({
                    'index': j,
                    'name': cosmic_signature_names[j],
                    'contribution': coeff,
                    'status': status
                })
        
        significant_signatures.sort(key=lambda x: x['contribution'], reverse=True)
        
        contribution_matrix.append(coeffs)
        similarity_scores.append(similarity)
        reconstruction_errors.append(residual)
        significant_contributions_list.append(significant_signatures)
    
    contribution_matrix = np.array(contribution_matrix).T
    best_cosmic_indices = np.argmax(contribution_matrix, axis=0)
    best_cosmic_names = [cosmic_signature_names[idx] for idx in best_cosmic_indices]
    best_cosmic_contributions = np.max(contribution_matrix, axis=0)
    
    return {
        'contributions': contribution_matrix,
        'similarities': np.array(similarity_scores),
        'errors': np.array(reconstruction_errors),
        'significant_contributions': significant_contributions_list,
        'best_cosmic_indices': best_cosmic_indices,
        'best_cosmic_names': best_cosmic_names,
        'best_cosmic_contributions': best_cosmic_contributions
    }


def sv_cosmic_matching(denovo_file, output_file,
                       nnls_add_penalty=0.05, nnls_remove_penalty=0.01):
    
    print("正在处理SV突变特征与COSMIC数据库比对...")
    
    cosmic_file = get_cosmic_file_path()
    if not os.path.exists(cosmic_file):
        print(f"错误: 找不到COSMIC文件 {cosmic_file}")
        print("请确保cosmic_sv32.txt文件与Python脚本在同一目录下")
        return None
    
    denovo_data = pd.read_csv(denovo_file, sep='\t', header=None)
    denovo_matrix = denovo_data.values
    
    if denovo_matrix.shape[0] >= 32:
        denovo_matrix = denovo_matrix[:32, :]
    else:
        print(f"错误: 输入矩阵只有{denovo_matrix.shape[0]}行，需要至少32行")
        return None
    
    cosmic_matrix, cosmic_signature_names = load_local_cosmic(cosmic_file)
    
    if denovo_matrix.shape[0] != cosmic_matrix.shape[0]:
        print(f"错误: 维度不匹配，denovo矩阵行数={denovo_matrix.shape[0]}, COSMIC矩阵行数={cosmic_matrix.shape[0]}")
        return None
    
    results = decompose_to_cosmic(
        denovo_matrix, cosmic_matrix, cosmic_signature_names,
        nnls_add_penalty, nnls_remove_penalty
    )
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# SV突变特征与COSMIC特征比对结果\n")
        f.write(f"# de novo特征数量: {denovo_matrix.shape[1]}\n")
        f.write(f"# COSMIC特征数量: {cosmic_matrix.shape[1]}\n")
        f.write(f"# 突变类型数量: {denovo_matrix.shape[0]}\n")
        f.write(f"# 强分配阈值: {nnls_add_penalty}\n")
        f.write(f"# 弱分配阈值: {nnls_remove_penalty}\n\n")
        
        for i in range(len(results['similarities'])):
            f.write(f"# ==================================================\n")
            f.write(f"# Denovo_Sig_{i+1}\n")
            f.write(f"# Cosine Similarity: {results['similarities'][i]:.6f}\n")
            f.write(f"# Reconstruction Error: {results['errors'][i]:.6f}\n")
            f.write(f"# Best Match: {results['best_cosmic_names'][i]} (contribution: {results['best_cosmic_contributions'][i]:.6f})\n")
            f.write(f"# ==================================================\n")
            
            non_zero_contributions = results['significant_contributions'][i]
            if non_zero_contributions:
                f.write(f"COSMIC_Signature\tContribution\tStatus\n")
                for sig in non_zero_contributions:
                    f.write(f"{sig['name']}\t{sig['contribution']:.6f}\t{sig['status']}\n")
            else:
                f.write("No significant COSMIC signatures found (all contributions <= 0)\n")
            
            f.write("\n\n")
        
        f.write("# ==================================================\n")
        f.write("# Full Contribution Matrix (rows: COSMIC signatures, cols: Denovo signatures)\n")
        f.write("# ==================================================\n")
        
        contribution_matrix = results['contributions']
        non_zero_rows = np.any(contribution_matrix > 0, axis=1)
        
        if np.any(non_zero_rows):
            filtered_matrix = contribution_matrix[non_zero_rows]
            filtered_names = [cosmic_signature_names[j] for j in range(len(cosmic_signature_names)) if non_zero_rows[j]]
            
            f.write("COSMIC_Signature\t" + "\t".join([f"Denovo_Sig_{k+1}" for k in range(contribution_matrix.shape[1])]) + "\n")
            for idx, name in enumerate(filtered_names):
                contributions_str = "\t".join([f"{val:.6f}" for val in filtered_matrix[idx]])
                f.write(f"{name}\t{contributions_str}\n")
    
    print("处理完成！")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='COSMIC数据库比对')
    parser.add_argument('-i', '--input', required=True, help='de novo特征矩阵文件')
    parser.add_argument('-o', '--output', required=True, help='输出结果文件路径')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        sys.exit(1)
    
    try:
        results = sv_cosmic_matching(
            denovo_file=args.input,
            output_file=args.output,
            nnls_add_penalty=args.add_penalty,
            nnls_remove_penalty=args.remove_penalty
        )
        
        if results is None:
            sys.exit(1)
        
        print(f"\n比对完成！结果已保存至: {args.output}")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()