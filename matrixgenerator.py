#!/usr/bin/env python
"""
矩阵生成
"""
import os
import pandas as pd
import numpy as np


class SVMatrixGenerator:
    
    def __init__(self, annotated_file, output_dir=None, cnv_file=None):
        self.annotated_file = annotated_file
        self.cnv_file = cnv_file
        
        self.input_dir = os.path.dirname(annotated_file)
        self.project_name = os.path.basename(os.path.normpath(self.input_dir))
        
        if output_dir is None:
            self.output_dir = self.input_dir + '/'
        else:
            self.output_dir = output_dir.rstrip('/') + '/'
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def calculate_dispersion(self, sv_df):
        dispersion = {}
        
        for sample in sv_df["sample"].unique():
            sample_sv = sv_df[(sv_df["sample"] == sample) & 
                              (sv_df["chrom1"] == sv_df["chrom2"])]
            
            if len(sample_sv) > 0:
                positions = pd.concat([sample_sv["pos1"], sample_sv["pos2"]])
                
                if len(positions) > 1:
                    mean_pos = positions.mean()
                    std_pos = positions.std()
                    if mean_pos > 0:
                        dispersion[sample] = std_pos / mean_pos
                    else:
                        dispersion[sample] = 0
                else:
                    dispersion[sample] = 0
            else:
                dispersion[sample] = 0
                
        return pd.Series(dispersion, name="dispersion_score")
    
    def calculate_cnv_scores(self, cnv_file, baseline=2, threshold_percent=0.15):
        if not os.path.exists(cnv_file):
            print(f"警告: CNV文件不存在: {cnv_file}")
            return pd.Series(), pd.Series()
        
        cnv_df = pd.read_csv(cnv_file, sep='\t')
        
        col_mapping = {}
        
        for col in ['chr', 'chrom', 'chromosome', 'CHR', 'Chrom']:
            if col in cnv_df.columns:
                col_mapping['chr'] = col
                break
        
        for col in ['start', 'Start', 'START', 'pos', 'position']:
            if col in cnv_df.columns:
                col_mapping['start'] = col
                break
        
        for col in ['end', 'End', 'END']:
            if col in cnv_df.columns:
                col_mapping['end'] = col
                break
        
        for col in ['total_cn', 'total_CN', 'cn', 'CN', 'copy_number', 'copyNumber']:
            if col in cnv_df.columns:
                col_mapping['total_cn'] = col
                break
        
        for col in ['sample', 'Sample', 'SAMPLE', 'sample_id', 'id']:
            if col in cnv_df.columns:
                col_mapping['sample'] = col
                break
        
        if col_mapping.get('chr'):
            cnv_df = cnv_df.rename(columns={col_mapping['chr']: 'chr'})
        if col_mapping.get('start'):
            cnv_df = cnv_df.rename(columns={col_mapping['start']: 'start'})
        if col_mapping.get('end'):
            cnv_df = cnv_df.rename(columns={col_mapping['end']: 'end'})
        if col_mapping.get('total_cn'):
            cnv_df = cnv_df.rename(columns={col_mapping['total_cn']: 'total_cn'})
        if col_mapping.get('sample'):
            cnv_df = cnv_df.rename(columns={col_mapping['sample']: 'sample'})
        
        required_cols = ['chr', 'start', 'end', 'total_cn', 'sample']
        missing_cols = [col for col in required_cols if col not in cnv_df.columns]
        if missing_cols:
            print(f"错误: CNV文件缺少必要列: {missing_cols}")
            print(f"当前列名: {cnv_df.columns.tolist()}")
            return pd.Series(), pd.Series()
        
        cnv_df['chr'] = cnv_df['chr'].astype(str)
        
        autosomes = []
        for i in range(1, 23):
            autosomes.append(str(i))
            autosomes.append(f"chr{i}")
            autosomes.append(f"CHR{i}")
        
        original_count = len(cnv_df)
        
        cnv_df = cnv_df[cnv_df['chr'].isin(autosomes)]
        
        if len(cnv_df) == 0:
            print("请检查染色体名称格式，当前染色体值:", cnv_df['chr'].unique())
            return pd.Series(), pd.Series()
        
        cnv_df['total_cn'] = pd.to_numeric(cnv_df['total_cn'], errors='coerce')
        
        cnv_df = cnv_df.dropna(subset=['total_cn'])
        
        loss_threshold = baseline * (1 - threshold_percent)
        gain_threshold = baseline * (1 + threshold_percent)
        
        cnv_df['loss'] = cnv_df['total_cn'] < loss_threshold
        cnv_df['gain'] = cnv_df['total_cn'] > gain_threshold
        
        samples = cnv_df['sample'].unique()
        
        gain_scores = {}
        loss_scores = {}
        
        for sample in samples:
            sample_cnv = cnv_df[cnv_df['sample'] == sample]
            gain_count = sample_cnv['gain'].sum()
            loss_count = sample_cnv['loss'].sum()
            gain_scores[sample] = gain_count
            loss_scores[sample] = loss_count
        
        return pd.Series(gain_scores, name="GAIN"), pd.Series(loss_scores, name="LOSS")
    
    def generate_matrix(self):
        if not os.path.exists(self.annotated_file):
            raise FileNotFoundError(f"注释文件不存在: {self.annotated_file}")
        
        all_sv_df = pd.read_csv(self.annotated_file, sep='\t')
        
        required_cols = ['sample', 'svclass', 'size_bin', 'is_clustered', 
                        'chrom1', 'chrom2', 'pos1', 'pos2']
        for col in required_cols:
            if col not in all_sv_df.columns:
                if col == 'pos1' and 'start1' in all_sv_df.columns:
                    all_sv_df['pos1'] = all_sv_df['start1']
                elif col == 'pos2' and 'start2' in all_sv_df.columns:
                    all_sv_df['pos2'] = all_sv_df['start2']
                elif col == 'chrom1' and 'chr_start' in all_sv_df.columns:
                    all_sv_df['chrom1'] = all_sv_df['chr_start']
                elif col == 'chrom2' and 'chr_end' in all_sv_df.columns:
                    all_sv_df['chrom2'] = all_sv_df['chr_end']
                else:
                    raise ValueError(f"注释文件缺少必要列: {col} (或对应的别名)")
        
        if 'homolen' not in all_sv_df.columns:
            print("警告: 未找到 homolen 列，将使用默认值 0")
            all_sv_df['homolen'] = 0
        if 'insertlen' not in all_sv_df.columns:
            print("警告: 未找到 insertlen 列，将使用默认值 0")
            all_sv_df['insertlen'] = 0
        if 'mechanism' not in all_sv_df.columns:
            print("警告: 未找到 mechanism 列，将使用默认值 'NHEJ'")
            all_sv_df['mechanism'] = 'NHEJ'
        
        svclass_mapping = {
            'deletion': 'del',
            'tandem-duplication': 'tds', 
            'inversion': 'inv',
            'translocation': 'trans'
        }
        
        size_bins = ['1-10Kb', '10-100Kb', '100Kb-1Mb', '1Mb-10Mb', '>10Mb']
        
        features = []
        
        for svclass in svclass_mapping.keys():
            if svclass == 'translocation':
                features.append(f"non-clustered_{svclass_mapping[svclass]}")
            else:
                for size_bin in size_bins:
                    features.append(f"non-clustered_{svclass_mapping[svclass]}_{size_bin}")
        
        for svclass in svclass_mapping.keys():
            if svclass == 'translocation':
                features.append(f"clustered_{svclass_mapping[svclass]}")
            else:
                for size_bin in size_bins:
                    features.append(f"clustered_{svclass_mapping[svclass]}_{size_bin}")
        
        samples = sorted(all_sv_df['sample'].unique())
        print(f"发现 {len(samples)} 个样本")
        
        matrix = pd.DataFrame(0, index=features, columns=samples, dtype=int)
        matrix.index.name = 'MutationType'
        
        valid_df = all_sv_df.copy()
        valid_df = valid_df[valid_df['homolen'] != 'er']
        valid_df = valid_df[valid_df['insertlen'] != 'er']
        
        valid_df['homolen'] = pd.to_numeric(valid_df['homolen'], errors='coerce').fillna(0)
        valid_df['insertlen'] = pd.to_numeric(valid_df['insertlen'], errors='coerce').fillna(0)
        
        for _, row in all_sv_df.iterrows():
            sample = row['sample']
            svclass = row['svclass']
            size_bin = row['size_bin']
            is_clustered = row.get('is_clustered', False)
            
            if is_clustered:
                prefix = 'clustered'
            else:
                prefix = 'non-clustered'
            
            if svclass == 'translocation':
                feature = f"{prefix}_{svclass_mapping[svclass]}"
            else:
                feature = f"{prefix}_{svclass_mapping[svclass]}_{size_bin}"
            
            matrix.loc[feature, sample] += 1
        
        dispersion_scores = self.calculate_dispersion(all_sv_df)
        
        dispersion_row = []
        for sample in samples:
            score = dispersion_scores.get(sample, 0)
            dispersion_row.append(score)
        
        dispersion_df = pd.DataFrame([dispersion_row], 
                                     index=['dispersion_score'], 
                                     columns=samples)
        dispersion_df.index.name = 'MutationType'
        
        matrix_with_dispersion = pd.concat([matrix, dispersion_df], axis=0)
        
        cnv_gain_scores = pd.Series()
        cnv_loss_scores = pd.Series()
        
        if self.cnv_file:
            print(f"\n正在处理CNV文件: {self.cnv_file}")
            cnv_gain_scores, cnv_loss_scores = self.calculate_cnv_scores(self.cnv_file)
            
            gain_row = []
            loss_row = []
            for sample in samples:
                gain_row.append(cnv_gain_scores.get(sample, 0))
                loss_row.append(cnv_loss_scores.get(sample, 0))
            
            gain_df = pd.DataFrame([gain_row], 
                                   index=['GAIN'], 
                                   columns=samples)
            loss_df = pd.DataFrame([loss_row], 
                                   index=['LOSS'], 
                                   columns=samples)
            gain_df.index.name = 'MutationType'
            loss_df.index.name = 'MutationType'
            
            matrix_with_dispersion = pd.concat([matrix_with_dispersion, gain_df, loss_df], axis=0)
            print(f"\nCNV分数已添加到矩阵")
        
        stats_rows = []
        
        for sample in samples:
            sample_data = valid_df[valid_df['sample'] == sample]
            valid_svs = sample_data[~sample_data['svclass'].isin(['translocation'])]
            if len(valid_svs) > 0:
                avg_homolen = valid_svs['homolen'].mean()
            else:
                avg_homolen = 0
            stats_rows.append(('avg_homolen', sample, avg_homolen))
        
        for sample in samples:
            sample_data = valid_df[valid_df['sample'] == sample]
            insert_data = sample_data[sample_data['insertlen'] > 0]
            if len(insert_data) > 0:
                avg_insertlen = insert_data['insertlen'].mean()
            else:
                avg_insertlen = 0
            stats_rows.append(('avg_insertlen', sample, avg_insertlen))
        
        mechanisms = ['NHEJ', 'alt-EJ', 'NAHR', 'FoSTeS/MMBIR']
        for mechanism in mechanisms:
            for sample in samples:
                sample_data = valid_df[valid_df['sample'] == sample]
                valid_svs = sample_data[~sample_data['svclass'].isin(['translocation'])]
                count = len(valid_svs[valid_svs['mechanism'] == mechanism])
                stats_rows.append((f'{mechanism}_count', sample, count))
        
        stats_df = pd.DataFrame(stats_rows, columns=['Metric', 'Sample', 'Value'])
        
        stats_pivot = stats_df.pivot(index='Metric', columns='Sample', values='Value')
        stats_pivot.index.name = 'MutationType'
        
        final_matrix = pd.concat([matrix_with_dispersion, stats_pivot], axis=0)
        
        matrix_file = os.path.join(self.output_dir, f"{self.project_name}.MATRIX.tsv")
        final_matrix.to_csv(matrix_file, sep='\t', float_format='%.2f')
        print(f"\n矩阵已保存: {matrix_file}")
        
        return matrix, stats_pivot, dispersion_scores, (cnv_gain_scores, cnv_loss_scores)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='从注释TSV生成SV计数矩阵及统计信息')
    parser.add_argument('-i', '--annotated_file', help='注释TSV文件路径')
    parser.add_argument('-c', '--cnv', help='CNV文件路径', default=None)
    parser.add_argument('-o', '--output_dir', help='输出目录', default=None)

    
    args = parser.parse_args()
    
    generator = SVMatrixGenerator(args.annotated_file, args.output_dir, args.cnv)
    matrix, stats, dispersion, cnv_scores = generator.generate_matrix()
    
    
    if args.cnv:
        gain_scores, loss_scores = cnv_scores
        if len(gain_scores) == 0 and len(loss_scores) == 0:
            print("  警告: 未计算出任何CNV分数")
            print("  请检查CNV文件格式和内容")

    
    print(f"\n完成！")


if __name__ == "__main__":
    main()