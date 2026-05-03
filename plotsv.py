#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

SV32_CHANNELS = [
    "non-clustered_del_1-10Kb",
    "non-clustered_del_10-100Kb",
    "non-clustered_del_100Kb-1Mb",
    "non-clustered_del_1Mb-10Mb",
    "non-clustered_del_>10Mb",
    "non-clustered_tds_1-10Kb",
    "non-clustered_tds_10-100Kb",
    "non-clustered_tds_100Kb-1Mb",
    "non-clustered_tds_1Mb-10Mb",
    "non-clustered_tds_>10Mb",
    "non-clustered_inv_1-10Kb",
    "non-clustered_inv_10-100Kb",
    "non-clustered_inv_100Kb-1Mb",
    "non-clustered_inv_1Mb-10Mb",
    "non-clustered_inv_>10Mb",
    "non-clustered_trans",
    "clustered_del_1-10Kb",
    "clustered_del_10-100Kb",
    "clustered_del_100Kb-1Mb",
    "clustered_del_1Mb-10Mb",
    "clustered_del_>10Mb",
    "clustered_tds_1-10Kb",
    "clustered_tds_10-100Kb",
    "clustered_tds_100Kb-1Mb",
    "clustered_tds_1Mb-10Mb",
    "clustered_tds_>10Mb",
    "clustered_inv_1-10Kb",
    "clustered_inv_10-100Kb",
    "clustered_inv_100Kb-1Mb",
    "clustered_inv_1Mb-10Mb",
    "clustered_inv_>10Mb",
    "clustered_trans",
]

EXTRA_FEATURES = [
    "dispersion_score",
    "GAIN",
    "LOSS",
    "FoSTeS/MMBIR_count",
    "NAHR_count",
    "NHEJ_count",
    "alt-EJ_count",
    "avg_homolen",
    "avg_insertlen",
]

ALL_41_CHANNELS = SV32_CHANNELS + EXTRA_FEATURES

COLOR_MAPPING = {
    "del": {
        ">10Mb": "deeppink",
        "1Mb-10Mb": "hotpink",
        "100Kb-1Mb": "lightpink",
        "10-100Kb": "palevioletred",
        "1-10Kb": "lavenderblush",
    },
    "tds": {
        ">10Mb": "saddlebrown",
        "1Mb-10Mb": "sienna",
        "100Kb-1Mb": "sandybrown",
        "10-100Kb": "peru",
        "1-10Kb": "linen",
    },
    "inv": {
        ">10Mb": "rebeccapurple",
        "1Mb-10Mb": "blueviolet",
        "100Kb-1Mb": "plum",
        "10-100Kb": "mediumorchid",
        "1-10Kb": "thistle",
    },
}

EXTRA_COLORS = {
    "dispersion_score": "darkorange",
    "GAIN": "green",
    "LOSS": "red",
    "FoSTeS/MMBIR_count": "purple",
    "NAHR_count": "brown",
    "NHEJ_count": "gray",
    "alt-EJ_count": "olive",
    "avg_homolen": "cyan",
    "avg_insertlen": "magenta",
}

ACTIVITY_COLORS = [
    "tab:pink", "tab:orange", "tab:purple", "tab:olive", "tab:brown",
    "tab:red", "tab:green", "tab:cyan", "deeppink", "orangered",
    "blueviolet", "chocolate", "darkgreen", "dodgerblue", "mediumvioletred",
    "salmon", "magenta", "sandybrown", "forestgreen", "royalblue",
    "orchid", "indigo", "darkseagreen", "blue", "palevioletred",
    "darkslateblue", "olivedrab", "cyan", "hotpink", "rebeccapurple", "lime",
]


def _get_color(channel_name):
    if channel_name in EXTRA_COLORS:
        return EXTRA_COLORS[channel_name]
    
    parts = channel_name.split("_")
    if len(parts) >= 3:
        sv_type = parts[1]
        if sv_type == "trans":
            return "dimgray"
        size = "_".join(parts[2:]) if len(parts) > 2 else ""
        if sv_type in COLOR_MAPPING and size in COLOR_MAPPING[sv_type]:
            return COLOR_MAPPING[sv_type][size]
    return "gray"


def plot_sv_signature(W_matrix, output_dir, file_basename, signature_idx=None):

    os.makedirs(output_dir, exist_ok=True)
    
    if W_matrix.shape[0] < 41:
        print(f"[WARNING]W_matrix 只有 {W_matrix.shape[0]} 行，期望至少 41 行")
        return
    
    W_sv = W_matrix[:41, :]
    n_signatures = W_sv.shape[1]
    
    if signature_idx is not None:
        indices = [signature_idx]
    else:
        indices = range(n_signatures)
    
    for idx in indices:
        if idx >= n_signatures:
            continue
        
        weights = W_sv[:, idx]
        total = np.sum(weights)
        
        if total > 0:
            weights_percent = weights / total * 100
        else:
            weights_percent = weights
        
        bar_colors = [_get_color(name) for name in ALL_41_CHANNELS]
        
        fig, ax = plt.subplots(figsize=(18, 8))
        
        plt.style.use("ggplot")
        plt.rcParams["axes.facecolor"] = "white"
        ax.set_axisbelow(True)
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.spines["bottom"].set_color("black")
        ax.spines["top"].set_color("black")
        ax.spines["right"].set_color("black")
        ax.spines["left"].set_color("black")
        ax.grid(linestyle="-", linewidth=1, color="#EDEDED", axis="y")
        
        x_pos = np.arange(len(ALL_41_CHANNELS))
        ax.bar(x_pos, weights_percent, color=bar_colors, edgecolor="black", width=0.8)

        short_labels = []
        for name in ALL_41_CHANNELS:
            if name in EXTRA_FEATURES:
                short_labels.append(name)
            else:
                short = name.replace("non-clustered_", "NC_").replace("clustered_", "C_")
                short_labels.append(short)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(short_labels, rotation=90, fontsize=8, ha="center")
        ax.set_ylabel("Percentage (%)", fontsize=14, fontweight="bold")

        ax.axvline(x=31.5, color="black", linewidth=1.5, linestyle="--")
        if np.max(weights_percent) > 0:
            y_top = np.max(weights_percent) * 0.95
            ax.text(16, y_top, "SV32 Channels", fontsize=10, ha="center")
            ax.text(36, y_top, "Extra Features", fontsize=10, ha="center")
        
        plt.tight_layout(pad=0.5)
        ax.set_xlim(-0.5, len(ALL_41_CHANNELS) - 0.5)
        
        if n_signatures == 1:
            output_file = os.path.join(output_dir, f"{file_basename}_SV_signature.pdf")
        else:
            output_file = os.path.join(output_dir, f"{file_basename}_Signature{idx+1}.pdf")
        
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
    
    print(f"SV 特征图已保存到 {output_dir}")


def plot_activity(H_matrix, output_dir, file_basename, bin_size=50, log=False, percentage=True):

    os.makedirs(output_dir, exist_ok=True)
    
    if H_matrix.shape[0] < H_matrix.shape[1]:
        H_matrix = H_matrix.T
    
    n_samples, n_signatures = H_matrix.shape
    
    if n_samples == 0 or n_signatures == 0:
        print("[ERROR] H 矩阵为空")
        return
    
    df = pd.DataFrame(H_matrix)
    df['_sum'] = df.sum(axis=1)
    df = df.sort_values(by='_sum', ascending=False)
    df = df.drop(columns='_sum')
    
    df = df.loc[:, (df != 0).any(axis=0)]
    n_signatures_filtered = df.shape[1]
    
    signature_list = df.columns.tolist()
    
    color_list = []
    for i in range(len(signature_list)):
        if i < len(ACTIVITY_COLORS):
            color_list.append(ACTIVITY_COLORS[i])
        else:
            color_list.append('#%06x' % np.random.randint(0, 0xFFFFFF))
    
    n_pages = (n_samples + bin_size - 1) // bin_size
    output_file = os.path.join(output_dir, f"{file_basename}_Activity.pdf")
    pp = PdfPages(output_file)
    
    for page in range(n_pages):
        start_idx = page * bin_size
        end_idx = min(start_idx + bin_size, n_samples)
        
        df_page = df.iloc[start_idx:end_idx, :]
        n_samples_page = len(df_page)
        
        if n_samples_page == 0:
            continue
        
        plot_length = n_samples_page / 50 * 10
        figure_length = max(plot_length + 5, 8)
        left_margin = 1.5 / figure_length
        
        fig, ax = plt.subplots(figsize=(figure_length, 7))
        plt.rc('axes', edgecolor='lightgray')
        
        x_pos = np.arange(n_samples_page)
        bar_width = 0.8
        bottom = np.zeros(n_samples_page)
        bars = []
        
        for i, sig_name in enumerate(signature_list):
            values = df_page[sig_name].values
            
            if percentage:
                row_sums = df_page.sum(axis=1).values
                row_sums = np.where(row_sums == 0, 1, row_sums)
                values = values / row_sums * 100
            
            bar = ax.bar(x_pos, values, bottom=bottom, width=bar_width,
                        color=color_list[i], edgecolor='white', label=f"Sig{sig_name+1}" if isinstance(sig_name, int) else sig_name)
            bars.append(bar)
            bottom = bottom + values
        
        ax.set_xlim([-0.5, n_samples_page - 0.5])
        ax.set_xticks(x_pos)
        
        sample_labels = [f"S{i+start_idx+1}" for i in range(n_samples_page)]
        ax.set_xticklabels(sample_labels, rotation=90, ha='right', fontsize=8)
        
        ylabel = "Percentage (%)" if percentage else "Activity"
        if log:
            ax.set_yscale('log')
            ylabel = f"log10({ylabel})"
            if percentage:
                print("[WARNING] 对数刻度下堆叠条形图的比例会失真")
        
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.yaxis.grid(True, linestyle='-', linewidth=0.5, color='#EDEDED', zorder=0)
        ax.set_axisbelow(True)
        
        ax.legend(bars, [f"Sig{i+1}" for i in range(len(signature_list))], 
                 fontsize=8, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        
        if n_pages > 1:
            ax.set_title(f"{file_basename} - Page {page+1}/{n_pages}", fontsize=10)
        
        plt.tight_layout()
        pp.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    pp.close()
    print(f"Activity 图已保存到: {output_file}")
    
    return output_file

def plot_all(W_matrix, H_matrix, output_dir, file_basename):

    print("开始生成 SV 特征图...")
    plot_sv_signature(W_matrix, output_dir, file_basename)

    print("开始生成 Activity 图...")
    plot_activity(H_matrix, output_dir, file_basename)
    
    print(f"所有图已保存到: {output_dir}")

def calculate_sv_tmb_from_annotated_tsv(annotated_tsv, genome_size=2800):

    df = pd.read_csv(annotated_tsv, sep='\t')
    sample_counts = df.groupby('sample').size()
    
    tmb_data = pd.DataFrame({
        'sample': sample_counts.index,
        'Mut_burden': sample_counts.values / genome_size
    })
    
    return tmb_data


def plot_tmb(tmb_data, output_dir, file_basename, cutoff=0):

    os.makedirs(output_dir, exist_ok=True)
    
    if tmb_data.empty:
        print("[WARNING] TMB 数据为空，跳过绘图")
        return
    
    df_plot = tmb_data[tmb_data['Mut_burden'] > cutoff].copy()
    
    if df_plot.empty:
        print(f"[WARNING] 没有 Mut_burden > {cutoff} 的样本")
        return
 
    df_plot['log10_burden'] = np.log10(df_plot['Mut_burden'])
    
    df_plot = df_plot.sort_values('Mut_burden', ascending=False).reset_index(drop=True)
    
    n_samples = len(df_plot)
    
    if n_samples == 0:
        return
    
    fig_width = max(10, n_samples * 0.5)
    fig_height = 6
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    for i in range(n_samples):
        if i % 2 == 0:
            ax.axvspan(i - 0.4, i + 0.4, alpha=0.1, color='gray', zorder=0)
    
    x_pos = np.arange(n_samples)
    y_values = df_plot['log10_burden'].values
    
    ax.scatter(x_pos, y_values, color='black', s=15, zorder=2, edgecolors='black', linewidth=0.5)
    
    top_labels = [f"{row['sample']}" for _, row in df_plot.iterrows()]

    bottom_labels = [f"1/1" for _ in range(n_samples)]
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bottom_labels, rotation=0, ha='center', fontsize=8)
    
    for i, label in enumerate(top_labels):
        ax.text(i, ax.get_ylim()[0] - 0.15 * (ax.get_ylim()[1] - ax.get_ylim()[0]), 
                label, ha='center', va='top', fontsize=8, rotation=45)
    
    y_min = np.floor(np.min(y_values)) - 0.5
    y_max = np.ceil(np.max(y_values)) + 0.5
    
    y_ticks_log = np.arange(y_min, y_max + 1, 1)
    y_ticks_original = [10 ** y for y in y_ticks_log]
    
    ax.set_ylim(y_min - 0.2, y_max + 0.2)
    ax.set_yticks(y_ticks_log)
    
    y_labels = []
    for y in y_ticks_original:
        if y < 0.01:
            y_labels.append(f"{y:.4f}")
        elif y < 0.1:
            y_labels.append(f"{y:.3f}")
        elif y < 1:
            y_labels.append(f"{y:.2f}")
        elif y < 10:
            y_labels.append(f"{y:.1f}")
        else:
            y_labels.append(f"{y:.0f}")
    ax.set_yticklabels(y_labels)
    
    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=1)
    ax.grid(axis='x', linestyle='-', alpha=0.2, zorder=0)
    
    ax.set_ylabel('SV burden (per Mb)', fontsize=12, fontweight='bold')
    ax.set_xlabel('')
    
    median_val = np.median(y_values)

    bar_width = 0.6 
    for i in range(n_samples):
        x_center = i
        x_start = x_center - bar_width / 2
        x_end = x_center + bar_width / 2
        ax.hlines(y=median_val, xmin=x_start, xmax=x_end, 
                  colors='red', linewidth=2, zorder=3)
    
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='red', linewidth=2, label=f'Median: {10**median_val:.4f}')]
    ax.legend(handles=legend_elements, fontsize=8, loc='upper right')
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, f"{file_basename}_TMB.pdf")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"TMB 图已保存到: {output_file}")
    
    return output_file

def plot_tmb_from_annotated_tsv(annotated_tsv, output_dir, file_basename, genome_size=2800, cutoff=0):

    tmb_data = calculate_sv_tmb_from_annotated_tsv(annotated_tsv, genome_size)
    return plot_tmb(tmb_data, output_dir, file_basename, cutoff)