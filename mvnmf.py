#!/usr/bin/env python
"""
mvnmf
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
import argparse
import os

from initialization import initialize_nmf, nnls, beta_divergence, normalize_WH
import plotsv as sv_plot

EPSILON = np.finfo(np.float32).eps
EPSILON2 = np.finfo(np.float16).eps


def _volume_logdet(W, delta):
    K = W.shape[1]
    volume = np.log10(np.linalg.det(W.T @ W + delta * np.eye(K)))
    return volume


def _loss_mvnmf(X, W, H, Lambda, delta):
    reconstruction_error, matrix_frobenius = beta_divergence(X, W @ H, square_root=False)
    volume = _volume_logdet(W, delta)
    loss = reconstruction_error + Lambda * volume
    return loss, reconstruction_error, matrix_frobenius, volume


def _solve_mvnmf(X, W, H, lambda_tilde=1e-5, delta=1.0, gamma=1.0,
                 max_iter=200, min_iter=100, tol=1e-4,
                 conv_test_freq=10, conv_test_baseline=None, verbose=0):
    if (type(X) != np.ndarray) or (not np.issubdtype(X.dtype, np.floating)):
        X = np.array(X).astype(float)
    if (type(W) != np.ndarray) or (not np.issubdtype(W.dtype, np.floating)):
        W = np.array(W).astype(float)
    if (type(H) != np.ndarray) or (not np.issubdtype(H.dtype, np.floating)):
        H = np.array(H).astype(float)
    n_features, n_samples = X.shape
    n_components = W.shape[1]
    M = n_features
    T = n_samples
    K = n_components
    W, H = normalize_WH(W, H)
    W = W.clip(EPSILON)
    H = H.clip(EPSILON)
    reconstruction_error, matrix_frobenius = beta_divergence(X, W @ H, square_root=False)
    volume = _volume_logdet(W, delta)
    Lambda = lambda_tilde * reconstruction_error / abs(volume)
    loss = reconstruction_error + Lambda * volume
    ones = np.ones((M, T))
    if conv_test_baseline is None:
        conv_test_baseline = loss
    elif type(conv_test_baseline) is str and conv_test_baseline == 'min-iter':
        pass
    else:
        conv_test_baseline = float(conv_test_baseline)
    losses = [loss]
    reconstruction_errors = [reconstruction_error]
    volumes = [volume]
    line_search_steps = []
    gammas = [gamma]
    loss_previous = loss
    loss_previous_conv_test = loss
    converged = False
    for n_iter in range(1, max_iter + 1):
        H = H * ( ( W.T @ (X/(W @ H)) ) / (W.T @ ones) )
        H = H.clip(EPSILON)
        Y = np.linalg.inv(W.T @ W + delta * np.eye(K))
        Y_plus = np.maximum(Y, 0)
        Y_minus = np.maximum(-Y, 0)
        JHT = ones @ H.T
        LWYm = Lambda * (W @ Y_minus)
        LWY = Lambda * (W @ (Y_plus + Y_minus))
        numerator = ( (JHT - 4 * LWYm)**2 + 8 * LWY * ((X/(W @ H)) @ H.T) )**0.5 - JHT + 4 * LWYm
        denominator = 4 * LWY
        Wup = W * (numerator / denominator)
        Wup = Wup.clip(EPSILON)
        if gamma != -1:
            W_new = (1 - gamma) * W + gamma * Wup
            W_new, H_new = normalize_WH(W_new, H)
            W_new = W_new.clip(EPSILON)
            H_new = H_new.clip(EPSILON)
            loss, reconstruction_error, matrix_frobenius, volume = _loss_mvnmf(X, W_new, H_new, Lambda, delta)
            line_search_step = 0
            while (loss > loss_previous) and (gamma > 1e-16):
                gamma = gamma * 0.8
                W_new = (1 - gamma) * W + gamma * Wup
                W_new, H_new = normalize_WH(W_new, H)
                W_new = W_new.clip(EPSILON)
                H_new = H_new.clip(EPSILON)
                loss, reconstruction_error, matrix_frobenius, volume = _loss_mvnmf(X, W_new, H_new, Lambda, delta)
                line_search_step += 1
            W = W_new
            H = H_new
        else:
            line_search_step = 0
            W = Wup
            W, H = normalize_WH(W, H)
            W = W.clip(EPSILON)
            H = H.clip(EPSILON)
        line_search_steps.append(line_search_step)
        if gamma != -1:
            gamma = min(gamma*2.0, 1.0)
        gammas.append(gamma)
        loss, reconstruction_error, matrix_frobenius, volume = _loss_mvnmf(X, W, H, Lambda, delta)
        losses.append(loss)
        reconstruction_errors.append(reconstruction_error)
        volumes.append(volume)
        loss_previous = loss
        if n_iter == min_iter and conv_test_baseline == 'min-iter':
            conv_test_baseline = loss
        if n_iter >= min_iter and tol > 0 and n_iter % conv_test_freq == 0:
            relative_loss_change = (loss_previous_conv_test - loss) / conv_test_baseline
            if (loss <= loss_previous_conv_test) and (relative_loss_change <= tol):
                converged = True
            else:
                converged = False

            loss_previous_conv_test = loss
        if converged and n_iter >= min_iter:
            break

    losses = np.array(losses)
    reconstruction_errors = np.array(reconstruction_errors)
    volumes = np.array(volumes)
    line_search_steps = np.array(line_search_steps)
    gammas = np.array(gammas)

    return W, H, n_iter, converged, Lambda, losses, reconstruction_errors, volumes, line_search_steps, gammas


class MVNMF:
    def __init__(self,
                 X,
                 n_components,
                 init='cluster',
                 lambda_tilde=1e-5,
                 delta=1.0,
                 gamma=1.0,
                 max_iter=200,
                 min_iter=100,
                 tol=1e-4,
                 conv_test_freq=10,
                 conv_test_baseline=None,
                 verbose=0
                 ):
        if (type(X) != np.ndarray) or (not np.issubdtype(X.dtype, np.floating)):
            X = np.array(X).astype(float)
        self.X = X
        self.n_components = n_components
        self.init = init
        self.lambda_tilde = lambda_tilde
        self.delta = delta
        self.gamma = gamma
        self.max_iter = max_iter
        self.min_iter = min_iter
        self.tol = tol
        self.conv_test_freq = conv_test_freq
        self.conv_test_baseline = conv_test_baseline
        self.verbose = verbose

    def fit(self):
        W_init, H_init = initialize_nmf(self.X, self.n_components,
                                         init=self.init)
        self.W_init = W_init
        self.H_init = H_init

        (_W, _H, n_iter, converged, Lambda, losses, reconstruction_errors,
            volumes, line_search_steps, gammas) = _solve_mvnmf(
            X=self.X, W=self.W_init, H=self.H_init, lambda_tilde=self.lambda_tilde,
            delta=self.delta, gamma=self.gamma, max_iter=self.max_iter,
            min_iter=self.min_iter, tol=self.tol,
            conv_test_freq=self.conv_test_freq,
            conv_test_baseline=self.conv_test_baseline,
            verbose=self.verbose)
        self.n_iter = n_iter
        self.converged = converged
        self.Lambda = Lambda
        W = normalize(_W, norm='l1', axis=0)
        H = nnls(self.X, W)
        self._W = _W
        self._H = _H
        self._loss = losses[-1]
        self._reconstruction_error = reconstruction_errors[-1]
        self._volume = volumes[-1]
        self.W = W
        self.H = H
        loss, reconstruction_error, matrix_frobenius, volume = _loss_mvnmf(self.X, self.W, self.H, self.Lambda, self.delta)
        self.loss = loss
        self.reconstruction_error = reconstruction_error
        self.matrix_frobenius = matrix_frobenius
        self.volume = volume
        self.frobenius = matrix_frobenius
        self.loss_track = losses
        self.reconstruction_error_track = reconstruction_errors
        self.volume_track = volumes
        self.line_search_step_track = line_search_steps
        self.gamma_track = gammas

        return self


def is_pareto_efficient(costs):
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            dominated = np.all(costs >= c, axis=1) & np.any(costs > c, axis=1)
            is_efficient[dominated] = False
    return is_efficient


def main():
    parser = argparse.ArgumentParser(description='MVNMF')
    parser.add_argument('input_file')
    parser.add_argument('-o', '--output_dir', help='输出目录', default=None)
    parser.add_argument('-k', '--max_components', help='最大组件数（默认20）', type=int, default=20)
    parser.add_argument('--min_components', help='最小组件数（默认1）', type=int, default=1)
    parser.add_argument('--lambda_tilde', help='正则化参数（默认1e-5）', type=float, default=1e-5)
    parser.add_argument('--delta', help='Delta参数（默认1.0）', type=float, default=1.0)
    parser.add_argument('--gamma', help='Gamma参数（默认1.0）', type=float, default=1.0)
    parser.add_argument('--max_iter', help='最大迭代次数（默认200）', type=int, default=200)
    parser.add_argument('--min_iter', help='最小迭代次数（默认100）', type=int, default=100)
    parser.add_argument('--tol', help='收敛容差（默认1e-4）', type=float, default=1e-4)
    parser.add_argument('--init', help='初始化方法（默认cluster）', type=str, default='cluster')
    parser.add_argument('--verbose', help='详细输出（默认0）', type=int, default=0)
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        output_dir = os.path.dirname(args.input_file)
        if not output_dir:
            output_dir = '.'
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    file_basename = os.path.splitext(os.path.basename(args.input_file))[0]

    
    try:
        df = pd.read_csv(args.input_file, delimiter='\t', index_col=0)
        print(f"\n读取文件成功")
        has_header = True
    except Exception as e:
        try:
            df = pd.read_csv(args.input_file, delimiter='\t', header=None)
            print(f"\n读取文件成功（无表头格式）")
            has_header = False
        except Exception as e:
            print(f"错误: 无法读取文件 {args.input_file}")
            print(f"错误详情: {e}")
            return
    
    X = df.values
    
    if X.shape[0] == 0 or X.shape[1] == 0:
        print("错误: 矩阵为空")
        return
    
    col_sums = X.sum(axis=0)
    
    if np.allclose(col_sums, 1.0, rtol=1e-5, atol=1e-8):
        print("数据已经L1归一化")
    else:
        print("数据未完全归一化，将自动进行L1归一化")
        X = X / X.sum(axis=0, keepdims=True)
        new_col_sums = X.sum(axis=0)
    
    results = {}
    n_samples = X.shape[1]
    max_components = min(args.max_components, n_samples - 1)
    
    if max_components < args.min_components:
        print(f"错误: 最大组件数({max_components})小于最小组件数({args.min_components})")
        print(f"建议: 样本数为{n_samples}，最大组件数应不超过{n_samples-1}")
        return
    
    print(f"\n开始MVNMF...")
    
    for comn in range(args.min_components, max_components + 1):
        print(f"\n正在处理 n_components={comn}/{max_components}...")
        try:
            mvnmf = MVNMF(
                X=X,
                n_components=comn,  
                init=args.init,
                lambda_tilde=args.lambda_tilde,
                delta=args.delta,
                gamma=args.gamma,
                max_iter=args.max_iter,
                min_iter=args.min_iter,
                tol=args.tol,
                conv_test_freq=10,
                conv_test_baseline=None,
                verbose=args.verbose
            )
            
            mvnmf.fit()
            print(f"  完成: n_iter={mvnmf.n_iter}, converged={mvnmf.converged}, loss={mvnmf.loss:.6f}")
            
            results[comn] = {
                'n_iter': mvnmf.n_iter,
                'loss': mvnmf.loss,
                'reconstruction_error': mvnmf.reconstruction_error,
                'frobenius': mvnmf.matrix_frobenius,
                'volume': mvnmf.volume,
                'converged': mvnmf.converged,
                'W': mvnmf.W,
                'H': mvnmf.H
            }
        except Exception as e:
            print(f"  错误: n_components={comn} 处理失败: {e}")
            continue
    
    if results:
        
        n_components_list = sorted(results.keys())
        
        recon_errors = []
        frobenius_values = []
        
        for n_comp in n_components_list:
            result = results[n_comp]
            recon_errors.append(result['reconstruction_error'])
            frobenius_values.append(result['frobenius'])
        
        costs = np.column_stack([recon_errors, frobenius_values])
        
        pareto_mask = is_pareto_efficient(costs)
        pareto_indices = np.where(pareto_mask)[0]
        pareto_solutions = [n_components_list[i] for i in pareto_indices]
        
        for i, n_comp in enumerate(pareto_solutions):
            result = results[n_comp]
        
        if pareto_solutions:
            best_n_components = min(pareto_solutions, key=lambda x: 
                                   np.sqrt(results[x]['reconstruction_error'] * results[x]['frobenius']))
            
            best_result = results[best_n_components]
            
            W_optimal = best_result['W']
            H_optimal = best_result['H']
            
            W_output = os.path.join(output_dir, f"{file_basename}_W_matrix_optimal_n{best_n_components}.tsv")
            W_df = pd.DataFrame(W_optimal)
            W_df.to_csv(W_output, sep='\t', header=False, index=False, float_format='%.6f')

            if W_optimal.shape[0] >= 41 and H_optimal.shape[0] > 0:
                sv_plot.plot_all(
                    W_matrix=W_optimal,
                    H_matrix=H_optimal,
                    output_dir=output_dir,
                    file_basename=f"{file_basename}_n{best_n_components}"
                )
            else:
                print(f"[WARNING] 无法绘图: W行数={W_optimal.shape[0]}, H行数={H_optimal.shape[0]}")
            
            H_output = os.path.join(output_dir, f"{file_basename}_H_matrix_optimal_n{best_n_components}.tsv")
            H_df = pd.DataFrame(H_optimal)
            H_df.to_csv(H_output, sep='\t', header=False, index=False, float_format='%.6f')
            
            summary_output = os.path.join(output_dir, f"{file_basename}_results_summary.tsv")
            summary_data = []
            for n_comp in n_components_list:
                r = results[n_comp]
                summary_data.append([n_comp, r['reconstruction_error'], r['frobenius'], 
                                    r['volume'], r['loss'], r['converged'], r['n_iter']])
            summary_df = pd.DataFrame(summary_data, columns=['n_components', 'reconstruction_error', 
                                                             'frobenius', 'volume', 'loss', 'converged', 'n_iter'])
            summary_df.to_csv(summary_output, sep='\t', index=False, float_format='%.6f')
            
            pareto_output = os.path.join(output_dir, f"{file_basename}_pareto_solutions.tsv")
            pareto_data = []
            for n_comp in pareto_solutions:
                r = results[n_comp]
                pareto_data.append([n_comp, r['reconstruction_error'], r['frobenius'], 
                                   r['volume'], r['loss'], r['converged'], r['n_iter']])
            pareto_df = pd.DataFrame(pareto_data, columns=['n_components', 'reconstruction_error', 
                                                           'frobenius', 'volume', 'loss', 'converged', 'n_iter'])
            pareto_df.to_csv(pareto_output, sep='\t', index=False, float_format='%.6f')
            
            print(f"\n最优 n_components: {best_n_components}")

            print("完成!")
        else:
            print("没有找到 Pareto 最优解")
    else:
        print("没有有效的计算结果")


if __name__ == "__main__":
    main()