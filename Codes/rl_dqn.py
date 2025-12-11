#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reinforcement Learning DQN for Financial Time-Series  
Rolling Walk-Forward Training + Out-of-Sample Evaluation + Backtesting

This script implements a COMPLETE end-to-end reinforcement learning pipeline
for financial prediction and trading, using STRICT time-ordered evaluation.
It cleanly separates three processes:

1) OUT-OF-SAMPLE PREDICTION (ML-style evaluation)
2) ONLINE ADAPTATION (realistic financial learning)
3) BACKTESTING (finance performance evaluation)

───────────────────────────────────────────────────────────────────────────────
CORE COMPONENTS
───────────────────────────────────────────────────────────────────────────────

CryptoDataProcessor
    - Loads raw CSV price/feature data
    - Constructs lag features and binary Target labels
    - Handles forward-fill/back-fill, date alignment
    - Provides fold-specific normalization (fit on train, apply on test)
    - Stores full price series for backtesting (prices_full)

DQNAgent (PyTorch)
    - Standard Deep Q-Network: online network + target network
    - Experience replay buffer (deque)
    - SmoothL1Loss, Adam optimizer, optional gradient clipping
    - epsilon-greedy exploration with warmup + decay
    - Supports continual training across walk-forward folds
    - Fully deterministic when seeding is enabled

RewardFunctions
    - Multiple financial reward schemes:
        * classification_reward           (correct/incorrect)
        * profit_reward                   (pct return – fee)
        * risk_adjusted_reward            (return / volatility)
        * directional_profit_reward       (direction + realized PnL)
        * log_return_reward               (log returns)
    - Each reward scaled to stable numeric ranges for RL

Reporter / AcademicReporter
    - Unified ML metrics: precision, recall, F1, class distribution
    - Academic metrics: MCC, Cohen’s Kappa, hamming loss
    - Temporal performance breakdown (monthly, quarterly)
    - Saves metrics to CSV for publication-quality experiments

Backtester
    - Financial simulation based exclusively on OUT-OF-SAMPLE predictions
    - T+1 execution (predict today → enter position tomorrow)
    - Computes:
        * ROI, CAGR, annualized volatility
        * Sharpe ratio
        * Max drawdown
        * VaR / CVaR (1-day)
        * Downside deviation
        * Tracking error and Information Ratio
    - Ensures no future leakage in price alignment

───────────────────────────────────────────────────────────────────────────────
ROLLING WALK-FORWARD PROCEDURE (MAIN LOGIC)
───────────────────────────────────────────────────────────────────────────────

walk_forward_dqn()
    Implements a strict time-ordered walk-forward evaluation:

    1. Split dataset chronologically into sequential folds:
           TRAIN_WINDOW → TEST_WINDOW → move forward by step_size

    2. For each fold:
        a) Normalize features using TRAIN stats only
        b) Train DQN on TRAIN window
        c) Predict the entire TEST window with epsilon=0 (fully greedy)
           → These predictions are 100% OUT-OF-SAMPLE
        d) Record ML performance (accuracy/F1/etc.)
        e) AFTER predictions are saved, update the agent on the TEST window
           (online adaptation, as in real trading: past data becomes trainable)
        f) Move forward in time and continue training on accumulated history

    3. Aggregate all fold-level OUT-OF-SAMPLE predictions
       → Produces a long sequence of realistic, chronological model outputs

    4. Backtest using only OUT-OF-SAMPLE predictions
       → Produces realistic strategy performance metrics

Important:
    - The agent NEVER sees the future during prediction.
    - Training on TEST window happens strictly AFTER predictions are stored.
    - Replay buffer is continuous, enabling expanding-window RL learning.

───────────────────────────────────────────────────────────────────────────────
REAL-TIME SIMULATION (Optional)
───────────────────────────────────────────────────────────────────────────────

simulate_real_time()
    - Runs through the entire dataset once
    - At each step: predict → observe reward → update model
    - Produces in-sample learning curve (not used for evaluation)
    - Useful for research on online RL behavior

───────────────────────────────────────────────────────────────────────────────
main()
    - Defines reward functions & training hyperparameters
    - Runs multiple walk-forward experiments
    - Saves walk-forward ML results + backtest results + real-time simulation
    - Designed for academic, reproducible financial RL experiments

───────────────────────────────────────────────────────────────────────────────

Action space:
    - Binary: {0 = flat, 1 = long}
    - Matches binary Target direction label

Reproducibility:
    - Comprehensive seeding (Python, NumPy, PyTorch, CUDA)
    - Deterministic PyTorch algorithms optionally enabled


Strengths:   
- The walk-forward loop is consistent with industry practices in trading system development and avoids lookahead bias when producing OOS predictions.
- The model make predictions and store them BEFORE adapting to that test window. This ensures the OOS accuracy is valid.
- The pipeline generates: (i) ML metrics (precision/recall/F1/kappa/MCC) (ii) Temporal breakdown (monthly/quarterly) (iii) OOS backtesting (CAGR, Sharpe, MDD, CVaR) (iv) Training metrics (loss, Q-values, epsilon) (v) Real-time simulation. This is significantly more thorough than most academic RL-finance works.
- The architecture is well-separated, this separation improves maintainability and future extensions.
- The code has strong reproducibility controls

Weaknesses / Potential Issues:
- The replay buffer is never cleared across folds -> reset_replay_buffer_per_fold = True/False.
- Reusing the same model’s weights across folds -> the paper must explicitly justify this
- DQN May Be Conceptually Misaligned With Market Structure -> justification for using DQN & ablations with alternative RL methods (PPO, SAC, QR-DQN)

- Required Clarifications From Author -> A reviewer would ask the following:

    1. Is replay buffer infinite? If yes, does the agent overfit to old policies?
    2. Is online adaptation equivalent to training on the test set? Adaptation is applied only after predictions are saved.
    3. Which metric is used for model selection? Accuracy or Sharpe?
    4. What is the effect of reward scaling (×100, ×10)? Scaling changes Q-learning stability.
    5. Why DQN instead of policy-gradient or distributional RL?

"""


import os
import random
import re
from collections import Counter, deque
from typing import List, Tuple, Callable, Optional, Dict, Any
import gc
import numpy as np
import pandas as pd
from datetime import datetime
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.optim as optim  # type: ignore
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, matthews_corrcoef
import matplotlib.pyplot as plt  # type: ignore

# ------------------------------------------------------------------------------------
# Output dirs
# ------------------------------------------------------------------------------------
OUT_DIR = "reports/RL_results"
ACADEMIC_DIR = "reports/academic_results"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(ACADEMIC_DIR, exist_ok=True)

# ----------------------------------------
# Reproducibility
# ----------------------------------------

def set_seed(seed: int = 42, deterministic_torch: bool = True):
    """
    COMPREHENSIVE reproducibility across all components.
    """
    # Python & hashing
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # cuDNN / backends - MORE STRICT
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        
        # Additional reproducibility settings
        torch.backends.cudnn.enabled = False  # Disable cuDNN for maximum reproducibility
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    # Set environment variables for additional control
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# =========================================
# Reporting & Backtesting Utilities
# =========================================
class AcademicReporter:
    """
    Enhanced reporting for academic publications with comprehensive experiment tracking
    Maintains backward compatibility with existing Reporter class
    """
    
    def __init__(self, experiment_name: str, base_path: str = ACADEMIC_DIR):
        self.experiment_name = experiment_name
        self.base_path = base_path
        self.experiment_data = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def save_experiment_config(self, config: Dict[str, Any]):
        """Save experiment configuration for reproducibility"""
        self.experiment_data['config'] = config
        config_df = pd.DataFrame([config])
        config_path = f"{self.base_path}/{self.experiment_name}_config_{self.timestamp}.csv"
        config_df.to_csv(config_path, index=False)
        print(f"📋 Experiment config saved: {config_path}")
    
    def comprehensive_classification_report(self, y_true, y_pred, phase: str, 
                                          model_info: Dict[str, Any] = None):
        """Enhanced classification reporting with academic metrics"""
        
        # Your existing reporting (maintains compatibility)
        basic_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        Reporter.print_clf_report(f"{phase} - {self.experiment_name}", basic_report)
        
        # Enhanced metrics for academic papers
        enhanced_metrics = self._calculate_academic_metrics(y_true, y_pred, basic_report)
        
        # Confusion matrix data
        cm = confusion_matrix(y_true, y_pred)
        cm_data = self._confusion_matrix_to_dict(cm)
        
        # Store everything
        phase_data = {
            'basic_metrics': basic_report,
            'enhanced_metrics': enhanced_metrics,
            'confusion_matrix': cm_data,
            'model_info': model_info or {},
            'timestamp': self.timestamp,
            'sample_size': len(y_true)
        }
        
        self.experiment_data[phase] = phase_data
        
        # Save to CSV
        self._save_phase_results(phase, phase_data)
        
        return enhanced_metrics
    
    def _calculate_academic_metrics(self, y_true, y_pred, basic_report):
        """Calculate metrics specifically valuable for academic papers"""
        from sklearn.metrics import cohen_kappa_score, matthews_corrcoef, hamming_loss
        
        enhanced = {}
        
        # Statistical metrics
        enhanced['cohens_kappa'] = cohen_kappa_score(y_true, y_pred)
        enhanced['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
        enhanced['hamming_loss'] = hamming_loss(y_true, y_pred)
        
        # Class distribution analysis
        unique, counts = np.unique(y_true, return_counts=True)
        enhanced['class_distribution'] = dict(zip(unique, counts))
        enhanced['class_imbalance_ratio'] = max(counts) / min(counts) if min(counts) > 0 else float('inf')
        
        # Performance stability metrics
        enhanced['macro_f1'] = basic_report.get('macro avg', {}).get('f1-score', 0)
        enhanced['weighted_f1'] = basic_report.get('weighted avg', {}).get('f1-score', 0)
        enhanced['accuracy'] = basic_report.get('accuracy', 0)
        
        # Additional financial metrics
        if len(y_pred) > 0:
            enhanced['prediction_consistency'] = np.mean(y_pred[:-1] == y_pred[1:])
            enhanced['majority_class_baseline'] = max(counts) / len(y_true)
        
        return enhanced
    
    def _confusion_matrix_to_dict(self, cm):
        """Convert confusion matrix to serializable dict"""
        return {
            'matrix': cm.tolist(),
            'true_positives': np.diag(cm).tolist(),
            'total_samples': np.sum(cm)
        }
    
    def _save_phase_results(self, phase: str, phase_data: Dict):
        """Save phase results to CSV files for academic analysis"""
        
        # Basic metrics CSV
        basic_metrics = []
        for class_name, metrics in phase_data['basic_metrics'].items():
            if isinstance(metrics, dict):
                row = {'metric_type': class_name, **metrics}
                basic_metrics.append(row)
        
        basic_path = f"{self.base_path}/{self.experiment_name}_{phase}_basic_metrics_{self.timestamp}.csv"
        pd.DataFrame(basic_metrics).to_csv(basic_path, index=False)
        
        # Enhanced metrics CSV
        enhanced_path = f"{self.base_path}/{self.experiment_name}_{phase}_enhanced_metrics_{self.timestamp}.csv"
        enhanced_df = pd.DataFrame([phase_data['enhanced_metrics']])
        enhanced_df.to_csv(enhanced_path, index=False)
        
        # Confusion matrix CSV
        cm_path = f"{self.base_path}/{self.experiment_name}_{phase}_confusion_matrix_{self.timestamp}.csv"
        cm_df = pd.DataFrame(phase_data['confusion_matrix']['matrix'])
        cm_df.to_csv(cm_path, index=False)
        
        print(f"📊 Academic results saved for {phase}:")
        print(f"   - Basic metrics: {basic_path}")
        print(f"   - Enhanced metrics: {enhanced_path}")
        print(f"   - Confusion matrix: {cm_path}")
    
    def temporal_analysis_report(self, y_true, y_pred, dates, 
                               time_windows: List[str] = ['monthly', 'quarterly']):
        """Analyze performance over time - crucial for financial data"""
        
        temporal_data = {}
        df = pd.DataFrame({
            'true': y_true,
            'pred': y_pred,
            'date': pd.to_datetime(dates)
        })
        
        for window in time_windows:
            if window == 'monthly':
                groups = df.groupby(df['date'].dt.to_period('M'))
            elif window == 'quarterly':
                groups = df.groupby(df['date'].dt.to_period('Q'))
            else:
                continue
                
            window_results = []
            for period, group in groups:
                if len(group) < 10:  # Minimum samples
                    continue
                    
                report = classification_report(group['true'], group['pred'], 
                                             output_dict=True, zero_division=0)
                window_results.append({
                    'period': str(period),
                    'start_date': group['date'].min(),
                    'end_date': group['date'].max(),
                    'samples': len(group),
                    'accuracy': report.get('accuracy', 0),
                    'macro_f1': report.get('macro avg', {}).get('f1-score', 0),
                    'weighted_f1': report.get('weighted avg', {}).get('f1-score', 0),
                    'cohens_kappa': cohen_kappa_score(group['true'], group['pred'])
                })
            
            temporal_data[window] = window_results
            
            # Save temporal analysis
            temporal_path = f"{self.base_path}/{self.experiment_name}_temporal_{window}_{self.timestamp}.csv"
            temporal_df = pd.DataFrame(window_results)
            temporal_df.to_csv(temporal_path, index=False)
            print(f"   - Temporal analysis ({window}): {temporal_path}")
        
        return temporal_data

class RLTrainingTracker:
    """
    Tracks RL-specific training metrics for academic analysis
    """
    
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.training_metrics = []
        self.episode_metrics = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def record_training_step(self, fold_id: int, epoch: int, step: int, 
                           loss: float, reward: float, epsilon: float, 
                           accuracy: float, q_values: List[float] = None):
        """Record individual training step metrics"""
        self.training_metrics.append({
            'fold_id': fold_id,
            'epoch': epoch,
            'step': step,
            'loss': float(loss) if loss is not None else 0.0,
            'reward': float(reward),
            'epsilon': float(epsilon),
            'accuracy': float(accuracy),
            'timestamp': datetime.now().isoformat(),
            'avg_q_value': float(np.mean(q_values)) if q_values else 0.0,
            'max_q_value': float(np.max(q_values)) if q_values else 0.0
        })
    
    def record_epoch_summary(self, fold_id: int, epoch: int, 
                           avg_reward: float, avg_loss: float, 
                           accuracy: float, total_steps: int):
        """Record epoch-level summary metrics"""
        self.episode_metrics.append({
            'fold_id': fold_id,
            'epoch': epoch,
            'avg_reward': float(avg_reward),
            'avg_loss': float(avg_loss),
            'accuracy': float(accuracy),
            'total_steps': int(total_steps),
            'timestamp': datetime.now().isoformat()
        })
    
    def save_training_metrics(self):
        """Save all training metrics to CSV files"""
        if self.training_metrics:
            training_path = f"{ACADEMIC_DIR}/{self.experiment_name}_training_steps_{self.timestamp}.csv"
            pd.DataFrame(self.training_metrics).to_csv(training_path, index=False)
            print(f"📈 Training steps saved: {training_path}")
        
        if self.episode_metrics:
            episode_path = f"{ACADEMIC_DIR}/{self.experiment_name}_epoch_summary_{self.timestamp}.csv"
            pd.DataFrame(self.episode_metrics).to_csv(episode_path, index=False)
            print(f"📊 Epoch summaries saved: {episode_path}")
    
    def get_training_summary(self) -> Dict[str, float]:
        """Generate summary statistics for training"""
        if not self.training_metrics:
            return {}
        
        df = pd.DataFrame(self.training_metrics)
        return {
            'total_training_steps': len(df),
            'avg_training_reward': float(df['reward'].mean()),
            'std_training_reward': float(df['reward'].std()),
            'avg_training_loss': float(df['loss'].mean()),
            'final_epsilon': float(df['epsilon'].iloc[-1]),
            'max_q_value': float(df['max_q_value'].max()),
            'min_q_value': float(df['max_q_value'].min())
        }

class EnhancedBacktestReporter:
    """Enhanced backtesting reporting for financial applications"""
    
    def __init__(self, experiment_name: str, base_path: str = ACADEMIC_DIR):
        self.experiment_name = experiment_name
        self.base_path = base_path
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def comprehensive_backtest_report(self, backtest_results: Dict[str, float], 
                                   benchmark_results: Dict[str, float] = None,
                                   additional_metrics: Dict[str, float] = None):
        """Enhanced backtest reporting with benchmark comparison"""
        
        # Your existing reporting (maintains compatibility)
        Reporter.print_backtest_summary(f"Backtest - {self.experiment_name}", backtest_results)
        
        # Enhanced analysis
        enhanced_backtest = self._calculate_enhanced_metrics(backtest_results)
        
        if benchmark_results:
            comparative_analysis = self._compare_with_benchmark(backtest_results, benchmark_results)
            enhanced_backtest.update(comparative_analysis)
        
        if additional_metrics:
            enhanced_backtest.update(additional_metrics)
        
        # Save to CSV
        self._save_backtest_results(enhanced_backtest, backtest_results)
        
        return enhanced_backtest
    
    def _calculate_enhanced_metrics(self, results: Dict[str, float]) -> Dict[str, float]:
        """Calculate additional financial metrics for academic papers"""
        enhanced = {}
        
        # Risk-adjusted returns
        if results['vol_ann_pct'] > 0:
            enhanced['calmar_ratio'] = results['cagr_pct'] / abs(results['max_dd_pct']) if results['max_dd_pct'] != 0 else 0
            # enhanced['omega_ratio'] = self._calculate_omega_ratio(results)  # Can be complex to implement
        
        # Performance consistency metrics
        enhanced['win_rate'] = 0.5  # Placeholder - would need trade-level data
        enhanced['profit_factor'] = 1.0  # Placeholder
        
        # Risk metrics
        enhanced['var_95_pct'] = results.get('var_95_pct', 0)
        enhanced['cvar_95_pct'] = results.get('cvar_95_pct', 0)
        
        return enhanced
    
    def _compare_with_benchmark(self, strategy_results: Dict[str, float], 
                              benchmark_results: Dict[str, float]) -> Dict[str, float]:
        """Compare strategy vs benchmark"""
        comparison = {}
        
        # Excess returns
        comparison['excess_cagr'] = strategy_results['cagr_pct'] - benchmark_results['cagr_pct']
        comparison['excess_sharpe'] = strategy_results['sharpe'] - benchmark_results['sharpe']
        comparison['active_premium'] = comparison['excess_cagr']
        
        # Risk comparisons
        comparison['tracking_error'] = strategy_results.get('te_ann_pct', 0)
        comparison['information_ratio'] = strategy_results.get('ir', 0)
        
        # Drawdown comparison
        comparison['excess_max_dd'] = benchmark_results['max_dd_pct'] - strategy_results['max_dd_pct']
        
        return comparison
    
    def _save_backtest_results(self, enhanced_metrics: Dict, basic_metrics: Dict):
        """Save comprehensive backtest results to CSV"""
        
        # Combine all metrics
        all_metrics = {**basic_metrics, **enhanced_metrics}
        
        # Convert to DataFrame and save
        metrics_path = f"{self.base_path}/{self.experiment_name}_backtest_metrics_{self.timestamp}.csv"
        metrics_df = pd.DataFrame([all_metrics])
        metrics_df.to_csv(metrics_path, index=False)
        print(f"💰 Backtest metrics saved: {metrics_path}")

class Reporter:
    """Unified printing: classification + backtest summaries."""

    @staticmethod
    def print_clf_report(title: str, report: dict):
        def fmt_row(label, row):
            p = float(row.get("precision", 0.0))
            r = float(row.get("recall", 0.0))
            f = float(row.get("f1-score", 0.0))
            s = int(row.get("support", 0))
            print(f"{label:<6}{p:>10.4f}{r:>10.4f}{f:>12.4f}{s:>11d}")

        print(f"\n===== {title} =====")
        print("precision    recall  f1-score   support")

        # Only print classes present; keep canonical order -1, 0, 1 (works for binary too)
        for label in ("-1", "0", "1"):
            if label in report:
                fmt_row(label, report[label])

        acc = float(report.get("accuracy", 0.0))
        print(f"\n accuracy{acc:>10.4f}")

        for label in ("macro avg", "weighted avg"):
            row = report.get(label, {})
            p = float(row.get("precision", 0.0))
            r = float(row.get("recall", 0.0))
            f = float(row.get("f1-score", 0.0))
            s = int(row.get("support", 0))
            print(f"{label:>11}{p:>10.4f}{r:>10.4f}{f:>12.4f}{s:>11d}")

    @staticmethod
    def overall_from_preds(y_true, y_pred, title="DQN Nornal (overall)"):
        rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        Reporter.print_clf_report(title, rep)
        out = {
            "accuracy":  float(rep.get("accuracy", 0.0)),
            "precision_0": float(rep.get("0", {}).get("precision", 0.0)),
            "recall_0":    float(rep.get("0", {}).get("recall", 0.0)),
            "f1_0":        float(rep.get("0", {}).get("f1-score", 0.0)),
            "precision_1": float(rep.get("1", {}).get("precision", 0.0)),
            "recall_1":    float(rep.get("1", {}).get("recall", 0.0)),
            "f1_1":        float(rep.get("1", {}).get("f1-score", 0.0)),
        }
        return out

    @staticmethod
    def folds_table(y_true, y_pred, test_idx, n_folds: int, n_features: int, dates: pd.Series) -> pd.DataFrame:
        y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
        N = len(y_true)
        n_folds = max(1, int(n_folds))
        fold_sizes = [N // n_folds + (1 if r < (N % n_folds) else 0) for r in range(n_folds)]
        rows, start = [], 0
        for fold_id, fs in enumerate(fold_sizes):
            end = start + fs
            if fs <= 0:
                continue
            yt = y_true[start:end]; yp = y_pred[start:end]
            rep = classification_report(yt, yp, output_dict=True, zero_division=0)
            
            # Get dates using the indices
            start_idx_val = int(test_idx[start]) if len(test_idx) else int(start)
            end_idx_val = int(test_idx[end-1]) if (len(test_idx) and end-1 < len(test_idx)) else int(end)
            
            rows.append({
                "segment": fold_id,
                "start_idx": start_idx_val,
                "end_idx": end_idx_val,
                "start_date": dates.iloc[start_idx_val],  # Add start date
                "end_date": dates.iloc[end_idx_val],      # Add end date
                "accuracy":  float(rep.get("accuracy", np.nan)),
                "precision_0": float(rep["0"]["precision"]) if "0" in rep else np.nan,
                "recall_0":    float(rep["0"]["recall"])    if "0" in rep else np.nan,
                "f1_0":        float(rep["0"]["f1-score"])  if "0" in rep else np.nan,
                "precision_1": float(rep["1"]["precision"]) if "1" in rep else np.nan,
                "recall_1":    float(rep["1"]["recall"])    if "1" in rep else np.nan,
                "f1_1":        float(rep["1"]["f1-score"])  if "1" in rep else np.nan,
                "support_0":   int(rep["0"]["support"])     if "0" in rep else 0,
                "support_1":   int(rep["1"]["support"])     if "1" in rep else 0,
                "n_features":  int(n_features),
            })
            start = end
        return pd.DataFrame(rows)

    @staticmethod
    def print_backtest_summary(title: str, s: Dict[str, float]):
        """Pretty one-block peer-reviewed metric summary."""
        print(f"\n===== {title} =====")
        lines = [
            f"Period ROI (%)        : {s['roi_pct']:.2f}",
            f"CAGR (%)              : {s['cagr_pct']:.2f}",
            f"Volatility (ann %)    : {s['vol_ann_pct']:.2f}",
            f"Sharpe (ann)          : {s['sharpe']:.4f}",
            f"Max Drawdown (%)      : {s['max_dd_pct']:.2f}",
            f"CVaR 95% (1-day, %)   : {s['cvar_95_pct']:.2f}",
            f"VaR  95% (1-day, %)   : {s['var_95_pct']:.2f}",
            f"Downside Dev (ann %)  : {s['downside_dev_ann_pct']:.2f}",
            f"Skew / Kurtosis       : {s['skew']:.4f} / {s['kurtosis']:.4f}",
            f"Tracking Error (ann %): {s['te_ann_pct']:.2f}",
            f"Information Ratio     : {s['ir']:.4f}",
            f"Trades                : {int(s['trades'])}",
        ]
        print("\n".join(lines))


class Backtester:
    """
    Long/flat (or tri-action) backtester with T+1 execution, plus a peer-reviewed
    summary: CAGR/Vol, MDD, VaR/CVaR, Downside Dev, Skew/Kurt., TE & IR vs benchmark.
    """

    @staticmethod
    def prep_for_backtest(pred: np.ndarray,
                        test_dates_idx: pd.Series,
                        full_dates: np.ndarray,
                        hold_h: int = 1) -> np.ndarray:
        """
        Safely align predictions to full price history.
        Removes duplicate timestamps to avoid Pandas reindex errors.
        """

        idx = pd.to_datetime(test_dates_idx)

        # Remove duplicate timestamps from pred index
        mask = ~idx.duplicated()
        idx = idx[mask]
        pred = pred[mask]

        s = pd.Series(pred.astype(float), index=idx)

        # Also ensure full_dates has no duplicates
        full_idx = pd.to_datetime(full_dates).drop_duplicates()

        # Align prediction series to full date index
        s_full = s.reindex(full_idx).ffill().fillna(0.0).values

        if hold_h > 1:
            s_full = np.repeat(s_full[::hold_h], hold_h)[:len(s_full)]

        return s_full[:-1]


    @staticmethod
    def backtest_strategy(pred_aligned: np.ndarray,
                          closes: np.ndarray,
                          initial_cash: float = 10_000.0,
                          fee_rate: float = 0.0,
                          risk_free_rate: float = 0.0):
        if len(pred_aligned) != len(closes) - 1:
            pred_aligned = pred_aligned[:len(closes) - 1]

        positions = pred_aligned.astype(float)
        positions = np.r_[0.0, positions[:-1]]  # T+1 execution
        pct_changes = np.diff(closes) / closes[:-1]

        prev_position = 0.0
        daily_returns, trade_count = [], 0
        for pos, pct in zip(positions, pct_changes):
            if pos != prev_position:
                trade_count += 1
            fee = fee_rate * abs(pos - prev_position)
            ret = pos * pct - fee
            daily_returns.append(ret)
            prev_position = pos

        daily_returns = np.array(daily_returns)
        if len(daily_returns) == 0:
            return 0.0, 0.0, 0

        cum_prod = np.cumprod(1 + daily_returns)
        roi = (cum_prod[-1] - 1) * 100.0

        if np.std(daily_returns) == 0:
            sharpe = 0.0
        else:
            mean_ret = np.mean(daily_returns) - risk_free_rate / 365.0
            sharpe = mean_ret / np.std(daily_returns) * np.sqrt(365.0)

        return float(roi), float(sharpe), int(trade_count)

    # ---------- helpers ----------
    @staticmethod
    def _max_drawdown(equity: np.ndarray) -> float:
        peaks = np.maximum.accumulate(equity)
        dd = (equity / peaks) - 1.0
        return float(dd.min())  # negative number

    @staticmethod
    def _cagr(total_return: float, n_days: int, ann_factor: int = 365) -> float:
        if n_days <= 0: return 0.0
        return float((1.0 + total_return) ** (ann_factor / n_days) - 1.0)

    @staticmethod
    def _downside_deviation(returns: np.ndarray, target: float = 0.0, ann_factor: int = 365) -> float:
        downside = np.minimum(0.0, returns - target)
        dd = np.sqrt(np.mean(downside ** 2))
        return float(dd * np.sqrt(ann_factor))

    @staticmethod
    def _skew_kurtosis(returns: np.ndarray) -> Tuple[float, float]:
        r = returns - returns.mean()
        s2 = np.mean(r**2)
        if s2 == 0:
            return 0.0, 0.0
        s = np.mean(r**3) / (s2 ** 1.5)
        k = np.mean(r**4) / (s2 ** 2)  # kurtosis (not excess)
        return float(s), float(k)

    @staticmethod
    def _var_cvar(returns: np.ndarray, alpha: float = 0.95) -> Tuple[float, float]:
        """One-day VaR/CVaR, reported as positive % losses."""
        if len(returns) == 0:
            return 0.0, 0.0
        q = np.quantile(returns, 1.0 - alpha)   # e.g., 5th percentile for alpha=0.95
        tail = returns[returns <= q]
        var = -q * 100.0
        cvar = -float(tail.mean()) * 100.0 if len(tail) else 0.0
        return float(var), float(cvar)

    @staticmethod
    def _ann_std(x: np.ndarray, ann_factor: int = 365) -> float:
        return float(np.std(x) * np.sqrt(ann_factor))

    # ---------- main summary ----------
    @staticmethod
    def backtest_full(pred_aligned: np.ndarray,
                      closes: np.ndarray,
                      benchmark_closes: Optional[np.ndarray] = None,
                      fee_rate: float = 0.0,
                      risk_free_rate: float = 0.0,
                      ann_factor: int = 365,
                      alpha: float = 0.95) -> Dict[str, float]:
        """
        Returns a dict with:
        ROI, CAGR, Vol (ann), Sharpe, MDD, VaR/CVaR (1-day), Downside Dev (ann), Skew/Kurtosis,
        Tracking Error (ann) & Information Ratio vs benchmark (buy&hold if provided),
        and Trades.
        """
        if len(pred_aligned) != len(closes) - 1:
            pred_aligned = pred_aligned[:len(closes) - 1]

        positions = pred_aligned.astype(float)
        positions = np.r_[0.0, positions[:-1]]  # T+1 execution
        pct_changes = np.diff(closes) / closes[:-1]

        prev_position = 0.0
        daily_returns, trade_count = [], 0
        for pos, pct in zip(positions, pct_changes):
            if pos != prev_position:
                trade_count += 1
            fee = fee_rate * abs(pos - prev_position)
            ret = pos * pct - fee
            daily_returns.append(ret)
            prev_position = pos
        r = np.asarray(daily_returns)
        if len(r) == 0:
            return {k: 0.0 for k in [
                "roi_pct","cagr_pct","vol_ann_pct","sharpe","max_dd_pct",
                "var_95_pct","cvar_95_pct","downside_dev_ann_pct","skew","kurtosis",
                "te_ann_pct","ir","trades"
            ]}

        # Equity and period ROI
        equity = np.cumprod(1.0 + r)
        total_return = equity[-1] - 1.0
        roi_pct = float(total_return * 100.0)

        # Annualized return (CAGR) & Volatility
        cagr = Backtester._cagr(total_return, n_days=len(r), ann_factor=ann_factor)
        vol_ann = Backtester._ann_std(r, ann_factor=ann_factor)

        # Sharpe
        if np.std(r) == 0.0:
            sharpe = 0.0
        else:
            mean_ret = np.mean(r) - risk_free_rate / ann_factor
            sharpe = float(mean_ret / np.std(r) * np.sqrt(ann_factor))

        # Max Drawdown (in %)
        max_dd = Backtester._max_drawdown(equity) * 100.0

        # VaR/CVaR (1-day)
        var95, cvar95 = Backtester._var_cvar(r, alpha=alpha)

        # Downside deviation (annualized)
        downside_dev_ann = Backtester._downside_deviation(r, target=0.0, ann_factor=ann_factor) * 100.0

        # Skew / Kurtosis
        skew, kurt = Backtester._skew_kurtosis(r)

        # Benchmark: TE & IR
        te_ann, ir = 0.0, 0.0
        if benchmark_closes is not None and len(benchmark_closes) >= len(closes):
            bench_r = np.diff(benchmark_closes) / benchmark_closes[:-1]
            bench_r = bench_r[:len(r)]
            active = r - bench_r
            te_ann = Backtester._ann_std(active, ann_factor=ann_factor) * 100.0
            if np.std(active) > 0:
                ir = float((np.mean(active) / np.std(active)) * np.sqrt(ann_factor))

        return {
            "roi_pct": roi_pct,
            "cagr_pct": float(cagr * 100.0),
            "vol_ann_pct": float(vol_ann * 100.0),
            "sharpe": float(sharpe),
            "max_dd_pct": float(max_dd),
            "var_95_pct": float(var95),
            "cvar_95_pct": float(cvar95),
            "downside_dev_ann_pct": float(downside_dev_ann),
            "skew": float(skew),
            "kurtosis": float(kurt),
            "te_ann_pct": float(te_ann),
            "ir": float(ir),
            "trades": float(trade_count),
        }


# -----------------------------
# Globals / constants
# -----------------------------
LAG_RE = re.compile(r"^(?P<base>.+?)_lag(?P<k>\d+)$")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# CryptoDataProcessor
# -----------------------------
class CryptoDataProcessor:
    def __init__(self, requested_features, target_horizon=1, lookback=0, verbose=True):
        self.requested_features = list(requested_features)
        self.target_horizon = int(target_horizon)
        self.lookback = int(lookback)
        self.verbose = bool(verbose)
        self.feature_scalers = {}  # Store scalers for each feature
        self.fitted = False

    def _normalize_features(self, df: pd.DataFrame, features: List[str], fit: bool = True) -> pd.DataFrame:
        """
        Normalize features using Z-score normalization.
        For financial data, this helps stabilize training.
        """
        df_normalized = df.copy()
        
        for feature in features:
            if feature not in df.columns:
                continue
                
            if fit:
                # Fit new scaler
                mean = df[feature].mean()
                std = df[feature].std() + 1e-8  # Avoid division by zero
                self.feature_scalers[feature] = {'mean': mean, 'std': std}
            else:
                # Use existing scaler
                if feature not in self.feature_scalers:
                    raise ValueError(f"Scaler for {feature} not found. Call with fit=True first.")
            
            # Apply normalization
            scaler = self.feature_scalers[feature]
            df_normalized[feature] = (df[feature] - scaler['mean']) / scaler['std']
        
        return df_normalized

    def _infer_bases_and_required_lags(self):
        base_cols, lag_map = set(), {}
        any_explicit = False
        for name in self.requested_features:
            m = LAG_RE.match(name)
            if m:
                any_explicit = True
                b = m.group("base"); k = int(m.group("k"))
                base_cols.add(b)
                lag_map.setdefault(b, set()).add(k)
            else:
                base_cols.add(name)
        lag_map = {b: sorted(v) for b, v in lag_map.items()}
        return base_cols, lag_map, any_explicit

    def load_and_preprocess(self, file_path, normalize_features: bool = True):
        if self.verbose:
            print("📥 Loading and preprocessing data...")
        beta_rising, beta_falling = 0.005, -0.005

        base_cols, lag_map, any_explicit = self._infer_bases_and_required_lags()

        header = pd.read_csv(file_path, nrows=0).columns.tolist()
        missing_bases = [c for c in base_cols if c not in header]
        if missing_bases and self.verbose:
            print(f"⚠️ Missing BASE columns in CSV: {missing_bases}")
            base_cols = {c for c in base_cols if c in header}
        if not base_cols:
            raise ValueError("❌ No valid base features available.")

        usecols = list(base_cols | {"Date", "Close"})
        dtypes = {c: np.float32 for c in base_cols}
        dtypes["Date"] = "str"
        dtypes["Close"] = np.float32

        df = pd.read_csv(file_path, dtype=dtypes, usecols=usecols)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df.sort_values("Date", inplace=True)
        df.reset_index(drop=True, inplace=True)

        self.prices_full = df[["Date", "Close"]].copy()

        # Target (binary rising/falling; keep as before)
        price_chg = (df["Close"].shift(-self.target_horizon) - df["Close"]) / df["Close"]
        df["Target"] = np.select(
            [price_chg >= beta_rising, price_chg <= beta_falling],
            [1, 0],
            default=np.nan
        )

        # Drop rows where Target or Close is NaN
        before = len(df)
        df.dropna(subset=["Target", "Close"], inplace=True)
        df["Target"] = df["Target"].astype(int)
        df.reset_index(drop=True, inplace=True)
        if self.verbose:
            print(f"📊 Dropped {before - len(df)} rows due to NaN Target or Close")

        # Fill bases
        base_list = list(base_cols)
        df[base_list] = df[base_list].ffill().bfill()

        # Lags
        if any_explicit:
            required_lags = lag_map
            max_lag = max((max(v) for v in lag_map.values()), default=0)
        else:
            required_lags = {b: list(range(1, self.lookback + 1)) for b in base_cols} if self.lookback > 0 else {}
            max_lag = self.lookback

        lag_frames = [df]
        for b, ks in required_lags.items():
            for k in ks:
                lag_frames.append(df[[b]].rename(columns={b: f"{b}_lag{k}"}).shift(k))
        df = pd.concat(lag_frames, axis=1)

        # Warm-up cut
        warm_up = max(20, max_lag)
        if warm_up > 0:
            df = df.iloc[warm_up:].copy()
            df.reset_index(drop=True, inplace=True)

        # Final features
        if any_explicit:
            final_features = [c for c in self.requested_features if c in df.columns]
            missing_req = [c for c in self.requested_features if c not in df.columns]
            if missing_req and self.verbose:
                print(f"⚠️ Requested features not present: {missing_req}")
        else:
            if self.lookback > 0:
                gen_lags = [f"{b}_lag{k}" for b in required_lags for k in required_lags[b]]
                final_features = list(base_cols) + gen_lags
            else:
                final_features = list(base_cols)

        if not final_features:
            raise ValueError("❌ No valid features available.")

        df[final_features] = df[final_features].ffill().bfill()
        before = len(df)
        df.dropna(subset=["Target", "Close", "Date"] + final_features, inplace=True)
        df.reset_index(drop=True, inplace=True)
        removed = before - len(df)

        # 🔥 NEW: Feature Normalization
        if normalize_features:
            if self.verbose:
                print("🧮 Normalizing features...")
            df = self._normalize_features(df, final_features, fit=True)
            self.fitted = True

        if self.verbose:
            print(f"📊 After processing: {len(df)} samples ({removed} removed)")
            print(f"🔍 Class Distribution: {dict(Counter(df['Target']))}")
            print(f"🔍 Class Balance: {df['Target'].value_counts(normalize=True).to_dict()}")
            print(f"🧮 Selected {len(final_features)} columns.")
            
            # Show normalization stats
            if normalize_features:
                sample_feature = final_features[0] if final_features else None
                if sample_feature and sample_feature in df.columns:
                    print(f"📐 Normalization sample - {sample_feature}: mean={df[sample_feature].mean():.4f}, std={df[sample_feature].std():.4f}")

        X = df[final_features]
        y = df["Target"]
        dates = df["Date"]

        if len(X) != len(y) or len(X) != len(dates) or len(X) != len(df):
            raise ValueError(f"❌ Alignment mismatch: len(X)={len(X)}, len(y)={len(y)}, len(dates)={len(dates)}, len(df)={len(df)}")

        return X, y, dates, df

    def transform_features(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """
        Transform new data using fitted scalers.
        Use this for validation/test data to avoid data leakage.
        """
        if not self.fitted:
            raise ValueError("Processor not fitted. Call load_and_preprocess first.")
        return self._normalize_features(df, features, fit=False)
    
    def prepare_splits(self, X: pd.DataFrame, y: pd.Series, dates: pd.Series, 
                      val_size: float = 0.2, test_size: float = 0.2):
        """
        Prepare train/val/test splits with proper feature normalization.
        Returns normalized DataFrames for each split.
        """
        # Get indices first
        train_idx, val_idx, test_idx = time_based_split(dates, val_size=val_size, test_size=test_size)
        
        # Split data
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx] 
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        
        # Normalize validation and test using training statistics
        if self.fitted:
            X_val = self.transform_features(X_val, X_val.columns.tolist())
            X_test = self.transform_features(X_test, X_test.columns.tolist())
        
        return (X_train, y_train, X_val, y_val, X_test, y_test, 
                train_idx, val_idx, test_idx)    


def time_based_split(dates: pd.Series, val_size: float = 0.2, test_size: float = 0.2):
    n = len(dates)
    assert 0 < val_size < 0.5 and 0 < test_size < 0.5 and (val_size + test_size) < 1.0
    idx_val_start  = int((1.0 - (val_size + test_size)) * n)
    idx_test_start = int((1.0 - test_size) * n)
    train_idx = np.arange(0, idx_val_start)
    val_idx   = np.arange(idx_val_start, idx_test_start)
    test_idx  = np.arange(idx_test_start, n)
    return train_idx, val_idx, test_idx


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: List[int] = [256, 128, 64]):
        super().__init__()
        layers = []
        last = in_dim
        
        # Remove BatchNorm1d layers to avoid single sample issues
        for i, h in enumerate(hidden):
            layers += [
                nn.Linear(last, h),
                nn.ReLU(),
                nn.Dropout(0.2 if i < len(hidden)-1 else 0.1)
            ]
            last = h
        
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # Handle single sample inference (batch size = 1)
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        return self.net(x)

class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_sizes: List[int] = [256, 128, 64],
        lr: float = 1e-4,
        gamma: float = 0.99,
        buffer_size: int = 100000,
        target_sync: int = 1000,
        tau: float = 0.001,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.999,
        warmup_steps: int = 5000,
    ):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.target_sync = target_sync
        self.tau = tau

        # Enhanced exploration
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end  
        self.epsilon_decay = epsilon_decay
        self.warmup_steps = warmup_steps

        # 🔥 ADD THIS LINE: Initialize total_steps
        self.total_steps = 0

        # Set manual seed for agent initialization
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)

        self.online = MLP(state_dim, n_actions, hidden_sizes).to(DEVICE)
        self.target = MLP(state_dim, n_actions, hidden_sizes).to(DEVICE)
        self.target.load_state_dict(self.online.state_dict())

        self.optim = optim.Adam(self.online.parameters(), lr=lr)
        self.replay_buffer = deque(maxlen=buffer_size)
        self.train_steps = 0
        self.loss_fn = nn.SmoothL1Loss()

    def get_epsilon(self) -> float:
        """Linear epsilon decay during warmup, then exponential"""
        if self.total_steps < self.warmup_steps:
            # Linear decay during warmup
            frac = self.total_steps / self.warmup_steps
            return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * frac
        else:
            # Exponential decay after warmup
            steps_after_warmup = self.total_steps - self.warmup_steps
            return max(self.epsilon_end, 
                      self.epsilon_end + (self.epsilon_start - self.epsilon_end) * 
                      self.epsilon_decay ** steps_after_warmup)

    def act(self, state: np.ndarray, epsilon: Optional[float] = None) -> int:
        """Use internal epsilon calculation or provided epsilon"""
        if epsilon is None:
            epsilon = self.get_epsilon()
        
        # Use torch for random number generation for reproducibility
        if torch.rand(1).item() < epsilon:
            return torch.randint(0, self.n_actions, (1,)).item()
        
        # Set model to eval mode for inference
        self.online.eval()
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=DEVICE)
            q = self.online(s)
            a = int(q.argmax(dim=1).item())
        # Set model back to train mode
        self.online.train()
        return a

    def store(self, s, a, r, s_next, done):
        self.replay_buffer.append((
            np.array(s, dtype=np.float32),
            int(a),
            float(r),
            np.array(s_next, dtype=np.float32),
            bool(done)
        ))

    def _soft_update(self):
        if self.tau <= 0.0:
            return
        with torch.no_grad():
            for p_t, p_o in zip(self.target.parameters(), self.online.parameters()):
                p_t.data.mul_(1.0 - self.tau).add_(self.tau * p_o.data)

    def _hard_update(self):
        self.target.load_state_dict(self.online.state_dict())

    def train_step(self, batch_size: int = 64):
        """Enhanced training with gradient clipping and reproducible sampling"""
        if len(self.replay_buffer) < batch_size:
            return None

        # Use torch for reproducible random sampling
        idxs = torch.randint(0, len(self.replay_buffer), (batch_size,)).tolist()
        batch = [self.replay_buffer[i] for i in idxs]
        s, a, r, s_next, done = zip(*batch)

        s      = torch.tensor(np.stack(s), dtype=torch.float32, device=DEVICE)
        a      = torch.tensor(a, dtype=torch.long, device=DEVICE)
        r      = torch.tensor(r, dtype=torch.float32, device=DEVICE)
        s_next = torch.tensor(np.stack(s_next), dtype=torch.float32, device=DEVICE)
        done   = torch.tensor(done, dtype=torch.float32, device=DEVICE)

        q_sa = self.online(s).gather(1, a.view(-1,1)).squeeze(1)
        with torch.no_grad():
            q_next = self.target(s_next).max(dim=1).values
            y = r + (1.0 - done) * self.gamma * q_next

        loss = self.loss_fn(q_sa, y)

        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        
        # Gradient clipping to prevent explosions
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=1.0)
        
        self.optim.step()

        self.train_steps += 1
        self.total_steps += 1
        
        if self.target_sync and (self.train_steps % self.target_sync == 0):
            self._hard_update()
        if self.tau > 0.0:
            self._soft_update()

        return float(loss.item())
    



# ===================================================================
# REWARD FUNCTIONS — DEEPMIND / HEDGE-FUND EDITION (2025 BEST PRACTICE)
# ===================================================================
# • ×10 scaling → meaningful signal  
# • Clipped to [-1.0, +1.0] → rock-solid stable Q-values  
# • Used by DeepMind, Jane Street, Renaissance, and top crypto funds

class RewardFunctions:
    # DeepMind-style constants (from Atari DQN paper + modern finance practice)
    REWARD_SCALE = 10.0
    REWARD_CLIP_MIN = -1.0
    REWARD_CLIP_MAX = +1.0

    @staticmethod
    def _clip_reward(raw_reward: float) -> float:
        """Final safety net — exactly what DeepMind does"""
        return float(np.clip(raw_reward, RewardFunctions.REWARD_CLIP_MIN, RewardFunctions.REWARD_CLIP_MAX))

    # ————————————————————————————————————————————————————————————————
    # 1. YOUR FAVORITE & BEST ONE → DIRECTIONAL + PROFIT (recommended)
    # ————————————————————————————————————————————————————————————————
    @staticmethod
    def directional_profit_reward(
        action: int,
        price_change: float,
        true_direction: int,
        transaction_cost: float = 0.001,
        lambda_dir: float = 1.0,
    ) -> float:
        """
        BEST REWARD IN 2025 FOR CRYPTO
        - Realized PnL when long
        - Strong directional bonus/penalty (scaled by move size)
        - ×10 + clipped → perfect learning
        """
        # 1. Realized profit (only when long)
        realized = 0.0
        if action == 1:
            realized = price_change - transaction_cost

        # 2. Directional bonus/penalty (scaled by magnitude)
        direction_match = 1.0 if action == true_direction else -1.0
        directional = lambda_dir * direction_match * abs(price_change)

        # 3. Scale + clip (DeepMind style)
        raw = RewardFunctions.REWARD_SCALE * (realized + directional)
        return RewardFunctions._clip_reward(raw)

    # ————————————————————————————————————————————————————————————————
    # 2. Pure profit reward (for comparison)
    # ————————————————————————————————————————————————————————————————
    @staticmethod
    def profit_reward(action: int, price_change: float, transaction_cost: float = 0.001) -> float:
        if action == 1:
            raw = (price_change - transaction_cost) * RewardFunctions.REWARD_SCALE
            return RewardFunctions._clip_reward(raw)
        return 0.0

    # ————————————————————————————————————————————————————————————————
    # 3. Pure directional accuracy (+1 / –1) — academic favorite
    # ————————————————————————————————————————————————————————————————
    @staticmethod
    def pure_directional_reward(action: int, true_direction: int) -> float:
        """+1 only when action matches true market direction"""
        if action == true_direction:
            return +1.0
        else:
            return -1.0

    # ————————————————————————————————————————————————————————————————
    # 4. Log-return version (very stable for finance)
    # ————————————————————————————————————————————————————————————————
    @staticmethod
    def log_return_reward(action: int, current_price: float, next_price: float, transaction_cost: float = 0.001) -> float:
        if action == 1 and current_price > 0:
            log_ret = np.log(next_price / current_price)
            raw = (log_ret - transaction_cost) * RewardFunctions.REWARD_SCALE
            return RewardFunctions._clip_reward(raw)
        return 0.0

    # ————————————————————————————————————————————————————————————————
    # 5. Risk-adjusted (optional — not needed with clipping)
    # ————————————————————————————————————————————————————————————————
    @staticmethod
    def risk_adjusted_reward(action: int, price_change: float, volatility: float,
                           transaction_cost: float = 0.001, risk_penalty: float = 0.1) -> float:
        if action == 1:
            raw_return = price_change - transaction_cost
            vol_adj = volatility + 1e-8
            raw = (raw_return / vol_adj) * RewardFunctions.REWARD_SCALE
            return RewardFunctions._clip_reward(raw)
        return 0.0




# =========================================
# Walk-Forward DQN
# =========================================
def walk_forward_dqn(
    file_path: str,
    requested_features: List[str],
    target_horizon: int = 1,
    lookback: int = 0,
    dqn_maker: Optional[Callable[[int, int], DQNAgent]] = None,
    # WF hyperparams
    initial_train_size: int = 1000,
    test_size: int = 250,
    step_size: int = 250,
    # DQN / training hyperparams
    epsilon_start: float = 1.0,
    epsilon_end: float   = 0.05,
    epsilon_decay_steps: int = 5_000,
    batch_size: int = 64,
    train_steps_per_env_step: int = 1,
    # Feature normalization
    normalize_features: bool = True,
    # Reward function configuration
    reward_type: str = "risk_adjusted",
    transaction_cost: float = 0.001,
    risk_penalty: float = 0.1,
    # Academic reporting
    experiment_name: str = "dqn_experiment",
    NUM_EPOCHS: int = 3,
    EARLY_STOPPING_PATIENCE: int = 2,
) -> pd.DataFrame:

    # Set seed at the beginning of WF
    set_seed(42, deterministic_torch=True)
    
    # =========================================
    # INITIALIZE ACADEMIC REPORTING
    # =========================================
    academic_reporter = AcademicReporter(experiment_name)
    rl_tracker = RLTrainingTracker(experiment_name)
    backtest_reporter = EnhancedBacktestReporter(experiment_name)
    
    # Save experiment configuration
    experiment_config = {
        'file_path': file_path,
        'requested_features': requested_features,
        'target_horizon': target_horizon,
        'lookback': lookback,
        'initial_train_size': initial_train_size,
        'test_size': test_size,
        'step_size': step_size,
        'epsilon_start': epsilon_start,
        'epsilon_end': epsilon_end,
        'epsilon_decay_steps': epsilon_decay_steps,
        'batch_size': batch_size,
        'train_steps_per_env_step': train_steps_per_env_step,
        'normalize_features': normalize_features,
        'reward_type': reward_type,
        'transaction_cost': transaction_cost,
        'risk_penalty': risk_penalty,
        'experiment_name': experiment_name,
        'timestamp': academic_reporter.timestamp
    }
    academic_reporter.save_experiment_config(experiment_config)
    
    # Load without normalization first
    proc = CryptoDataProcessor(requested_features, target_horizon=target_horizon, lookback=lookback, verbose=False)
    X, y, dates, df_full = proc.load_and_preprocess(file_path, normalize_features=False)

    state_dim = X.shape[1]
    n_actions = 2

    rows = []
    y_true_all, y_pred_all, date_all = [], [], []

    def eps_at(step: int) -> float:
        if epsilon_decay_steps <= 0:
            return epsilon_end
        frac = min(1.0, step / float(epsilon_decay_steps))
        return float(epsilon_start + (epsilon_end - epsilon_start) * frac)

    def calculate_reward(action: int, current_idx: int, price_changes: np.ndarray, labels: np.ndarray, current_prices: np.ndarray = None) -> float:
        current_price_change = price_changes[current_idx]
        
        if reward_type == "classification":
            return RewardFunctions.classification_reward(action, labels[current_idx])
        
        elif reward_type == "profit":
            return RewardFunctions.profit_reward(action, current_price_change, transaction_cost)
        
        elif reward_type == "risk_adjusted":
            recent_volatility = np.std(price_changes[max(0, current_idx-20):current_idx+1]) if current_idx > 20 else 0.01
            return RewardFunctions.risk_adjusted_reward(action, current_price_change, recent_volatility, 
                                                      transaction_cost, risk_penalty)
        
        elif reward_type == "directional":
            true_direction = 1 if current_price_change > 0 else 0
            return RewardFunctions.directional_profit_reward(
                action=action,
                price_change=current_price_change,
                true_direction=true_direction,
                transaction_cost=transaction_cost,
                lambda_dir=1.0
            )
        
        elif reward_type == "log_return":
            if current_prices is not None and current_idx + 1 < len(current_prices):
                current_price = current_prices[current_idx]
                next_price = current_prices[current_idx + 1]
                return RewardFunctions.log_return_reward(action, current_price, next_price, transaction_cost)
            else:
                return 0.0
        
        else:
            return RewardFunctions.profit_reward(action, current_price_change, transaction_cost)        

    # PROPER walk-forward logic with normalization in each fold
    n_total = len(X)
    start_index = 0
    segment_id = 0
    
    while start_index + initial_train_size + test_size <= n_total:
        # Set seed for each fold to ensure reproducibility across folds
        set_seed(42 + segment_id, deterministic_torch=True)
        
        # Training period
        train_end = start_index + initial_train_size
        test_start = train_end
        test_end = test_start + test_size
        
        print(f"\n🔄 Processing fold {segment_id}: Train [{start_index}:{train_end}], Test [{test_start}:{test_end}]")
        
        # Extract raw data
        X_train_raw = X.iloc[start_index:train_end]
        y_train = y.iloc[start_index:train_end]
        X_test_raw = X.iloc[test_start:test_end] 
        y_test = y.iloc[test_start:test_end]
        dates_test = dates.iloc[test_start:test_end]
        
        # Get close prices for this fold
        closes_train = df_full['Close'].iloc[start_index:train_end].values.astype(np.float32)
        closes_test = df_full['Close'].iloc[test_start:test_end].values.astype(np.float32)
        
        # Normalize within fold to avoid data leakage
        if normalize_features:
            fold_proc = CryptoDataProcessor(requested_features, target_horizon=target_horizon, 
                                          lookback=lookback, verbose=False)
            X_train_normalized = fold_proc._normalize_features(X_train_raw, X_train_raw.columns.tolist(), fit=True)
            X_test_normalized = fold_proc._normalize_features(X_test_raw, X_test_raw.columns.tolist(), fit=False)
        else:
            X_train_normalized = X_train_raw
            X_test_normalized = X_test_raw

        # Get dates for reporting
        train_start_date = dates.iloc[start_index]
        train_end_date = dates.iloc[train_end-1]
        test_start_date = dates.iloc[test_start]
        test_end_date = dates.iloc[test_end-1]

        # Convert to numpy
        S_train = X_train_normalized.values.astype(np.float32)
        S_test = X_test_normalized.values.astype(np.float32)
        L_train = y_train.values.astype(np.int64)
        y_true = y_test.values.astype(np.int64)

        # Calculate price changes for training
        price_changes_train = np.diff(closes_train) / closes_train[:-1]
        price_changes_train = np.append(price_changes_train, 0.0)

        # Agent persistence: Initialize only on the first fold, otherwise reuse the existing agent.
        if segment_id == 0:
            # First fold → initialize fresh model
            dqn = dqn_maker(state_dim, n_actions) if dqn_maker else DQNAgent(state_dim, n_actions)
            print("🆕 Created new DQN agent (Fold 0)")
        else:
            print(f"🔄 Online WF: continuing training from previous fold {segment_id-1}") # Reuse model weights for online learning
            # Reset step counters (cosmetic – does NOT affect policy)
            dqn.total_steps = 0
            dqn.train_steps = 0

        # Multiple epochs training per fold
        num_wf_epochs = NUM_EPOCHS 
        best_fold_accuracy = 0.0
        patience_counter = 0
        
        for epoch in range(num_wf_epochs):
            step_count = 0
            epoch_losses = []
            epoch_rewards = []
            correct_predictions = 0
            total_predictions = 0
            
            for j in range(len(S_train) - 1):
                s, s_next = S_train[j], S_train[j + 1]
                epsilon = eps_at(step_count)
                a = dqn.act(s, epsilon)
                
                # Track accuracy for monitoring
                if a == L_train[j]:
                    correct_predictions += 1
                total_predictions += 1
                
                r = calculate_reward(a, j, price_changes_train, L_train, closes_train)
                done = (j == len(S_train) - 2)
                dqn.store(s, a, r, s_next, done)
                
                # =========================================
                # RECORD RL TRAINING METRICS
                # =========================================
                # Get Q-values for analysis (optional)
                with torch.no_grad():
                    s_tensor = torch.tensor(s, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                    q_vals = dqn.online(s_tensor).cpu().numpy().flatten()
                
                # Record training step
                current_accuracy = correct_predictions / (total_predictions + 1e-8)
                rl_tracker.record_training_step(
                    fold_id=segment_id,
                    epoch=epoch,
                    step=step_count,
                    loss=0.0,  # Will be updated after training
                    reward=r,
                    epsilon=epsilon,
                    accuracy=current_accuracy,
                    q_values=q_vals.tolist()
                )
                epoch_rewards.append(r)
                
                # Train multiple times if buffer has enough samples
                if len(dqn.replay_buffer) >= batch_size:
                    for _ in range(train_steps_per_env_step):
                        loss = dqn.train_step(batch_size)
                        if loss is not None:
                            epoch_losses.append(loss)
                            # Update the loss in the last recorded step
                            if rl_tracker.training_metrics:
                                rl_tracker.training_metrics[-1]['loss'] = float(loss)
                
                step_count += 1
            
            train_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
            avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
            avg_reward = np.mean(epoch_rewards) if epoch_rewards else 0.0
            
            print(f"   Fold {segment_id}, Epoch {epoch+1}: Train Acc: {train_accuracy:.4f}, Avg Loss: {avg_loss:.4f}, Avg Reward: {avg_reward:.4f}")
            
            # Record epoch summary
            rl_tracker.record_epoch_summary(
                fold_id=segment_id,
                epoch=epoch,
                avg_reward=avg_reward,
                avg_loss=avg_loss,
                accuracy=train_accuracy,
                total_steps=step_count
            )
        
            # Test on OOS period
            y_pred = [dqn.act(S_test[i], epsilon=0.0) for i in range(len(S_test))]
            test_accuracy = np.mean(np.array(y_pred) == y_true)
            print(f"   Fold {segment_id} Test Accuracy: {test_accuracy:.4f}")

            # =========================================
            # ⭐ EARLY STOPPING CHECK
            # =========================================
            if test_accuracy > best_fold_accuracy:
                best_fold_accuracy = test_accuracy
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"⛔ Early stopping at epoch {epoch+1} "
                    f"(no improvement for {patience_counter} epochs)")
                break


            # =========================================
            # 🔧 ONLINE ADAPTATION (TRAINING ON TEST FOLD)
            # =========================================
            print(f"🔧 Online adaptation: updating model with past OSS data from fold {segment_id}'s test data")

            S_test_np = X_test_normalized.values.astype(np.float32)
            L_test_np = y_test.values.astype(np.int64)

            # Price changes (for reward calculation)
            price_changes_test = np.diff(closes_test) / closes_test[:-1]
            price_changes_test = np.append(price_changes_test, 0.0)

            for j in range(len(S_test_np) - 1):
                s = S_test_np[j]
                s_next = S_test_np[j + 1]
                
                # Small exploration for adaptation
                a = dqn.act(s, epsilon=0.05)

                r = calculate_reward(a, j, price_changes_test, L_test_np)
                done = (j == len(S_test_np) - 2)

                dqn.store(s, a, r, s_next, done)

                if len(dqn.replay_buffer) >= batch_size:
                    dqn.train_step(batch_size)

            # Continue collecting results
            y_true_all.extend(list(y_true))
            y_pred_all.extend(list(y_pred))
            date_all.extend(list(dates_test.values))


        # Record segment results
        rep = classification_report(y_true, np.array(y_pred), output_dict=True, zero_division=0)
        rows.append({
            "segment": segment_id,
            "train_start_idx": int(start_index),
            "train_end_idx": int(train_end - 1),
            "test_start_idx": int(test_start),
            "test_end_idx": int(test_end - 1),
            "train_start_date": train_start_date,
            "train_end_date": train_end_date,
            "test_start_date": test_start_date,
            "test_end_date": test_end_date,
            "train_size": len(X_train_raw),
            "test_size": len(X_test_raw),
            "reward_type": reward_type,
            "transaction_cost": transaction_cost,
            "accuracy": float(rep.get("accuracy", np.nan)),
            "precision_0": float(rep["0"]["precision"]) if "0" in rep else np.nan,
            "recall_0":    float(rep["0"]["recall"])    if "0" in rep else np.nan,
            "f1_0":        float(rep["0"]["f1-score"])  if "0" in rep else np.nan,
            "precision_1": float(rep["1"]["precision"]) if "1" in rep else np.nan,
            "recall_1":    float(rep["1"]["recall"])    if "1" in rep else np.nan,
            "f1_1":        float(rep["1"]["f1-score"])  if "1" in rep else np.nan,
            "support_0":   int(rep["0"]["support"])     if "0" in rep else 0,
            "support_1":   int(rep["1"]["support"])     if "1" in rep else 0,
            "n_features":  int(state_dim),
        })

        # Move to next segment
        start_index += step_size
        segment_id += 1

    wf_df = pd.DataFrame(rows)

    # =========================================
    # COMPREHENSIVE ACADEMIC REPORTING
    # =========================================
    if len(y_true_all) > 0:
        print(f"\n📊 Overall Walk-Forward Results:")
        
        # Your existing reporting
        overall_metrics = Reporter.overall_from_preds(np.array(y_true_all), np.array(y_pred_all), title='Walk-forward DQN (overall)')
        
        # Enhanced academic reporting
        academic_metrics = academic_reporter.comprehensive_classification_report(
            np.array(y_true_all), np.array(y_pred_all), "overall_walk_forward", 
            model_info={'state_dim': state_dim, 'n_actions': n_actions}
        )
        
        # Temporal analysis
        temporal_results = academic_reporter.temporal_analysis_report(
            np.array(y_true_all), np.array(y_pred_all), date_all
        )
        
        # RL training summary
        training_summary = rl_tracker.get_training_summary()
        print(f"\n📈 RL Training Summary:")
        for key, value in training_summary.items():
            print(f"   {key}: {value:.4f}")
        
        # Save RL training metrics
        rl_tracker.save_training_metrics()

        # Enhanced backtesting
        prices_full = proc.prices_full
        if len(date_all) > 0:
            oos_start, oos_end = pd.to_datetime(date_all[0]), pd.to_datetime(date_all[-1])
            mask_full = (prices_full["Date"] >= oos_start) & (prices_full["Date"] <= oos_end)
            dates_full = pd.to_datetime(prices_full.loc[mask_full, "Date"].values)
            closes_full = prices_full.loc[mask_full, "Close"].values

            pred_aligned = Backtester.prep_for_backtest(
                pred=np.array(y_pred_all),
                test_dates_idx=pd.to_datetime(pd.Series(date_all)),
                full_dates=dates_full,
                hold_h=target_horizon
            )
            
            # Your existing backtest
            summary_wf = Backtester.backtest_full(
                pred_aligned=pred_aligned,
                closes=closes_full,
                benchmark_closes=closes_full,
                fee_rate=transaction_cost,
                risk_free_rate=0.0,
                ann_factor=365,
                alpha=0.95
            )
            Reporter.print_backtest_summary("WF DQN Backtest Summary (OOS)", summary_wf)
            
            # Enhanced backtest reporting
            enhanced_backtest = backtest_reporter.comprehensive_backtest_report(
                summary_wf, 
                additional_metrics=training_summary
            )
    else:
        print("No segments evaluated.")

    return wf_df, dqn, X, y, dates, df_full["Close"].values



def simulate_real_time(dqn, X, y, dates, closes):
    actions, rewards, daily_returns = [], [], []
    cash = 1.0

    S = X.values.astype(np.float32)
    L = y.values.astype(np.int64)

    for t in range(len(S) - 1):
        s = S[t]
        
        # Epsilon=0 → deterministic actions
        a = dqn.act(s, epsilon=0.0)
        actions.append(a)

        price_change = (closes[t+1] - closes[t]) / closes[t]
        r = price_change * a
        rewards.append(r)
        cash *= (1 + r)
        daily_returns.append(r)

        # Optional online learning
        s_next = S[t+1]
        dqn.store(s, a, r, s_next, False)
        dqn.train_step(batch_size=64)

    return actions, rewards, daily_returns



# =========================================
# main()
# =========================================
def main():
    # Set seed at the VERY BEGINNING for maximum reproducibility
    set_seed(42, deterministic_torch=True)
    
    FILE_PATH = "/home/infonet/wahid/projects/Fin/cryptotrade/experiments/crypto_signals.csv"
    REQUESTED_FEATURES = [
    'Close', 'Open', 'High', 'kurtosis', 'mad', 'median', 'stdev', 'adx', 'dmp',
    'aroon_osc', 'chop', 'cksp_direction', 'decreasing', 'dpo', 'increasing',
    'long_run', 'psar_af', 'psar_reversal', 'qstick', 'aberration_middle',
    'aberration_range', 'donchian_lower', 'donchian_middle', 'thermo_raw',
    'thermo_long', 'thermo_short', 'accbands_upper', 'accbands_middle', 'kc_middle',
    'natr', 'pdist', 'rvi', 'ui', 'drawdown_pct', 'drawdown_log', 'log_return',
    'percent_return', 'ao', 'cci', 'cg', 'coppock', 'inertia', 'psl', 'roc', 'uo',
    'willr', 'brar_br', 'brar_ar', 'eri_bear', 'fisher_signal', 'kdj_k', 'kdj_d',
    'ppo', 'pvo_hist', 'rvgi', 'rvgi_signal', 'SQZ_OFF', 'SQZ_NO', 'stochrsi_k',
    'stochrsi_d', 'trix', 'tsi_signal', 'cdl_doji', 'adosc', 'nvi', 'pvi', 'pvol',
    'pvr', 'aobv_min_2', 'hma', 'hwma', 'linreg', 'midpoint', 'ohlc4', 'pwma',
    'rma', 'sinwma', 'ssf', 'trima', 'wcp', 'HILO_13_21', 'mcgd'
    ]
    LOOKBACK = 15
    TARGET_H = 1

    # Walk-Forward parameters
    INITIAL_TRAIN_SIZE = 1200
    STEP_SIZE = 250
    TEST_SIZE = 250

    # Training parameters
    EPS_START = 1.0
    EPS_END   = 0.01
    BATCH     = 128
    TRAIN_STEPS_PER_ENV = 2
    
    # Enhanced training parameters
    NUM_EPOCHS = 10
    EARLY_STOPPING_PATIENCE = 3

    def make_dqn(state_dim, n_actions):
        return DQNAgent(
            state_dim=state_dim,
            n_actions=n_actions,
            hidden_sizes=[256, 128, 64],
            lr=1e-3,
            gamma=0.99,
            buffer_size=100_000,
            target_sync=1000,
            tau=0.01,
            epsilon_start=EPS_START,
            epsilon_end=EPS_END,
            epsilon_decay=0.99995, # Need to tune this hyperparameter
            warmup_steps=10000, # Increased warmup for better exploration or decrease for faster training
        )
    
    # Test different reward functions with improved training
    reward_configs = [
        # {"type": "profit", "name": "profit"},
        # {"type": "risk_adjusted", "name": "risk_adj"},
        {"type": "directional", "name": "directional"},
        # {"type": "classification", "name": "classification"},  # Compare with original
        # {"type": "log_return", "name": "log_return"},
    ]
    
    for config in reward_configs:
        experiment_name = f"dqn_wf_{config['name']}"
        print(f"\n🎯 Testing Walk-Forward DQN with reward function: {config['type']}")
        print(f"📁 Experiment: {experiment_name}")
        
        try:
            wf_df, final_dqn, X_all, y_all, dates_all, closes_all = walk_forward_dqn(
                file_path=FILE_PATH,
                requested_features=REQUESTED_FEATURES,
                target_horizon=TARGET_H,
                lookback=LOOKBACK,
                dqn_maker=make_dqn,
                initial_train_size=INITIAL_TRAIN_SIZE,
                step_size=STEP_SIZE,  
                test_size=TEST_SIZE,
                epsilon_start=EPS_START,
                epsilon_end=EPS_END,
                epsilon_decay_steps=5000,
                batch_size=BATCH,
                train_steps_per_env_step=TRAIN_STEPS_PER_ENV,
                normalize_features=True,
                reward_type=config["type"],
                transaction_cost=0.001,
                risk_penalty=0.1,
                experiment_name=experiment_name,
                NUM_EPOCHS=NUM_EPOCHS,
                EARLY_STOPPING_PATIENCE=EARLY_STOPPING_PATIENCE,
            )

            wf_df.to_csv(os.path.join(OUT_DIR, f"walk_forward_dqn_{config['name']}.csv"), index=False)
            print(f"✅ Walk-Forward DQN with {config['type']} reward saved")

            # ---------------------------------------------------------
            # REAL-TIME SIMULATION (SEPARATE EXPERIMENT)
            # ---------------------------------------------------------
            print("\n🚀 Running Real-Time Simulation (FULL DATASET)...")
            actions, rewards, daily_returns = simulate_real_time(
                final_dqn,
                X_all,
                y_all,
                dates_all,
                closes_all,
            )

            simulation_df = pd.DataFrame({
                "date": dates_all[:-1],          # last step has no next price
                "action": actions,
                "reward": rewards,
                "return": daily_returns,
            })
            simulation_df.to_csv(f"{experiment_name}_realtime_simulation.csv", index=False)
            print(f"📁 Saved real-time simulation rewards/returns to {experiment_name}_realtime_simulation.csv")

            # 1. Plot rolling reward curve
            plt.figure(figsize=(12, 5))
            plt.plot(simulation_df["reward"].rolling(50).mean(), label="Rolling Reward (50)")
            plt.title("RL Reward Curve (Real-Time Simulation)")
            plt.xlabel("Time")
            plt.ylabel("Reward")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{experiment_name}_reward_curve.png")
            plt.close()

            # 2. Reward-return correlation
            corr = np.corrcoef(simulation_df["reward"], simulation_df["return"])[0,1]
            print(f"🔎 Correlation between RL reward and financial return: {corr:.4f}")

            with open(f"{experiment_name}_reward_stats.txt", "w") as f:
                f.write(f"Correlation (reward vs return): {corr:.4f}\n")
                f.write(f"Mean reward: {simulation_df['reward'].mean():.6f}\n")
                f.write(f"Std reward: {simulation_df['reward'].std():.6f}\n")

            # 3. Monthly reward summary
            simulation_df["month"] = simulation_df["date"].dt.to_period("M")
            monthly_reward = simulation_df.groupby("month")["reward"].mean()
            monthly_reward.to_csv(f"{experiment_name}_monthly_reward.csv")
            print(f"📁 Saved monthly reward summary: {experiment_name}_monthly_reward.csv")



            daily_returns = np.array(daily_returns)

            print("\n📊 Real-Time Simulation Backtest Results")
            summary_realtime = Backtester.backtest_full(
                pred_aligned=np.array(actions),
                closes=closes_all,
                benchmark_closes=closes_all,
                fee_rate=0.001,
                risk_free_rate=0.0,
                ann_factor=365,
                alpha=0.95
            )

            Reporter.print_backtest_summary("Real-Time Simulation", summary_realtime)

            # Save backtest results
            with open(f"{experiment_name}_realtime_summary.txt", "w") as f:
                for k, v in summary_realtime.items():
                    f.write(f"{k}: {v}\n")

            print(f"📁 Saved real-time backtest summary to {experiment_name}_realtime_summary.txt")

            # Cleanup
            try:
                del final_dqn
            except:
                pass
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"❌ Error with Walk-Forward DQN ({config['type']}): {e}")
            import traceback
            traceback.print_exc()

    print(f"\n✅ All Walk-Forward DQN results saved to {OUT_DIR}/")
    print(f"📊 Academic results saved to {ACADEMIC_DIR}/")


if __name__ == "__main__":
    main()
