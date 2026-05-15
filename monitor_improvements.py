"""
monitor_improvements.py — Track AI model improvements over training cycles.

Reports:
- Model AUC trends
- Label distribution changes
- Feature importance shifts
- Training speed
"""

import json
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

logger = logging.getLogger('AI-Trade-Monitor')

MONITOR_FILE = Path('data/training_monitor.json')


def log_training_metrics(metrics: dict) -> None:
    """
    Append training metrics to monitoring file.
    
    metrics: {
        'timestamp': ISO string,
        'n_bars': int,
        'n_samples': int,
        'n_samples_pos': int (bullish),
        'auc_scores': [float, ...],
        'precision_scores': [float, ...],
        'recall_scores': [float, ...],
        'f1_scores': [float, ...],
        'top_features': [(name, importance), ...],
        'training_seconds': float,
    }
    """
    try:
        history = []
        if MONITOR_FILE.exists():
            try:
                history = json.loads(MONITOR_FILE.read_text())
            except Exception:
                history = []
        
        entry = {
            'timestamp': metrics.get('timestamp', datetime.utcnow().isoformat()),
            'n_bars': metrics.get('n_bars', 0),
            'n_samples': metrics.get('n_samples', 0),
            'pos_ratio': metrics.get('pos_ratio', 0.0),
            'auc_mean': np.mean(metrics.get('auc_scores', [0.5])),
            'auc_std': np.std(metrics.get('auc_scores', [0])),
            'precision_mean': np.mean(metrics.get('precision_scores', [0])),
            'recall_mean': np.mean(metrics.get('recall_scores', [0])),
            'f1_mean': np.mean(metrics.get('f1_scores', [0])),
            'top_3_features': metrics.get('top_features', [])[:3],
            'training_seconds': metrics.get('training_seconds', 0),
        }
        
        history.append(entry)
        
        # Keep only last 100 training sessions
        if len(history) > 100:
            history = history[-100:]
        
        MONITOR_FILE.parent.mkdir(parents=True, exist_ok=True)
        MONITOR_FILE.write_text(json.dumps(history, indent=2))
        
    except Exception as exc:
        logger.warning(f"Failed to log monitoring metrics: {exc}")


def print_improvement_report() -> None:
    """Print summary of training improvements over time."""
    if not MONITOR_FILE.exists():
        logger.info("No training history yet.")
        return
    
    try:
        history = json.loads(MONITOR_FILE.read_text())
        if not history:
            logger.info("Training history is empty.")
            return
        
        df = pd.DataFrame(history)
        
        # Summary statistics
        logger.info("=" * 70)
        logger.info("TRAINING MONITOR REPORT")
        logger.info("=" * 70)
        
        logger.info(f"Total training cycles: {len(df)}")
        logger.info(f"Latest training: {df.iloc[-1]['timestamp']}")
        
        logger.info("\n--- Current Metrics (Latest Session) ---")
        latest = df.iloc[-1]
        logger.info(f"Samples: {int(latest['n_samples'])} (Bullish ratio: {latest['pos_ratio']:.1f}%)")
        logger.info(f"AUC: {latest['auc_mean']:.4f} ± {latest['auc_std']:.4f}")
        logger.info(f"Precision: {latest['precision_mean']:.4f}")
        logger.info(f"Recall: {latest['recall_mean']:.4f}")
        logger.info(f"F1: {latest['f1_mean']:.4f}")
        logger.info(f"Training Time: {latest['training_seconds']:.1f}s")
        
        # Trends
        if len(df) >= 2:
            logger.info("\n--- Trends (Last 2 Sessions) ---")
            prev = df.iloc[-2]
            
            auc_change = latest['auc_mean'] - prev['auc_mean']
            f1_change = latest['f1_mean'] - prev['f1_mean']
            
            logger.info(f"AUC Change: {auc_change:+.4f}")
            logger.info(f"F1 Change: {f1_change:+.4f}")
        
        logger.info("\n" + "=" * 70)
        
    except Exception as exc:
        logger.warning(f"Failed to generate report: {exc}")
