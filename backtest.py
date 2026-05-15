"""
backtest.py — Bar-by-bar strategy backtesting engine.

Usage
-----
# With MT5 (must be running):
  python backtest.py --symbol XAUUSD --timeframe M15 --bars 5000 --plot

# With a CSV file (columns: time, open, high, low, close, volume):
  python backtest.py --csv data/EURUSD_M15.csv --symbol EURUSD --plot

# Walk-forward analysis (6 train / 2 test splits):
  python backtest.py --symbol XAUUSD --walk-forward

# Monte Carlo simulation (1000 equity paths):
  python backtest.py --symbol XAUUSD --monte-carlo 1000
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from utils    import compute_indicators, detect_market_structure
from strategy import generate_signal

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger('backtest')


# ── Symbol metadata ───────────────────────────────────────────────────────────

SYMBOL_META = {
    'EURUSD': {'contract': 100_000, 'point_value': 100_000},
    'GBPUSD': {'contract': 100_000, 'point_value': 100_000},
    'USDJPY': {'contract': 100_000, 'point_value':       1},
    'XAUUSD': {'contract':     100, 'point_value':     100},
    'XAGUSD': {'contract':   5_000, 'point_value':   5_000},
    'DEFAULT':{'contract': 100_000, 'point_value': 100_000},
}


def _point_value(symbol: str) -> float:
    meta = SYMBOL_META.get(symbol.upper(), SYMBOL_META['DEFAULT'])
    return float(meta['point_value'])


# ── Data loading ──────────────────────────────────────────────────────────────

def load_csv(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, parse_dates=['time'])
    df.columns = [c.lower() for c in df.columns]
    df.set_index('time', inplace=True)
    df.sort_index(inplace=True)
    return df[['open', 'high', 'low', 'close', 'volume']].copy()


def download_from_mt5(symbol: str, timeframe: str, bars: int) -> pd.DataFrame:
    try:
        import MetaTrader5 as mt5
    except ImportError:
        raise RuntimeError("MetaTrader5 package not installed.")

    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

    tf_map = {
        'M1':  mt5.TIMEFRAME_M1,  'M5':  mt5.TIMEFRAME_M5,
        'M15': mt5.TIMEFRAME_M15, 'M30': mt5.TIMEFRAME_M30,
        'H1':  mt5.TIMEFRAME_H1,  'H4':  mt5.TIMEFRAME_H4,
        'D1':  mt5.TIMEFRAME_D1,
    }
    tf = tf_map.get(timeframe.upper(), mt5.TIMEFRAME_M15)

    rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)
    mt5.shutdown()

    if rates is None:
        raise RuntimeError(f"No data for {symbol} {timeframe}: {mt5.last_error()}")

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('time', inplace=True)
    df.rename(columns={'tick_volume': 'volume'}, inplace=True)
    return df[['open', 'high', 'low', 'close', 'volume']].copy()


# ── Backtesting engine ────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Simulates the live strategy bar by bar.

    Assumptions
    -----------
    • Orders execute at the close of the signal bar (conservative).
    • SL / TP are checked on the next bar's high / low.
    • Commission = flat $/lot round-turn (configurable).
    • One position per symbol at a time.
    """

    def __init__(
        self,
        config: dict,
        symbol: str,
        initial_balance: float = 10_000.0,
        commission_per_lot: float = 7.0,
    ):
        self.config      = config
        self.symbol      = symbol.upper()
        self.balance     = initial_balance
        self.peak_bal    = initial_balance
        self._init_bal   = initial_balance
        self.commission  = commission_per_lot
        self.pv          = _point_value(self.symbol)

        self.open_trade:   dict = {}
        self.trades:       list = []
        self.equity_curve: list = []

        self._risk = config['risk']

    # ── Position sizing ───────────────────────────────────────────────────────

    def _calc_lot(self, sl_distance: float) -> float:
        risk_amt    = self.balance * self._risk['risk_per_trade']
        val_per_lot = sl_distance * self.pv
        if val_per_lot <= 0:
            return 0.01
        lot = risk_amt / val_per_lot
        return max(0.01, min(100.0, round(lot, 2)))

    # ── Single-bar simulation ─────────────────────────────────────────────────

    def _process_bar(self, i: int, df: pd.DataFrame) -> None:
        bar = df.iloc[i]

        if self.open_trade:
            ot = self.open_trade
            if ot['direction'] == 'BUY':
                if float(bar['low']) <= ot['sl']:
                    self._close(bar, ot['sl'], 'SL')
                elif float(bar['high']) >= ot['tp']:
                    self._close(bar, ot['tp'], 'TP')
            else:
                if float(bar['high']) >= ot['sl']:
                    self._close(bar, ot['sl'], 'SL')
                elif float(bar['low']) <= ot['tp']:
                    self._close(bar, ot['tp'], 'TP')

        mtm = 0.0
        if self.open_trade:
            ot       = self.open_trade
            entry    = ot['entry']
            close_px = float(bar['close'])
            raw_pnl  = (close_px - entry) if ot['direction'] == 'BUY' else (entry - close_px)
            mtm      = raw_pnl * ot['lot'] * self.pv

        self.equity_curve.append({
            'time':    bar.name,
            'balance': self.balance,
            'equity':  self.balance + mtm,
        })

        if self.open_trade:
            return

        dd = (self.peak_bal - self.balance) / self.peak_bal if self.peak_bal > 0 else 0.0
        if dd >= self._risk['max_drawdown']:
            return

        window   = df.iloc[: i + 1].copy()
        bar_time = bar.name
        signal, atr, last_sh, last_sl = generate_signal(window, self.config, bar_time)

        if signal == 'HOLD' or atr <= 0:
            return

        entry = float(bar['close'])
        sl_m  = self._risk['atr_sl_multiplier']
        tp_m  = self._risk['atr_tp_multiplier']

        if signal == 'BUY':
            sl = entry - atr * sl_m
            tp = entry + atr * tp_m
        else:
            sl = entry + atr * sl_m
            tp = entry - atr * tp_m

        sl_dist = abs(entry - sl)
        tp_dist = abs(entry - tp)
        rr      = tp_dist / sl_dist if sl_dist > 0 else 0.0

        if rr < self._risk['min_rr_ratio']:
            return

        lot = self._calc_lot(sl_dist)

        self.open_trade = {
            'direction': signal,
            'entry':     entry,
            'sl':        sl,
            'tp':        tp,
            'lot':       lot,
            'open_time': bar.name,
        }

    def _close(self, bar, close_price: float, reason: str) -> None:
        ot  = self.open_trade
        raw = ((close_price - ot['entry']) if ot['direction'] == 'BUY'
               else (ot['entry'] - close_price))
        pnl = raw * ot['lot'] * self.pv - self.commission * ot['lot']

        self.balance += pnl
        if self.balance > self.peak_bal:
            self.peak_bal = self.balance

        self.trades.append({
            'open_time':   ot['open_time'],
            'close_time':  bar.name,
            'direction':   ot['direction'],
            'entry':       ot['entry'],
            'close_price': close_price,
            'sl':          ot['sl'],
            'tp':          ot['tp'],
            'lot':         ot['lot'],
            'pnl':         round(pnl, 2),
            'reason':      reason,
        })
        self.open_trade = {}

    # ── Full run ──────────────────────────────────────────────────────────────

    def run(self, df: pd.DataFrame) -> dict:
        cfg     = self.config['strategy']
        min_idx = cfg['ema_slow'] + cfg['atr_period'] + cfg['structure_lookback'] * 4 + 20

        logger.info(
            f"Backtest | {self.symbol} | {len(df)} bars "
            f"({df.index[0]} → {df.index[-1]})"
        )

        for i in range(min_idx, len(df)):
            self._process_bar(i, df)

        if self.open_trade:
            ot       = self.open_trade
            last_bar = df.iloc[-1]
            self._close(last_bar, float(last_bar['close']), 'EOD')

        return self._compute_metrics()

    # ── Metrics ───────────────────────────────────────────────────────────────

    def _compute_metrics(self) -> dict:
        if not self.trades:
            return {'note': 'No trades executed. Relax strategy filters or use more data.'}

        tr   = pd.DataFrame(self.trades)
        eq   = pd.DataFrame(self.equity_curve)
        wins = tr[tr['pnl'] > 0]
        loss = tr[tr['pnl'] < 0]

        win_rate     = len(wins) / len(tr) * 100
        gross_profit = wins['pnl'].sum() if len(wins) else 0.0
        gross_loss   = abs(loss['pnl'].sum()) if len(loss) else 0.0
        profit_factor= gross_profit / gross_loss if gross_loss > 0 else float('inf')
        net_profit   = tr['pnl'].sum()
        net_pct      = net_profit / self._init_bal * 100

        eq['peak']   = eq['equity'].cummax()
        eq['dd']     = (eq['peak'] - eq['equity']) / eq['peak'].replace(0, np.nan)
        max_dd       = float(eq['dd'].max()) * 100

        avg_win  = float(wins['pnl'].mean()) if len(wins) else 0.0
        avg_loss = float(loss['pnl'].mean()) if len(loss) else 0.0
        expect   = (win_rate / 100 * avg_win) + ((1 - win_rate / 100) * avg_loss)

        # Daily equity returns for Sharpe / Sortino
        eq_ts    = eq.set_index('time')['equity']
        try:
            eq_daily = eq_ts.resample('D').last().dropna()
        except Exception:
            eq_daily = eq_ts

        eq_ret   = eq_daily.pct_change().dropna()

        sharpe = 0.0
        if eq_ret.std() > 0:
            sharpe = float(eq_ret.mean() / eq_ret.std() * (252 ** 0.5))

        # Sortino ratio (downside deviation only)
        sortino = 0.0
        neg_ret = eq_ret[eq_ret < 0]
        if len(neg_ret) > 0:
            downside_std = float(neg_ret.std())
            if downside_std > 0:
                sortino = float(eq_ret.mean() / downside_std * (252 ** 0.5))

        # Calmar ratio
        calmar = (net_pct / max_dd) if max_dd > 0 else 0.0

        return {
            'symbol':          self.symbol,
            'bars_tested':     len(eq),
            'total_trades':    len(tr),
            'winning_trades':  int(len(wins)),
            'losing_trades':   int(len(loss)),
            'win_rate_pct':    round(win_rate, 2),
            'profit_factor':   round(profit_factor, 3),
            'net_profit':      round(net_profit, 2),
            'net_profit_pct':  round(net_pct, 2),
            'max_drawdown_pct':round(max_dd, 2),
            'avg_win':         round(avg_win, 2),
            'avg_loss':        round(avg_loss, 2),
            'expectancy':      round(expect, 2),
            'sharpe_ratio':    round(sharpe, 3),
            'sortino_ratio':   round(sortino, 3),
            'calmar_ratio':    round(calmar, 3),
            'initial_balance': self._init_bal,
            'final_balance':   round(self.balance, 2),
        }

    # ── Walk-forward analysis ─────────────────────────────────────────────────

    def walk_forward(
        self, df: pd.DataFrame, n_splits: int = 6, test_ratio: float = 0.25
    ) -> list:
        """
        Walk-forward validation: train on expanding window, test on next slice.
        Returns list of per-fold metric dicts.
        """
        total    = len(df)
        fold_len = total // (n_splits + 1)
        results  = []

        logger.info(
            f"Walk-forward | {n_splits} folds | "
            f"fold_len≈{fold_len} bars"
        )

        for i in range(n_splits):
            train_end = fold_len * (i + 1)
            test_end  = min(train_end + int(fold_len * test_ratio), total)

            train_df = df.iloc[:train_end]
            test_df  = df.iloc[train_end:test_end]

            if len(test_df) < 100:
                continue

            eng = BacktestEngine(
                self.config, self.symbol, self._init_bal, self.commission
            )
            metrics = eng.run(test_df.copy())
            metrics['fold'] = i + 1
            metrics['train_bars'] = len(train_df)
            metrics['test_bars']  = len(test_df)
            results.append(metrics)

            logger.info(
                f"  Fold {i+1}: WR={metrics.get('win_rate_pct',0):.1f}%  "
                f"PF={metrics.get('profit_factor',0):.2f}  "
                f"Sharpe={metrics.get('sharpe_ratio',0):.2f}  "
                f"MaxDD={metrics.get('max_drawdown_pct',0):.1f}%"
            )

        return results

    # ── Monte Carlo simulation ────────────────────────────────────────────────

    def monte_carlo(self, n_simulations: int = 1000) -> dict:
        """
        Resample the historical trade sequence to simulate N equity paths.
        Reports percentile distribution of key metrics.
        """
        if not self.trades:
            return {}

        pnls = np.array([t['pnl'] for t in self.trades])
        n    = len(pnls)
        rng  = np.random.default_rng(42)

        final_balances = []
        max_drawdowns  = []
        sharpes        = []

        for _ in range(n_simulations):
            sample  = rng.choice(pnls, size=n, replace=True)
            equity  = self._init_bal + np.cumsum(sample)
            peak    = np.maximum.accumulate(np.append(self._init_bal, equity))
            dd      = (peak[1:] - equity) / peak[1:]
            final_balances.append(float(equity[-1]))
            max_drawdowns.append(float(dd.max() * 100))

            daily_len = max(1, n // 20)
            daily_ret = equity[daily_len-1::daily_len] / np.append(
                self._init_bal, equity[daily_len-1:-1:daily_len]
            ) - 1
            if daily_ret.std() > 0:
                sharpes.append(float(daily_ret.mean() / daily_ret.std() * (252 ** 0.5)))

        fb = np.array(final_balances)
        dd = np.array(max_drawdowns)
        sh = np.array(sharpes) if sharpes else np.zeros(1)

        result = {
            'simulations': n_simulations,
            'trades_per_sim': n,
            'balance_p5':   round(float(np.percentile(fb,  5)), 2),
            'balance_p25':  round(float(np.percentile(fb, 25)), 2),
            'balance_p50':  round(float(np.percentile(fb, 50)), 2),
            'balance_p75':  round(float(np.percentile(fb, 75)), 2),
            'balance_p95':  round(float(np.percentile(fb, 95)), 2),
            'prob_profit':  round(float((fb > self._init_bal).mean() * 100), 1),
            'dd_p50_pct':   round(float(np.percentile(dd, 50)), 2),
            'dd_p95_pct':   round(float(np.percentile(dd, 95)), 2),
            'sharpe_p50':   round(float(np.percentile(sh, 50)), 3),
        }
        return result

    # ── Plotting ──────────────────────────────────────────────────────────────

    def plot(self) -> None:
        try:
            import plotly.graph_objects as go
        except ImportError:
            logger.warning("plotly not installed — cannot show chart.")
            return

        if not self.equity_curve:
            print("No equity data to plot.")
            return

        eq  = pd.DataFrame(self.equity_curve)
        tr  = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=eq['time'], y=eq['equity'], name='Equity',
            line=dict(color='#00d4aa', width=2),
            fill='tozeroy', fillcolor='rgba(0,212,170,0.07)',
        ))
        fig.add_trace(go.Scatter(
            x=eq['time'], y=eq['balance'], name='Balance',
            line=dict(color='#4e9af1', width=1.5, dash='dot'),
        ))

        if not tr.empty:
            wins = tr[tr['pnl'] > 0]
            loss = tr[tr['pnl'] < 0]
            eq_idx = eq.set_index('time')['equity']

            for sub, color, name in [(wins, '#00d4aa', 'Win'), (loss, '#ff4b4b', 'Loss')]:
                if not sub.empty:
                    y_vals = [float(eq_idx.iloc[max(0, eq_idx.index.searchsorted(t) - 1)])
                               for t in sub['close_time']]
                    fig.add_trace(go.Scatter(
                        x=sub['close_time'], y=y_vals,
                        mode='markers',
                        marker=dict(color=color, size=7, symbol='circle'),
                        name=name,
                    ))

        fig.update_layout(
            title=f'Backtest Equity Curve — {self.symbol}',
            xaxis_title='Time', yaxis_title='Account Value ($)',
            template='plotly_dark', height=520,
            legend=dict(orientation='h', yanchor='bottom', y=1.02),
        )
        fig.show()


# ── CLI entry point ───────────────────────────────────────────────────────────

def _load_config(path: str = 'config.yaml') -> dict:
    with open(path, encoding='utf-8') as fh:
        return yaml.safe_load(fh)


def _print_section(title: str, data: dict) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)
    for k, v in data.items():
        print(f"  {k:<28}: {v}")


def main() -> None:
    parser = argparse.ArgumentParser(description='Run a strategy backtest')
    parser.add_argument('--symbol',       default='XAUUSD',  help='Trading symbol')
    parser.add_argument('--timeframe',    default='M15',     help='Timeframe e.g. M15 H1')
    parser.add_argument('--bars',         type=int, default=5_000)
    parser.add_argument('--balance',      type=float, default=10_000.0)
    parser.add_argument('--commission',   type=float, default=7.0)
    parser.add_argument('--csv',          default=None)
    parser.add_argument('--plot',         action='store_true')
    parser.add_argument('--save-csv',     default=None)
    parser.add_argument('--walk-forward', action='store_true',
                        help='Run walk-forward validation')
    parser.add_argument('--wf-splits',    type=int, default=6)
    parser.add_argument('--monte-carlo',  type=int, default=0,
                        help='Run N Monte Carlo simulations (0 = skip)')
    args = parser.parse_args()

    config = _load_config()

    if args.csv:
        df = load_csv(args.csv)
    else:
        logger.info(f"Downloading {args.bars} bars of {args.symbol} {args.timeframe}…")
        df = download_from_mt5(args.symbol, args.timeframe, args.bars)

    logger.info(f"Data loaded: {len(df)} bars  ({df.index[0]} → {df.index[-1]})")

    engine  = BacktestEngine(config, args.symbol, args.balance, args.commission)
    metrics = engine.run(df)
    _print_section(f"BACKTEST RESULTS  |  {args.symbol}  {args.timeframe}", metrics)

    if args.walk_forward:
        engine2 = BacktestEngine(config, args.symbol, args.balance, args.commission)
        engine2.run(df)   # populate trades so walk_forward has history
        wf_engine = BacktestEngine(config, args.symbol, args.balance, args.commission)
        wf_results = wf_engine.walk_forward(df, n_splits=args.wf_splits)
        if wf_results:
            avg_wr  = np.mean([r.get('win_rate_pct', 0)   for r in wf_results])
            avg_pf  = np.mean([r.get('profit_factor', 0)  for r in wf_results])
            avg_sh  = np.mean([r.get('sharpe_ratio', 0)   for r in wf_results])
            avg_dd  = np.mean([r.get('max_drawdown_pct',0) for r in wf_results])
            _print_section(f"WALK-FORWARD SUMMARY  ({len(wf_results)} folds)", {
                'avg_win_rate_pct':   round(avg_wr, 2),
                'avg_profit_factor':  round(avg_pf, 3),
                'avg_sharpe_ratio':   round(avg_sh, 3),
                'avg_max_drawdown':   round(avg_dd, 2),
            })

    if args.monte_carlo > 0:
        mc = engine.monte_carlo(args.monte_carlo)
        _print_section(f"MONTE CARLO  ({args.monte_carlo} simulations)", mc)

    if args.save_csv and engine.trades:
        out = Path(args.save_csv)
        pd.DataFrame(engine.trades).to_csv(out, index=False)
        logger.info(f"Trade list saved to {out}")

    if args.plot:
        engine.plot()


if __name__ == '__main__':
    main()
