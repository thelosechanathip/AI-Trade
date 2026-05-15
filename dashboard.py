"""
dashboard.py — Real-time Streamlit trading dashboard.

Launch with:
  streamlit run dashboard.py

Data sources:
  data/state.json  — live account snapshot written by main.py every cycle
  data/trades.db   — SQLite trade history + equity curve
"""

import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI-Trade Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DB_PATH    = Path('data/trades.db')
STATE_PATH = Path('data/state.json')
REFRESH_S  = 5

# ── Data loaders ──────────────────────────────────────────────────────────────

def _load_state() -> dict:
    if not STATE_PATH.exists():
        return {}
    try:
        return json.loads(STATE_PATH.read_text(encoding='utf-8'))
    except Exception:
        return {}


def _load_equity_curve() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame(columns=['ts', 'balance', 'equity'])
    try:
        conn = sqlite3.connect(str(DB_PATH))
        df   = pd.read_sql(
            'SELECT ts, balance, equity FROM equity_curve ORDER BY ts',
            conn,
        )
        conn.close()
        df['ts'] = pd.to_datetime(df['ts'])
        return df
    except Exception:
        return pd.DataFrame(columns=['ts', 'balance', 'equity'])


def _load_trade_history() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(str(DB_PATH))
        df   = pd.read_sql(
            '''SELECT ticket, symbol, direction, lot_size,
                      entry_price, sl_price, tp_price,
                      open_time, close_time, close_price,
                      profit, status, ai_confidence
               FROM trades
               ORDER BY open_time DESC
               LIMIT 200''',
            conn,
        )
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _color_value(val):
    """Streamlit dataframe cell colour for profit column."""
    if pd.isna(val):
        return ''
    try:
        v = float(val)
        return 'color: #00d4aa' if v > 0 else ('color: #ff4b4b' if v < 0 else '')
    except Exception:
        return ''


def _badge(text: str, colour: str) -> str:
    return (
        f'<span style="background:{colour};padding:2px 8px;'
        f'border-radius:4px;font-size:12px;color:#fff">{text}</span>'
    )


# ── Main render ───────────────────────────────────────────────────────────────

def render():
    # Inject minimal custom CSS for dark-mode feel
    st.markdown(
        '''<style>
        div[data-testid="metric-container"] {
            background: #1e1e2e;
            border: 1px solid #313244;
            border-radius: 8px;
            padding: 8px 12px;
        }
        </style>''',
        unsafe_allow_html=True,
    )

    st.title("🤖 AI-Trade  |  Live Dashboard")
    st.caption(f"Last render: {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")

    state = _load_state()

    # ── Waiting screen ────────────────────────────────────────────────────────
    if not state:
        st.warning(
            "⏳ Waiting for the trading engine…  "
            "Start `python main.py` in a separate terminal."
        )
        time.sleep(REFRESH_S)
        st.rerun()
        return

    ts_str = state.get('timestamp', '')
    if ts_str:
        try:
            age = (datetime.utcnow() - datetime.fromisoformat(ts_str)).total_seconds()
            if age > 120:
                st.warning(f"⚠️ Engine data is {int(age)}s old — engine may be paused.")
        except Exception:
            pass

    # ── Top KPI row ───────────────────────────────────────────────────────────
    balance       = float(state.get('balance',  0))
    equity        = float(state.get('equity',   0))
    drawdown      = float(state.get('drawdown_pct', 0))
    daily_pnl     = float(state.get('daily_pnl', 0))
    stats         = state.get('stats', {})
    open_positions= state.get('open_positions', [])

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Balance",       f"${balance:,.2f}",   f"${daily_pnl:+,.2f}")
    c2.metric("Equity",        f"${equity:,.2f}")
    c3.metric("Drawdown",      f"{drawdown:.2f}%",
              delta_color="inverse" if drawdown > 0 else "normal")
    c4.metric("Win Rate",      f"{stats.get('win_rate', 0):.1f}%")
    c5.metric("Profit Factor", f"{stats.get('profit_factor', 0):.2f}")
    c6.metric("Total Trades",  stats.get('total_trades', 0))

    st.divider()

    # ── Open positions + account sidebar ─────────────────────────────────────
    col_left, col_right = st.columns([3, 1])

    with col_left:
        st.subheader(f"📊 Open Positions  ({len(open_positions)})")
        if open_positions:
            df_pos = pd.DataFrame(open_positions)
            # Format profit
            if 'profit' in df_pos.columns:
                df_pos['profit'] = df_pos['profit'].apply(lambda x: f"${x:+.2f}")
            st.dataframe(df_pos, use_container_width=True, hide_index=True)
        else:
            st.info("No open positions right now.")

    with col_right:
        st.subheader("💼 Account")
        st.metric("Free Margin",  f"${float(state.get('free_margin', 0)):,.2f}")
        st.metric("Used Margin",  f"${float(state.get('margin', 0)):,.2f}")
        st.metric("Net Profit",   f"${stats.get('total_profit', 0):+,.2f}")
        dd_color = "#ff4b4b" if drawdown > 5 else ("#f5a623" if drawdown > 2 else "#00d4aa")
        st.markdown(
            f"Drawdown&nbsp; {_badge(f'{drawdown:.2f}%', dd_color)}",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Equity curve ──────────────────────────────────────────────────────────
    st.subheader("💹 Equity Curve")
    eq_df = _load_equity_curve()

    if not eq_df.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=eq_df['ts'], y=eq_df['equity'],
            name='Equity', mode='lines',
            line=dict(color='#00d4aa', width=2),
            fill='tozeroy', fillcolor='rgba(0,212,170,0.05)',
        ))
        fig.add_trace(go.Scatter(
            x=eq_df['ts'], y=eq_df['balance'],
            name='Balance', mode='lines',
            line=dict(color='#4e9af1', width=1.5, dash='dot'),
        ))
        fig.update_layout(
            height=320, margin=dict(l=0, r=0, t=10, b=0),
            template='plotly_dark', showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Equity data will appear after the first trading cycle.")

    st.divider()

    # ── Trade history ─────────────────────────────────────────────────────────
    st.subheader("📋 Trade History")
    hist_df = _load_trade_history()

    if not hist_df.empty:
        col_tbl, col_dist = st.columns([2, 1])

        with col_tbl:
            display = hist_df.copy()
            if 'profit' in display.columns:
                styled = display.style.applymap(_color_value, subset=['profit'])
                st.dataframe(styled, use_container_width=True, hide_index=True)
            else:
                st.dataframe(display, use_container_width=True, hide_index=True)

        with col_dist:
            closed = hist_df[hist_df['status'] == 'closed'].copy()
            if not closed.empty and 'profit' in closed.columns:
                closed['profit'] = pd.to_numeric(closed['profit'], errors='coerce').dropna()

                wins   = (closed['profit'] > 0).sum()
                losses = (closed['profit'] < 0).sum()

                fig_pie = go.Figure(go.Pie(
                    labels=['Wins', 'Losses'],
                    values=[wins, losses],
                    marker_colors=['#00d4aa', '#ff4b4b'],
                    hole=0.5,
                ))
                fig_pie.update_layout(
                    height=200, margin=dict(l=0, r=0, t=10, b=0),
                    template='plotly_dark', showlegend=True,
                )
                st.plotly_chart(fig_pie, use_container_width=True)

                fig_hist = px.histogram(
                    closed, x='profit', nbins=25,
                    title='Profit Distribution',
                    color_discrete_sequence=['#4e9af1'],
                    template='plotly_dark',
                )
                fig_hist.add_vline(x=0, line_color='#ff4b4b', line_dash='dash')
                fig_hist.update_layout(
                    height=230, margin=dict(l=0, r=0, t=30, b=0),
                )
                st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("Trade history will appear after the first closed trade.")

    # ── Auto-refresh ──────────────────────────────────────────────────────────
    time.sleep(REFRESH_S)
    st.rerun()


if __name__ == '__main__':
    render()
