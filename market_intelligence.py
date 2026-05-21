"""
market_intelligence.py — Institutional-grade Market Intelligence Engine

Detects 10 market regimes and generates human-readable narratives for
every signal evaluation cycle. Used by strategy.py to apply additional
institutional-level hard blocks to signal generation.

Regime Classification:
  TREND_BULL     — Clean uptrend (ADX strong, aligned EMAs, price above both)
  TREND_BEAR     — Clean downtrend (ADX strong, aligned EMAs, price below both)
  RANGE          — Low ADX, oscillating price
  EXPANSION      — Volatility expansion (ATR rising, ADX just starting to climb)
  REVERSAL       — Trend reversal in progress (BOS/CHOCH + divergence)
  ACCUMULATION   — Sideways with bullish divergence / hidden buying pressure
  DISTRIBUTION   — Sideways with bearish divergence / hidden selling pressure
  LIQUIDITY_GRAB — Stop hunt followed by sharp reversal
  EXHAUSTION     — Trend overextended (far from EMA200, deceleration signals)
  HIGH_VOL       — Extreme volatility event (ATR > 2 standard deviations)

Detection Signals:
  - RSI divergence   (price new high/low but RSI doesn't confirm)
  - MACD divergence  (histogram diverging from price)
  - Displacement     (candle body > 2×ATR — institutional activity)
  - Liquidity sweep  (spike past swing S/R then sharp reversal)
  - BOS / CHOCH      (Break of Structure / Change of Character)
  - Volatility climax (ATR spike > 2σ above rolling mean)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger('AI-Trade')

# ── Regime constants ──────────────────────────────────────────────────────────
REGIME_TREND_BULL     = 'TREND_BULL'
REGIME_TREND_BEAR     = 'TREND_BEAR'
REGIME_RANGE          = 'RANGE'
REGIME_EXPANSION      = 'EXPANSION'
REGIME_REVERSAL       = 'REVERSAL'
REGIME_ACCUMULATION   = 'ACCUMULATION'
REGIME_DISTRIBUTION   = 'DISTRIBUTION'
REGIME_LIQUIDITY_GRAB = 'LIQUIDITY_GRAB'
REGIME_EXHAUSTION     = 'EXHAUSTION'
REGIME_HIGH_VOL       = 'HIGH_VOL'


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class MarketNarrative:
    """Complete market intelligence output for one analysis cycle."""

    # Regime classification
    regime: str = REGIME_RANGE
    regime_confidence: float = 0.5

    # Direction blocks (hard blocks applied in strategy.py)
    block_buy:    bool = False
    block_sell:   bool = False
    block_reason: str  = ''

    # Setup quality and uncertainty scores (0-1)
    setup_quality: float = 0.5
    uncertainty:   float = 0.5

    # Reversal composite
    reversal_detected:  bool = False
    reversal_direction: str  = ''   # 'UP' | 'DOWN'

    # Divergence signals
    rsi_divergence_bull:  bool = False
    rsi_divergence_bear:  bool = False
    macd_divergence_bull: bool = False
    macd_divergence_bear: bool = False

    # Institutional signals
    displacement_bull:    bool = False
    displacement_bear:    bool = False
    liquidity_sweep_bull: bool = False
    liquidity_sweep_bear: bool = False

    # Structure signals
    bos_bull:   bool = False
    bos_bear:   bool = False
    choch_bull: bool = False
    choch_bear: bool = False

    # Volatility signals
    volatility_climax:    bool = False
    volatility_expansion: bool = False

    # AI confidence adjustment (-0.15 to +0.15)
    confidence_adjustment: float = 0.0

    # Human-readable summary
    narrative:      str        = ''
    signals_active: List[str]  = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'regime':            self.regime,
            'regime_confidence': round(self.regime_confidence, 3),
            'block_buy':         self.block_buy,
            'block_sell':        self.block_sell,
            'block_reason':      self.block_reason,
            'setup_quality':     round(self.setup_quality, 3),
            'uncertainty':       round(self.uncertainty, 3),
            'reversal_detected': self.reversal_detected,
            'reversal_dir':      self.reversal_direction,
            'rsi_div_bull':      self.rsi_divergence_bull,
            'rsi_div_bear':      self.rsi_divergence_bear,
            'macd_div_bull':     self.macd_divergence_bull,
            'macd_div_bear':     self.macd_divergence_bear,
            'disp_bull':         self.displacement_bull,
            'disp_bear':         self.displacement_bear,
            'liq_sweep_bull':    self.liquidity_sweep_bull,
            'liq_sweep_bear':    self.liquidity_sweep_bear,
            'bos_bull':          self.bos_bull,
            'bos_bear':          self.bos_bear,
            'choch_bull':        self.choch_bull,
            'choch_bear':        self.choch_bear,
            'vol_climax':        self.volatility_climax,
            'vol_expansion':     self.volatility_expansion,
            'conf_adj':          round(self.confidence_adjustment, 3),
            'narrative':         self.narrative,
            'signals':           self.signals_active,
        }


# ── Swing point helpers ───────────────────────────────────────────────────────

def _find_swing_lows(arr: np.ndarray, strength: int = 3) -> List[Tuple[int, float]]:
    """Return (index, value) for each local minimum in arr."""
    result = []
    n = len(arr)
    for i in range(strength, n - strength):
        window_min = arr[i - strength: i + strength + 1].min()
        if arr[i] <= window_min:
            result.append((i, float(arr[i])))
    return result


def _find_swing_highs(arr: np.ndarray, strength: int = 3) -> List[Tuple[int, float]]:
    """Return (index, value) for each local maximum in arr."""
    result = []
    n = len(arr)
    for i in range(strength, n - strength):
        window_max = arr[i - strength: i + strength + 1].max()
        if arr[i] >= window_max:
            result.append((i, float(arr[i])))
    return result


# ── Main intelligence class ───────────────────────────────────────────────────

class MarketIntelligence:
    """
    Stateless market analysis engine.

    Call analyze() once per signal cycle with the M15 OHLCV+indicators DataFrame.
    Returns a MarketNarrative with all detected signals and direction blocks.
    """

    def analyze(
        self,
        df: pd.DataFrame,
        config: dict,
        htf_bias:    str   = 'NEUTRAL',
        htf_strength: float = 0.0,
    ) -> MarketNarrative:
        """
        Run full institutional-grade market analysis.

        Args:
            df:           M15 OHLCV + computed indicators (ema50, ema200, rsi,
                          macd_hist, atr, adx, bb_pct, stoch_k, etc.)
            config:       Full config dict
            htf_bias:     'BUY' | 'SELL' | 'NEUTRAL' from HTF filter
            htf_strength: 0-1 strength of HTF trend

        Returns:
            MarketNarrative with regime, blocks, setup quality, and narrative text
        """
        if len(df) < 40:
            return MarketNarrative(narrative='Insufficient data for MI analysis')

        mi_cfg  = config.get('market_intelligence', {})
        td_cfg  = config.get('trend_dominance', {})
        s_cfg   = config.get('strategy', {})
        narrative = MarketNarrative()
        signals   = []

        # ── Extract latest bar values ─────────────────────────────────────────
        latest = df.iloc[-1]
        close  = float(latest['close'])
        atr    = float(latest.get('atr', 1.0))
        adx    = float(latest.get('adx', 0.0))
        rsi    = float(latest.get('rsi', 50.0))
        ema50  = float(latest.get('ema50',  close))
        ema200 = float(latest.get('ema200', close))

        lookback  = mi_cfg.get('divergence_lookback', 35)
        strength  = mi_cfg.get('swing_strength',       3)
        disp_mult = mi_cfg.get('displacement_atr_mult', 2.0)

        # ── 1. RSI Divergence ─────────────────────────────────────────────────
        rsi_bull, rsi_bear = self._detect_rsi_divergence(df, lookback, strength)
        narrative.rsi_divergence_bull = rsi_bull
        narrative.rsi_divergence_bear = rsi_bear
        if rsi_bull: signals.append('RSI_DIV_BULL')
        if rsi_bear: signals.append('RSI_DIV_BEAR')

        # ── 2. MACD Divergence ────────────────────────────────────────────────
        macd_bull, macd_bear = self._detect_macd_divergence(df, lookback, strength)
        narrative.macd_divergence_bull = macd_bull
        narrative.macd_divergence_bear = macd_bear
        if macd_bull: signals.append('MACD_DIV_BULL')
        if macd_bear: signals.append('MACD_DIV_BEAR')

        # ── 3. Displacement Candles ───────────────────────────────────────────
        disp_bull, disp_bear = self._detect_displacement(df, disp_mult)
        narrative.displacement_bull = disp_bull
        narrative.displacement_bear = disp_bear
        if disp_bull: signals.append('DISP_BULL')
        if disp_bear: signals.append('DISP_BEAR')

        # ── 4. Liquidity Sweep ────────────────────────────────────────────────
        liq_bull, liq_bear = self._detect_liquidity_sweep(df, lookback)
        narrative.liquidity_sweep_bull = liq_bull
        narrative.liquidity_sweep_bear = liq_bear
        if liq_bull: signals.append('LIQ_SWEEP_BULL')
        if liq_bear: signals.append('LIQ_SWEEP_BEAR')

        # ── 5. BOS / CHOCH ────────────────────────────────────────────────────
        bos = self._detect_bos_choch(df, lookback, strength)
        narrative.bos_bull   = bos['bos_bull']
        narrative.bos_bear   = bos['bos_bear']
        narrative.choch_bull = bos['choch_bull']
        narrative.choch_bear = bos['choch_bear']
        for k, lbl in [('bos_bull', 'BOS_BULL'), ('bos_bear', 'BOS_BEAR'),
                        ('choch_bull', 'CHOCH_BULL'), ('choch_bear', 'CHOCH_BEAR')]:
            if bos[k]:
                signals.append(lbl)

        # ── 6. Volatility ─────────────────────────────────────────────────────
        vol_climax, vol_expand = self._detect_volatility(df)
        narrative.volatility_climax    = vol_climax
        narrative.volatility_expansion = vol_expand
        if vol_climax:  signals.append('VOL_CLIMAX')
        if vol_expand:  signals.append('VOL_EXPAND')

        # ── 7. Composite Reversal Score ───────────────────────────────────────
        rev_threshold = mi_cfg.get('reversal_score_threshold', 2)
        rev_up_score = sum([
            rsi_bull or macd_bull,
            bos['bos_bull'] or bos['choch_bull'],
            liq_bull,
            disp_bull,
        ])
        rev_dn_score = sum([
            rsi_bear or macd_bear,
            bos['bos_bear'] or bos['choch_bear'],
            liq_bear,
            disp_bear,
        ])
        if rev_up_score >= rev_threshold:
            narrative.reversal_detected  = True
            narrative.reversal_direction = 'UP'
            signals.append('REVERSAL_UP')
        elif rev_dn_score >= rev_threshold:
            narrative.reversal_detected  = True
            narrative.reversal_direction = 'DOWN'
            signals.append('REVERSAL_DOWN')

        # ── 8. Extended Regime Classification ────────────────────────────────
        exhaust_pct     = td_cfg.get('exhaustion_ema200_pct', 0.025)
        exhausted_above = close > ema200 * (1.0 + exhaust_pct)
        exhausted_below = close < ema200 * (1.0 - exhaust_pct)
        adx_strong   = float(mi_cfg.get('adx_strong',   28))
        adx_high_vol = float(s_cfg.get('adx_high_vol_threshold', 42))

        narrative.regime = self._classify_regime(
            adx            = adx,
            adx_strong     = adx_strong,
            adx_high_vol   = adx_high_vol,
            close          = close,
            ema50          = ema50,
            ema200         = ema200,
            exhausted_above= exhausted_above,
            exhausted_below= exhausted_below,
            reversal_det   = narrative.reversal_detected,
            reversal_dir   = narrative.reversal_direction,
            rsi_bull_div   = rsi_bull,
            rsi_bear_div   = rsi_bear,
            liq_bull       = liq_bull,
            liq_bear       = liq_bear,
            vol_climax     = vol_climax,
            vol_expand     = vol_expand,
        )

        # ── 9. Direction Blocks ───────────────────────────────────────────────
        self._apply_blocks(narrative, htf_bias, htf_strength, config)

        # ── 10. Setup Quality & Uncertainty ──────────────────────────────────
        narrative.setup_quality, narrative.uncertainty = self._compute_quality(
            narrative, adx, rsi, htf_strength
        )

        # ── 11. Confidence Adjustment ─────────────────────────────────────────
        narrative.confidence_adjustment = self._compute_confidence_adj(narrative)

        # ── 12. Narrative Text ────────────────────────────────────────────────
        narrative.signals_active = signals
        narrative.narrative = self._build_narrative(narrative, adx, rsi, close, ema200)

        if signals:
            logger.debug(
                f"MarketIntelligence | regime={narrative.regime} signals={signals} "
                f"quality={narrative.setup_quality:.2f} "
                f"block_buy={narrative.block_buy} block_sell={narrative.block_sell}"
            )

        return narrative

    # ── RSI Divergence ────────────────────────────────────────────────────────

    def _detect_rsi_divergence(
        self, df: pd.DataFrame, lookback: int, strength: int
    ) -> Tuple[bool, bool]:
        """Bullish: price lower low + RSI higher low. Bearish: price higher high + RSI lower high."""
        if 'rsi' not in df.columns or len(df) < lookback + strength * 2 + 2:
            return False, False

        w     = df.tail(lookback)
        lows  = w['low'].values
        highs = w['high'].values
        rsi   = w['rsi'].values

        # Bullish divergence
        price_lows = _find_swing_lows(lows, strength)
        rsi_lows   = _find_swing_lows(rsi,  strength)
        bull_div = self._check_divergence(price_lows, rsi_lows, strength, want_lower_price=True)

        # Bearish divergence
        price_highs = _find_swing_highs(highs, strength)
        rsi_highs   = _find_swing_highs(rsi,   strength)
        bear_div = self._check_divergence(price_highs, rsi_highs, strength, want_lower_price=False)

        return bull_div, bear_div

    # ── MACD Divergence ───────────────────────────────────────────────────────

    def _detect_macd_divergence(
        self, df: pd.DataFrame, lookback: int, strength: int
    ) -> Tuple[bool, bool]:
        """Divergence between price swings and MACD histogram swings."""
        if 'macd_hist' not in df.columns or len(df) < lookback + strength * 2 + 2:
            return False, False

        w     = df.tail(lookback)
        lows  = w['low'].values
        highs = w['high'].values
        mhist = w['macd_hist'].values

        price_lows  = _find_swing_lows(lows, strength)
        macd_lows   = _find_swing_lows(mhist, strength)
        bull_div = self._check_divergence(price_lows, macd_lows, strength, want_lower_price=True)

        price_highs = _find_swing_highs(highs, strength)
        macd_highs  = _find_swing_highs(mhist,  strength)
        bear_div = self._check_divergence(price_highs, macd_highs, strength, want_lower_price=False)

        return bull_div, bear_div

    @staticmethod
    def _check_divergence(
        price_swings: List[Tuple[int, float]],
        indicator_swings: List[Tuple[int, float]],
        strength: int,
        want_lower_price: bool,
    ) -> bool:
        """
        Core divergence logic.
        want_lower_price=True  → look for bullish div (price lower, indicator higher).
        want_lower_price=False → look for bearish div (price higher, indicator lower).
        """
        if len(price_swings) < 2 or len(indicator_swings) < 2:
            return False

        p1_idx, p1_val = price_swings[-2]
        p2_idx, p2_val = price_swings[-1]
        tol = strength * 2

        # Find nearest indicator swings to each price swing
        near1 = [x for x in indicator_swings if abs(x[0] - p1_idx) <= tol]
        near2 = [x for x in indicator_swings if abs(x[0] - p2_idx) <= tol]
        if not near1 or not near2:
            return False

        i1_val = min(near1, key=lambda x: abs(x[0] - p1_idx))[1]
        i2_val = min(near2, key=lambda x: abs(x[0] - p2_idx))[1]

        if want_lower_price:
            # Bullish: price made lower low, indicator made higher low
            return p2_val < p1_val and i2_val > i1_val
        else:
            # Bearish: price made higher high, indicator made lower high
            return p2_val > p1_val and i2_val < i1_val

    # ── Displacement Candles ──────────────────────────────────────────────────

    def _detect_displacement(
        self, df: pd.DataFrame, atr_mult: float = 2.0, lookback: int = 3
    ) -> Tuple[bool, bool]:
        """
        Displacement: candle body >= atr_mult × ATR.
        Checks the last `lookback` closed bars.
        """
        if 'atr' not in df.columns or len(df) < lookback + 2:
            return False, False

        bull = bear = False
        for i in range(-lookback - 1, -1):
            bar  = df.iloc[i]
            o    = float(bar['open'])
            c    = float(bar['close'])
            _atr = float(bar.get('atr', 1.0))
            body = abs(c - o)
            if body >= atr_mult * _atr:
                if c > o:
                    bull = True
                else:
                    bear = True

        return bull, bear

    # ── Liquidity Sweep ───────────────────────────────────────────────────────

    def _detect_liquidity_sweep(
        self, df: pd.DataFrame, lookback: int = 25
    ) -> Tuple[bool, bool]:
        """
        Bullish sweep: wick pierces below recent swing low then closes back above.
        Bearish sweep: wick pierces above recent swing high then closes back below.
        """
        if len(df) < lookback + 5:
            return False, False

        # Reference range: exclude last 3 bars (the sweep candles themselves)
        history = df.iloc[-(lookback + 3):-3]
        if len(history) < 5:
            return False, False

        recent_sh = float(history['high'].max())
        recent_sl = float(history['low'].min())

        bull_sweep = bear_sweep = False
        for i in range(-3, 0):
            bar = df.iloc[i]
            o   = float(bar['open'])
            c   = float(bar['close'])
            h   = float(bar['high'])
            l   = float(bar['low'])

            candle_range = h - l
            if candle_range < 1e-8:
                continue

            body        = abs(c - o)
            lower_wick  = min(o, c) - l
            upper_wick  = h - max(o, c)

            # Bullish sweep: low dipped below recent swing low, closed bullish,
            # lower wick dominates (at least 1.5× body), closed above midpoint
            if (l < recent_sl
                    and c > o
                    and lower_wick > max(body * 1.5, candle_range * 0.3)
                    and c > (h + l) / 2):
                bull_sweep = True

            # Bearish sweep: high pierced above recent swing high, closed bearish,
            # upper wick dominates, closed below midpoint
            if (h > recent_sh
                    and c < o
                    and upper_wick > max(body * 1.5, candle_range * 0.3)
                    and c < (h + l) / 2):
                bear_sweep = True

        return bull_sweep, bear_sweep

    # ── BOS / CHOCH ───────────────────────────────────────────────────────────

    def _detect_bos_choch(
        self, df: pd.DataFrame, lookback: int = 25, strength: int = 3
    ) -> Dict[str, bool]:
        """
        BOS  (Break of Structure): close breaks the most recent significant swing.
        CHOCH (Change of Character): BOS in the opposite direction of prior structure.
        """
        result = {k: False for k in ['bos_bull', 'bos_bear', 'choch_bull', 'choch_bear']}
        min_bars = lookback + strength * 2 + 3
        if len(df) < min_bars:
            return result

        # Structure reference: exclude last 2 bars
        history = df.iloc[-(lookback + strength * 2):-2]
        if len(history) < strength * 2 + 2:
            return result

        current_close = float(df.iloc[-1]['close'])
        highs = history['high'].values
        lows  = history['low'].values

        sh_list = _find_swing_highs(highs, strength)
        sl_list = _find_swing_lows(lows,  strength)
        if not sh_list or not sl_list:
            return result

        last_sh = sh_list[-1][1]
        last_sl = sl_list[-1][1]

        # BOS
        if current_close > last_sh:
            result['bos_bull'] = True
        if current_close < last_sl:
            result['bos_bear'] = True

        # CHOCH: prior structure was trending in the opposite direction
        if len(sh_list) >= 2:
            prev_sh = sh_list[-2][1]
            # Downtrend structure (lower highs) then price breaks above last SH
            if prev_sh > last_sh and current_close > last_sh:
                result['choch_bull'] = True

        if len(sl_list) >= 2:
            prev_sl = sl_list[-2][1]
            # Uptrend structure (higher lows) then price breaks below last SL
            if prev_sl < last_sl and current_close < last_sl:
                result['choch_bear'] = True

        return result

    # ── Volatility ────────────────────────────────────────────────────────────

    def _detect_volatility(self, df: pd.DataFrame) -> Tuple[bool, bool]:
        """
        Climax:    current ATR > rolling_mean + 2×rolling_std  (50-bar window).
        Expansion: current ATR > ATR 5 bars ago × 1.30.
        """
        if 'atr' not in df.columns or len(df) < 55:
            return False, False

        atrs        = df['atr'].values
        current_atr = atrs[-1]
        hist        = atrs[-52:-2]          # 50-bar window, avoid last bar

        mean_atr = float(np.mean(hist))
        std_atr  = float(np.std(hist))

        climax    = bool(current_atr > mean_atr + 2.0 * std_atr)
        expansion = bool(current_atr > atrs[-6] * 1.30) if len(atrs) >= 6 else False

        return climax, expansion

    # ── Extended Regime Classification ────────────────────────────────────────

    def _classify_regime(
        self, *,
        adx: float, adx_strong: float, adx_high_vol: float,
        close: float, ema50: float, ema200: float,
        exhausted_above: bool, exhausted_below: bool,
        reversal_det: bool, reversal_dir: str,
        rsi_bull_div: bool, rsi_bear_div: bool,
        liq_bull: bool, liq_bear: bool,
        vol_climax: bool, vol_expand: bool,
    ) -> str:
        # Priority order (highest → lowest)
        if vol_climax:
            return REGIME_HIGH_VOL
        if liq_bull or liq_bear:
            return REGIME_LIQUIDITY_GRAB
        if reversal_det:
            return REGIME_REVERSAL
        if exhausted_above or exhausted_below:
            return REGIME_EXHAUSTION
        if adx >= adx_high_vol:
            return REGIME_HIGH_VOL
        if adx >= adx_strong:
            if close > ema50 > ema200:
                return REGIME_TREND_BULL
            if close < ema50 < ema200:
                return REGIME_TREND_BEAR
            return REGIME_EXPANSION
        # Low ADX
        if rsi_bull_div:
            return REGIME_ACCUMULATION
        if rsi_bear_div:
            return REGIME_DISTRIBUTION
        if vol_expand:
            return REGIME_EXPANSION
        return REGIME_RANGE

    # ── Direction Blocks ──────────────────────────────────────────────────────

    def _apply_blocks(
        self,
        n: MarketNarrative,
        htf_bias: str,
        htf_strength: float,
        config: dict,
    ) -> None:
        mi_cfg = config.get('market_intelligence', {})
        if not mi_cfg.get('enabled', True):
            return

        reasons: List[str] = []

        # Volatility climax: block ALL new entries
        if n.volatility_climax:
            n.block_buy  = True
            n.block_sell = True
            n.block_reason = 'Volatility climax — no new entries during ATR spike'
            return

        # Reversal UP: don't SELL into a developing bullish reversal
        if n.reversal_detected and n.reversal_direction == 'UP':
            n.block_sell = True
            reasons.append('Reversal UP (div+BOS/CHOCH+sweep) — blocking SELL')

        # Reversal DOWN: don't BUY into a developing bearish reversal
        if n.reversal_detected and n.reversal_direction == 'DOWN':
            n.block_buy = True
            reasons.append('Reversal DOWN (div+BOS/CHOCH+sweep) — blocking BUY')

        # Dual bearish divergence: don't add BUY exposure
        if n.rsi_divergence_bear and n.macd_divergence_bear:
            n.block_buy = True
            reasons.append('Dual bearish divergence (RSI+MACD) — blocking BUY')

        # Dual bullish divergence: don't add SELL exposure
        if n.rsi_divergence_bull and n.macd_divergence_bull:
            n.block_sell = True
            reasons.append('Dual bullish divergence (RSI+MACD) — blocking SELL')

        # Bearish displacement against strong HTF BUY — momentum shift
        if n.displacement_bear and htf_bias == 'BUY' and htf_strength > 0.7:
            n.block_buy = True
            reasons.append('Bearish displacement candle while HTF=BUY(strong) — momentum warning')

        # Bullish displacement against strong HTF SELL
        if n.displacement_bull and htf_bias == 'SELL' and htf_strength > 0.7:
            n.block_sell = True
            reasons.append('Bullish displacement candle while HTF=SELL(strong) — momentum warning')

        # CHOCH bearish: character changed to bearish — block BUY
        if n.choch_bear and not n.block_buy:
            n.block_buy = True
            reasons.append('Bearish CHoCH (market character flipped bearish)')

        # CHOCH bullish: character changed to bullish — block SELL
        if n.choch_bull and not n.block_sell:
            n.block_sell = True
            reasons.append('Bullish CHoCH (market character flipped bullish)')

        # Liquidity sweep: favour the reversal direction
        if n.liquidity_sweep_bull and not n.block_sell:
            n.block_sell = True
            reasons.append('Bullish liquidity sweep — stop hunt below, expect bounce UP')
        if n.liquidity_sweep_bear and not n.block_buy:
            n.block_buy = True
            reasons.append('Bearish liquidity sweep — stop hunt above, expect dump DOWN')

        if reasons:
            n.block_reason = ' | '.join(reasons)

    # ── Setup Quality & Uncertainty ───────────────────────────────────────────

    def _compute_quality(
        self,
        n: MarketNarrative,
        adx: float,
        rsi: float,
        htf_strength: float,
    ) -> Tuple[float, float]:
        quality     = 0.50
        uncertainty = 0.30

        # Quality boosters
        if n.regime in (REGIME_TREND_BULL, REGIME_TREND_BEAR):
            quality += 0.15
        elif n.regime in (REGIME_EXHAUSTION, REGIME_REVERSAL):
            quality -= 0.10

        if n.bos_bull or n.bos_bear:
            quality += 0.10
        if n.displacement_bull or n.displacement_bear:
            quality += 0.08

        # HTF alignment bonus (proportional)
        quality += htf_strength * 0.15

        # ADX strength → more predictable direction
        quality += min(adx / 50.0, 1.0) * 0.10

        # Uncertainty drivers
        if n.volatility_climax:
            uncertainty += 0.30
        if n.reversal_detected:
            uncertainty += 0.15

        # Conflicting bull+bear signals → uncertain
        n_bull = sum([n.rsi_divergence_bull, n.macd_divergence_bull,
                      n.liquidity_sweep_bull, n.bos_bull, n.choch_bull])
        n_bear = sum([n.rsi_divergence_bear, n.macd_divergence_bear,
                      n.liquidity_sweep_bear, n.bos_bear, n.choch_bear])
        if n_bull > 0 and n_bear > 0:
            uncertainty += 0.15

        # RSI extreme → uncertainty for trend continuation
        if rsi > 75 or rsi < 25:
            uncertainty += 0.10

        return float(np.clip(quality, 0.0, 1.0)), float(np.clip(uncertainty, 0.0, 1.0))

    # ── Confidence Adjustment ─────────────────────────────────────────────────

    def _compute_confidence_adj(self, n: MarketNarrative) -> float:
        adj = 0.0

        # Boosters
        if n.bos_bull or n.bos_bear:
            adj += 0.05
        if n.displacement_bull or n.displacement_bear:
            adj += 0.05
        if n.regime in (REGIME_TREND_BULL, REGIME_TREND_BEAR):
            adj += 0.03

        # Reducers
        if n.volatility_climax:
            adj -= 0.12
        if n.reversal_detected:
            adj -= 0.05
        if n.uncertainty > 0.60:
            adj -= 0.05

        return float(np.clip(adj, -0.15, 0.15))

    # ── Narrative Text ────────────────────────────────────────────────────────

    def _build_narrative(
        self,
        n: MarketNarrative,
        adx: float,
        rsi: float,
        close: float,
        ema200: float,
    ) -> str:
        ema_dist_pct = abs(close - ema200) / max(ema200, 1.0) * 100

        desc = {
            REGIME_TREND_BULL:     f'Bullish trend (ADX={adx:.0f})',
            REGIME_TREND_BEAR:     f'Bearish trend (ADX={adx:.0f})',
            REGIME_RANGE:          f'Range (ADX={adx:.0f}, low momentum)',
            REGIME_EXPANSION:      f'Volatility expansion (ADX={adx:.0f})',
            REGIME_REVERSAL:       f'Potential reversal {n.reversal_direction}',
            REGIME_ACCUMULATION:   'Accumulation (bullish divergence detected)',
            REGIME_DISTRIBUTION:   'Distribution (bearish divergence detected)',
            REGIME_LIQUIDITY_GRAB: 'Liquidity grab / stop hunt',
            REGIME_EXHAUSTION:     f'Trend exhaustion ({ema_dist_pct:.1f}% from EMA200)',
            REGIME_HIGH_VOL:       'Extreme volatility — elevated risk',
        }.get(n.regime, n.regime)

        parts = [desc]
        if n.signals_active:
            parts.append(f"[{', '.join(n.signals_active)}]")
        if n.block_buy and n.block_sell:
            parts.append('ALL ENTRIES BLOCKED')
        elif n.block_buy:
            parts.append('BUY blocked')
        elif n.block_sell:
            parts.append('SELL blocked')
        parts.append(f'RSI={rsi:.0f} quality={n.setup_quality:.2f}')
        return ' | '.join(parts)
