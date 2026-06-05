"""
execution_mt5.py — All MetaTrader 5 interactions.

Key improvements over naive initialize():
  1. _wait_terminal_connected() — polls terminal_info().connected until True
     before making any API call (fixes "Terminal: Call failed")
  2. _find_real_symbol()        — searches broker's symbol list for the
     correct name (handles suffixes like 'm', '.', '+', 'x', etc.)
  3. _symbol_map                — stores config-name -> broker-name mapping
     so every method is transparent to the caller
"""

import logging
import math
import time
from typing import Dict, List, Optional

import MetaTrader5 as mt5

logger = logging.getLogger('AI-Trade')


class MT5Executor:
    def __init__(self, config: dict):
        self._cfg        = config
        self._exec_cfg   = config['execution']
        self.connected   = False
        self._symbol_map: Dict[str, str] = {}   # config name -> real broker name

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _actual(self, symbol: str) -> str:
        """Translate config symbol name to the real broker name."""
        return self._symbol_map.get(symbol, symbol)

    def actual_symbol(self, symbol: str) -> str:
        """Public wrapper for code that needs to compare MT5 position symbols."""
        return self._actual(symbol)

    def canonical_symbol(self, broker_symbol: str) -> str:
        """Translate a broker symbol back to the configured symbol when possible."""
        broker_upper = (broker_symbol or '').upper()
        for configured, actual in self._symbol_map.items():
            if broker_upper == actual.upper():
                return configured
        for configured, actual in self._symbol_map.items():
            cfg_upper = configured.upper()
            act_upper = actual.upper()
            if broker_upper.startswith(cfg_upper) or cfg_upper in broker_upper:
                return configured
            if broker_upper == act_upper:
                return configured
        return broker_symbol

    def _wait_terminal_connected(self, timeout: int = 30) -> bool:
        """
        Block until terminal_info().connected is True.
        Returns False if it never becomes True within `timeout` seconds.

        This is required because mt5.initialize() succeeds as soon as the
        terminal process is found, but the terminal may still be logging in
        to the broker.  All data-feed calls fail with 'Terminal: Call failed'
        until the broker handshake completes.
        """
        logger.info("Waiting for MT5 terminal to connect to broker...")
        for elapsed in range(timeout):
            info = mt5.terminal_info()
            if info and info.connected:
                logger.info(f"Terminal connected to broker (took {elapsed+1}s)")
                return True
            time.sleep(1)
        logger.error(f"Terminal did not connect within {timeout}s")
        return False

    def _find_real_symbol(self, base: str) -> str:
        """
        Find the symbol name actually available on this broker.

        Brokers append suffixes to symbol names (e.g. XAUUSDm, XAUUSD.,
        EURUSD+, EURUSDpro).  We try:
          1. Exact match
          2. Starts-with match  (e.g. 'XAUUSD' matches 'XAUUSDm')
          3. Contains match     (e.g. 'XAUUSD' matches 'fxXAUUSD')
        """
        # Exact
        if mt5.symbol_info(base) is not None:
            return base

        all_symbols = mt5.symbols_get()
        if not all_symbols:
            return base

        upper = base.upper()

        # Starts-with (most common case — broker appends suffix)
        for s in all_symbols:
            if s.name.upper().startswith(upper):
                logger.info(f"Symbol resolved: {base} -> {s.name}")
                return s.name

        # Contains
        for s in all_symbols:
            if upper in s.name.upper():
                logger.info(f"Symbol resolved: {base} -> {s.name}")
                return s.name

        logger.warning(f"Symbol '{base}' not found on this broker — using as-is")
        return base

    # ── Connection ────────────────────────────────────────────────────────────

    def connect(self) -> bool:
        """
        Connect to the MT5 terminal that is already running and logged in.
        Waits until the terminal is fully connected to the broker before
        subscribing symbols.
        """
        if not mt5.initialize():
            logger.error(
                f"MT5 initialize() failed: {mt5.last_error()} "
                "— make sure MetaTrader 5 is open and logged in."
            )
            return False

        # Wait for broker connection (the critical step)
        if not self._wait_terminal_connected(timeout=30):
            mt5.shutdown()
            return False

        # Resolve + subscribe all configured symbols
        for sym in self._cfg['trading']['symbols']:
            real = self._find_real_symbol(sym)
            self._symbol_map[sym] = real
            if mt5.symbol_select(real, True):
                logger.info(f"Subscribed: {sym} -> '{real}'")
            else:
                logger.warning(
                    f"symbol_select('{real}') failed: {mt5.last_error()}"
                )

        info = mt5.account_info()
        if info is None:
            logger.error(f"account_info() failed: {mt5.last_error()}")
            mt5.shutdown()
            return False

        self.connected = True
        logger.info(
            f"MT5 ready | account={info.login} | "
            f"balance={info.balance:.2f} {info.currency} | "
            f"server={info.server} | leverage=1:{info.leverage}"
        )
        logger.info(f"Symbol map: {self._symbol_map}")
        return True

    def disconnect(self) -> None:
        mt5.shutdown()
        self.connected = False
        logger.info("MT5 disconnected")

    # ── Account & symbol info ─────────────────────────────────────────────────

    def get_account_info(self):
        info = mt5.account_info()
        if info is None:
            logger.warning(f"account_info failed: {mt5.last_error()}")
        return info

    def get_tick(self, symbol: str):
        """Return the latest bid/ask tick for a symbol (using resolved broker name)."""
        return mt5.symbol_info_tick(self._actual(symbol))

    def get_symbol_info(self, symbol: str):
        real = self._actual(symbol)
        info = mt5.symbol_info(real)
        if info is None:
            logger.error(f"symbol_info('{real}') failed: {mt5.last_error()}")
            return None
        if not info.visible:
            if not mt5.symbol_select(real, True):
                logger.error(f"Cannot select '{real}' in MarketWatch")
                return None
            info = mt5.symbol_info(real)
        return info

    # ── OHLCV data ────────────────────────────────────────────────────────────

    _TF_MAP = {
        'M1':  mt5.TIMEFRAME_M1,  'M5':  mt5.TIMEFRAME_M5,
        'M15': mt5.TIMEFRAME_M15, 'M30': mt5.TIMEFRAME_M30,
        'H1':  mt5.TIMEFRAME_H1,  'H4':  mt5.TIMEFRAME_H4,
        'D1':  mt5.TIMEFRAME_D1,
    }

    def get_ohlcv(self, symbol: str, timeframe_str: str, bars: int = 600):
        real = self._actual(symbol)
        tf   = self._TF_MAP.get(timeframe_str.upper(), mt5.TIMEFRAME_M15)

        mt5.symbol_select(real, True)

        for attempt in range(1, 4):
            rates = mt5.copy_rates_from_pos(real, tf, 0, bars)
            if rates is not None and len(rates) > 0:
                return rates
            logger.warning(
                f"get_ohlcv('{real}') attempt {attempt}/3 failed: "
                f"{mt5.last_error()} — retrying in 3s"
            )
            time.sleep(3)

        logger.error(f"get_ohlcv('{real}'): all 3 attempts failed.")
        return None

    # ── Position queries ──────────────────────────────────────────────────────

    def get_all_open_positions(self) -> List:
        positions = mt5.positions_get()
        return list(positions) if positions else []

    def get_positions_for_symbol(self, symbol: str, magic: int) -> List:
        real      = self._actual(symbol)
        positions = mt5.positions_get(symbol=real)
        if not positions:
            return []
        return [p for p in positions if p.magic == magic]

    def get_magic_positions(self, magic: int) -> List:
        """Return only positions opened by this strategy magic number."""
        return [p for p in self.get_all_open_positions() if p.magic == magic]

    def symbol_has_position(self, symbol: str, magic: int) -> bool:
        return len(self.get_positions_for_symbol(symbol, magic)) > 0

    # ── Filling mode detection ────────────────────────────────────────────────

    def _resolve_filling(self, symbol: str) -> int:
        real = self._actual(symbol)
        info = mt5.symbol_info(real)
        if info is None:
            return getattr(mt5, 'ORDER_FILLING_IOC', 1)

        mode = info.filling_mode

        # MT5 Python binding attribute names vary by package version — use
        # getattr with the known integer fallbacks (spec: FOK=1, IOC=2).
        SYM_FOK = getattr(mt5, 'SYMBOL_FILLING_FOK',    1)
        SYM_IOC = getattr(mt5, 'SYMBOL_FILLING_IOC',    2)
        ORD_FOK    = getattr(mt5, 'ORDER_FILLING_FOK',    0)
        ORD_IOC    = getattr(mt5, 'ORDER_FILLING_IOC',    1)
        ORD_RETURN = getattr(mt5, 'ORDER_FILLING_RETURN', 2)

        if mode & SYM_IOC:
            return ORD_IOC
        if mode & SYM_FOK:
            return ORD_FOK
        return ORD_RETURN

    # ── Order placement ───────────────────────────────────────────────────────

    @staticmethod
    def _normalize_volume(volume: float, sym_info) -> float:
        """Round volume down to broker step and reject anything below minimum."""
        try:
            step = float(sym_info.volume_step)
            vol_min = float(sym_info.volume_min)
            vol_max = float(sym_info.volume_max)
            requested = float(volume)
        except (AttributeError, TypeError, ValueError):
            return 0.0
        if step <= 0 or vol_min <= 0 or vol_max < vol_min or requested <= 0:
            return 0.0
        normalized = math.floor((min(requested, vol_max) + 1e-12) / step) * step
        if normalized + 1e-12 < vol_min:
            return 0.0
        return round(normalized, 8)

    @staticmethod
    def _valid_protective_prices(
        is_buy: bool,
        price: float,
        sl_price: float,
        tp_price: float,
        sym_info,
    ) -> bool:
        """Fail closed when SL/TP are missing, inverted, or inside broker stops level."""
        values = (price, sl_price, tp_price)
        if any(not math.isfinite(float(v)) or float(v) <= 0 for v in values):
            return False
        if is_buy and not (sl_price < price < tp_price):
            return False
        if not is_buy and not (tp_price < price < sl_price):
            return False

        point = float(getattr(sym_info, 'point', 0.0) or 0.0)
        stops_level = float(getattr(sym_info, 'trade_stops_level', 0.0) or 0.0)
        min_distance = point * stops_level
        if min_distance > 0:
            if abs(price - sl_price) + 1e-12 < min_distance:
                return False
            if abs(tp_price - price) + 1e-12 < min_distance:
                return False
        return True

    def place_market_order(
        self,
        symbol: str,
        direction: str,
        lot_size: float,
        sl_price: float,
        tp_price: float,
        magic: int,
        comment: str = '',
    ) -> Optional[Dict]:
        real = self._actual(symbol)
        direction = direction.upper()
        if direction not in ('BUY', 'SELL'):
            logger.error(f"{real}: invalid order direction '{direction}'")
            return None
        is_buy     = direction == 'BUY'
        order_type = mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL
        filling    = self._resolve_filling(symbol)
        sym_info   = mt5.symbol_info(real)
        if sym_info is None:
            logger.error(f"{real}: symbol_info unavailable before order")
            return None
        lot_size = self._normalize_volume(lot_size, sym_info)
        if lot_size <= 0:
            logger.error(f"{real}: invalid or below-minimum order volume")
            return None
        digits     = sym_info.digits
        deviation  = self._exec_cfg['deviation']

        def _price() -> float:
            tick = mt5.symbol_info_tick(real)
            return (tick.ask if is_buy else tick.bid) if tick else 0.0

        for attempt in range(1, self._exec_cfg['max_retries'] + 1):
            price = _price()
            if price == 0.0:
                logger.error(f"{real}: cannot get tick (attempt {attempt})")
                time.sleep(self._exec_cfg['retry_delay'])
                continue
            if not self._valid_protective_prices(
                is_buy, price, sl_price, tp_price, sym_info
            ):
                logger.error(
                    f"{real}: rejected unsafe protective prices | "
                    f"direction={direction} price={price} sl={sl_price} tp={tp_price}"
                )
                return None

            request = {
                'action':       mt5.TRADE_ACTION_DEAL,
                'symbol':       real,
                'volume':       float(lot_size),
                'type':         order_type,
                'price':        price,
                'sl':           round(sl_price, digits),
                'tp':           round(tp_price, digits),
                'deviation':    deviation,
                'magic':        magic,
                'comment':      comment[:31],
                'type_time':    mt5.ORDER_TIME_GTC,
                'type_filling': filling,
            }

            check = mt5.order_check(request)
            accepted_checks = {
                0,
                getattr(mt5, 'TRADE_RETCODE_DONE', 10009),
                getattr(mt5, 'TRADE_RETCODE_PLACED', 10008),
            }
            if check is None or getattr(check, 'retcode', -1) not in accepted_checks:
                logger.error(
                    f"{real}: order_check rejected request | "
                    f"retcode={getattr(check, 'retcode', 'None')} "
                    f"comment='{getattr(check, 'comment', mt5.last_error())}'"
                )
                return None

            result = mt5.order_send(request)

            if result is None:
                logger.warning(f"{real}: order_send None (attempt {attempt}): {mt5.last_error()}")
            elif result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(
                    f"ORDER PLACED | {direction} {lot_size} {real} "
                    f"@ {result.price:.5f} | SL={sl_price:.5f} TP={tp_price:.5f} "
                    f"| ticket={result.order}"
                )
                return {
                    'ticket':   result.order,
                    'price':    result.price,
                    'lot_size': lot_size,
                    'sl_price': sl_price,
                    'tp_price': tp_price,
                }
            elif result.retcode in (
                mt5.TRADE_RETCODE_REQUOTE,
                mt5.TRADE_RETCODE_PRICE_OFF,
                mt5.TRADE_RETCODE_PRICE_CHANGED,
            ):
                logger.warning(f"{real}: requote (attempt {attempt}), retrying...")
            else:
                logger.error(
                    f"{real}: order failed | retcode={result.retcode} "
                    f"comment='{result.comment}' (attempt {attempt})"
                )
                break

            if attempt < self._exec_cfg['max_retries']:
                time.sleep(self._exec_cfg['retry_delay'])

        logger.error(f"{real}: failed to place {direction} after {self._exec_cfg['max_retries']} attempts")
        return None

    # ── Position closure ──────────────────────────────────────────────────────

    def close_position(self, position) -> bool:
        is_long    = position.type == mt5.ORDER_TYPE_BUY
        close_type = mt5.ORDER_TYPE_SELL if is_long else mt5.ORDER_TYPE_BUY
        filling    = self._resolve_filling(position.symbol)
        tick       = mt5.symbol_info_tick(position.symbol)
        if tick is None:
            logger.error(f"No tick for {position.symbol}")
            return False

        result = mt5.order_send({
            'action':       mt5.TRADE_ACTION_DEAL,
            'symbol':       position.symbol,
            'volume':       position.volume,
            'type':         close_type,
            'position':     position.ticket,
            'price':        tick.bid if is_long else tick.ask,
            'deviation':    self._exec_cfg['deviation'],
            'magic':        position.magic,
            'comment':      'close',
            'type_time':    mt5.ORDER_TIME_GTC,
            'type_filling': filling,
        })

        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"CLOSED ticket={position.ticket} @ {result.price:.5f}")
            return True

        logger.error(
            f"Failed to close ticket={position.ticket}: "
            f"{mt5.last_error() if result is None else result.comment}"
        )
        return False

    def close_by_ticket(self, ticket: int) -> bool:
        """Close a position identified by ticket number only."""
        pos_list = mt5.positions_get(ticket=ticket)
        if not pos_list:
            logger.warning(f"close_by_ticket: ticket {ticket} not found")
            return False
        return self.close_position(pos_list[0])

    def partial_close(self, ticket: int, volume: float) -> bool:
        """Partially close `volume` lots of an open position."""
        pos_list = mt5.positions_get(ticket=ticket)
        if not pos_list:
            logger.warning(f"partial_close: ticket {ticket} not found")
            return False
        pos     = pos_list[0]
        is_long = pos.type == mt5.ORDER_TYPE_BUY
        tick    = mt5.symbol_info_tick(pos.symbol)
        if tick is None:
            logger.error(f"No tick for {pos.symbol}")
            return False

        vol = min(round(volume, 2), round(pos.volume, 2))
        if vol <= 0:
            return False

        result = mt5.order_send({
            'action':       mt5.TRADE_ACTION_DEAL,
            'symbol':       pos.symbol,
            'volume':       vol,
            'type':         mt5.ORDER_TYPE_SELL if is_long else mt5.ORDER_TYPE_BUY,
            'position':     ticket,
            'price':        tick.bid if is_long else tick.ask,
            'deviation':    self._exec_cfg['deviation'],
            'magic':        pos.magic,
            'comment':      'partial_close',
            'type_time':    mt5.ORDER_TIME_GTC,
            'type_filling': self._resolve_filling(pos.symbol),
        })

        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(
                f"PARTIAL CLOSE ticket={ticket} vol={vol:.2f} @ {result.price:.5f}"
            )
            return True

        logger.error(
            f"partial_close failed ticket={ticket}: "
            f"{mt5.last_error() if result is None else result.comment}"
        )
        return False

    def modify_sl(self, ticket: int, new_sl: float) -> bool:
        """Move stop-loss for an open position."""
        pos_list = mt5.positions_get(ticket=ticket)
        if not pos_list:
            logger.warning(f"modify_sl: ticket {ticket} not found")
            return False
        pos = pos_list[0]

        result = mt5.order_send({
            'action':   mt5.TRADE_ACTION_SLTP,
            'position': ticket,
            'symbol':   pos.symbol,
            'sl':       round(new_sl, 5),
            'tp':       pos.tp,
        })

        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"SL modified ticket={ticket} → {new_sl:.5f}")
            return True

        logger.error(
            f"modify_sl failed ticket={ticket}: "
            f"{mt5.last_error() if result is None else result.comment}"
        )
        return False

    # ── Deal history ──────────────────────────────────────────────────────────

    def get_closed_deals(self, from_ts: float, to_ts: float, magic: int) -> List:
        deals = mt5.history_deals_get(from_ts, to_ts)
        if not deals:
            return []
        return [d for d in deals if d.magic == magic and d.entry == mt5.DEAL_ENTRY_OUT]
