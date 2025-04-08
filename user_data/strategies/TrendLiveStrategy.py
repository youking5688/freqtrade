# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Any, Optional, Union
from collections.abc import Callable
from collections.abc import Mapping, Sequence
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation

from freqtrade.strategy import (
    IStrategy,
    informative,  # @informative 装饰器
    # Hyperopt 参数
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
    # 时间框架帮助函数
    timeframe_to_minutes,
    timeframe_to_next_date,
    timeframe_to_prev_date,
    # 策略帮助函数
    merge_informative_pair,
    stoploss_from_absolute,
    stoploss_from_open,
)

# 导入技术分析库
import talib.abstract as ta
import pandas_ta as pta
from numpy.lib import math
from technical import qtpylib
import datetime as _datetime
from freqtrade.persistence import Order, Trade
from freqtrade.exchange import Exchange
import time
import uuid  # 导入 uuid 模块以生成唯一 ID

import logging

logger = logging.getLogger(__name__)


class TrendLiveStrategy(IStrategy):
    # 在类定义中声明这个属性
    exchange: Exchange
    # 简化ROI和止损设置
    minimal_roi = {
        "0": 0.115,  # 在交易开始后的0分钟内，如果盈利达到5%，可以卖出
        "25": 0.084,  # 30分钟后，如果盈利达到3%，可以卖出
        "73": 0.028,  # 60分钟后，如果盈利达到1%，可以卖出
        "188": 0.0,  # 120分钟后，如果盈利达到0.5%，可以卖出
    }

    timeframe = "5m"
    startup_candle_count = 200  # 覆盖 1h 数据的最大周期
    can_short = True  # 可以做空

    stoploss = -1  # 初始止损设置为 -10%

    trailing_stop = True
    trailing_stop_positive = 0.011  # 盈利 1% 后启动跟踪止损
    trailing_stop_positive_offset = 0.048  # 偏移5%
    trailing_only_offset_is_reached = True  # 只有在达到一定盈利后才开始跟踪止损

    # 送进ICU的订单亏损阈值
    icu_loss_threshold = -0.30
    icu_time_threshold = 480

    # 定义超参数优化空间
    buy_macd_1h_fastperiod = IntParameter(5, 50, default=9, space="buy")
    buy_macd_1h_slowperiod = IntParameter(15, 100, default=30, space="buy")
    buy_macd_1h_signalperiod = IntParameter(5, 35, default=27, space="buy")

    buy_macd_5m_fastperiod = IntParameter(5, 50, default=11, space="buy")
    buy_macd_5m_slowperiod = IntParameter(15, 100, default=81, space="buy")
    buy_macd_5m_signalperiod = IntParameter(5, 35, default=10, space="buy")

    buy_adx_period = IntParameter(10, 30, default=29, space="buy")
    buy_adx_threshold = IntParameter(20, 40, default=26, space="buy")

    buy_rsi_period = IntParameter(10, 30, default=16, space="buy")
    buy_rsi_threshold = IntParameter(20, 40, default=21, space="buy")

    sell_rsi_threshold = IntParameter(60, 90, default=62, space="sell")
    sell_adx_period = IntParameter(10, 30, default=15, space="sell")
    sell_adx_threshold = IntParameter(20, 40, default=21, space="sell")

    rsi_high_window = IntParameter(10, 30, default=14, space="buy, sell")
    rsi_low_window = IntParameter(10, 30, default=14, space="buy, sell")
    atr_period = IntParameter(5, 30, default=16, space="sell")
    atr_multiplier = DecimalParameter(1.0, 5.0, default=4.364, space="sell")
    trend_ewma_period = IntParameter(5, 30, default=20, space="buy, sell")

    buy_cooldown_period = IntParameter(5, 30, default=6)
    sell_cooldown_period = IntParameter(5, 30, default=6)

    @property
    def protections(self):
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": self.buy_cooldown_period.value,
                "side": "long",
            },
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": self.sell_cooldown_period.value,
                "side": "short",
            },
        ]

    # 动态杠杆
    use_custom_leverage = True

    def informative_pairs(self):
        # 获取白名单中的所有交易对。
        pairs = self.dp.current_whitelist()
        # 为每对交易对分配tf，以便可以为策略下载和缓存它们。
        informative_pairs = [(pair, "1h") for pair in pairs]
        # 可选的附加“静态”交易对
        informative_pairs += [
            ("BTC/USDT", "1h", "futures"),
        ]
        return informative_pairs

    # 检查订单盈利情况，将亏损到一定程度的订单送进“ICU”
    def bot_loop_start(self, **kwargs):
        # 获取所有活跃订单
        trades = Trade.get_open_trades()
        for trade in trades:
            # 获取当前价格并计算亏损率（所有订单都需要这一步）
            current_rate = self.get_current_price(trade.pair)
            if current_rate is None:
                logger.warning(f"无法获取交易对 {trade.pair} 的当前价格")
                continue

            profit_ratio = trade.calc_profit_ratio(current_rate)
            icu_tag = trade.get_custom_data("icu_tag")

            # 已标记订单的逻辑分支
            if icu_tag is not None:
                if profit_ratio >= self.icu_loss_threshold:
                    # 当亏损恢复时清除标记
                    trade.set_custom_data("icu_tag", None)
                    logger.info(
                        f"订单 {trade.id} 亏损恢复至{self.icu_loss_threshold}以内，已清除ICU标记"
                    )
                else:
                    # 保持标记并更新最新数据（可选）
                    icu_tag.update({"profit_ratio": profit_ratio, "current_rate": current_rate})
                    trade.set_custom_data("icu_tag", icu_tag)
                    logger.debug(f"订单 {trade.id} 持续亏损，更新ICU标记数据")

            # 未标记订单的逻辑分支
            else:
                if profit_ratio < self.icu_loss_threshold:
                    # 当新亏损达标时添加标记
                    icu_tag = {
                        "ICU": True,
                        "profit_ratio": profit_ratio,
                        "current_rate": current_rate,
                    }
                    trade.set_custom_data("icu_tag", icu_tag)
                    logger.warning(
                        f"订单 {trade.id} 亏损超过{self.icu_loss_threshold}，标记为手动处理"
                    )

    def get_current_price(self, pair: str) -> float | None:
        """兼容回测和实盘的价格获取方法"""
        try:
            # 方案1：实盘模式
            return self.exchange.fetch_ticker(pair)["last"]
        except AttributeError:
            # 方案2：回测模式
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            return dataframe["close"].iloc[-1] if not dataframe.empty else None

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if not self.dp:
            # Don't do anything if DataProvider is not available.
            return dataframe

        inf_tf = "1h"

        # Get the informative pair
        informative_1h = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe=inf_tf)
        # 确保 'close' 列存在且没有缺失值
        informative_1h["close"] = informative_1h["close"].ffill().bfill()

        # Calculate MACD
        macd, macdsignal, macdhist = ta.MACD(
            informative_1h["close"],  # 传入 'close' 列
            fastperiod=self.buy_macd_1h_fastperiod.value,
            slowperiod=self.buy_macd_1h_slowperiod.value,
            signalperiod=self.buy_macd_1h_signalperiod.value,
        )
        informative_1h["macd"] = macd  # MACD快线
        informative_1h["macdsignal"] = macdsignal  # MACD慢线
        informative_1h["macdhist"] = macdhist  # MACD柱
        dataframe = merge_informative_pair(
            dataframe, informative_1h, self.timeframe, inf_tf, ffill=True
        )

        macd, macdsignal, macdhist = ta.MACD(
            dataframe["close"],
            fastperiod=self.buy_macd_5m_fastperiod.value,
            slowperiod=self.buy_macd_5m_slowperiod.value,
            signalperiod=self.buy_macd_5m_signalperiod.value,
        )
        dataframe["macd"] = macd
        dataframe["macdsignal"] = macdsignal
        # ADX 作为趋势强度过滤
        dataframe["bullish_adx"] = ta.ADX(
            dataframe["high"],
            dataframe["low"],
            dataframe["close"],
            timeperiod=self.buy_adx_period.value,
        )
        dataframe["bearish_adx"] = ta.ADX(
            dataframe["high"],
            dataframe["low"],
            dataframe["close"],
            timeperiod=self.sell_adx_period.value,
        )

        # 添加动态止盈止损需要的指标
        dataframe["trend"] = (
            dataframe["close"].ewm(span=self.trend_ewma_period.value, adjust=False).mean()
        )
        # ATR用于衡量波动性
        dataframe["atr"] = ta.ATR(
            dataframe["high"],
            dataframe["low"],
            dataframe["close"],
            timeperiod=self.atr_period.value,
        )

        # 计算价格动量
        dataframe["momentum"] = dataframe["close"].pct_change(3)

        # 计算趋势强度
        dataframe["trend_strength"] = (
            abs(dataframe["close"] - dataframe["trend"]) / dataframe["trend"] * 100
        )  # noqa: E501

        # 引入RSI背离：检测价格与RSI的背离，提前识别潜在反转
        dataframe["rsi"] = ta.RSI(dataframe["close"], timeperiod=self.buy_rsi_period.value)
        dataframe["rsi_high"] = dataframe["rsi"].rolling(window=self.rsi_high_window.value).max()
        dataframe["rsi_low"] = dataframe["rsi"].rolling(window=self.rsi_low_window.value).min()
        dataframe["bullish_divergence"] = (dataframe["close"] < dataframe["close"].shift(1)) & (
            dataframe["rsi"] > dataframe["rsi"].shift(1)
        )
        dataframe["bearish_divergence"] = (dataframe["close"] > dataframe["close"].shift(1)) & (
            dataframe["rsi"] < dataframe["rsi"].shift(1)
        )

        # 填充 NaN 值
        dataframe.ffill()  # 前向填充
        dataframe.bfill()  # 后向填充初始值

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 开多仓条件：
        # 1. 5 分钟 MACD 快线大于慢线
        # 2. 1 小时 MACD 快线大于慢线
        # 3. 价格在 200 EMA 之上（趋势过滤）
        # 4. ADX > 25（趋势强度过滤）
        # 5. RSI < 70（避免超买）
        dataframe.loc[
            (
                (
                    dataframe["volume"] > dataframe["volume"].quantile(0.1)
                )  # 过滤掉交易量较小的K线 # noqa: E501
                & (dataframe["macd"] > dataframe["macdsignal"])  # 5 分钟 MACD 快线大于慢线
                & (dataframe["macd_1h"] > dataframe["macdsignal_1h"])  # 1 小时 MACD 快线大于慢线
                & (
                    dataframe["bullish_adx"] > self.buy_adx_threshold.value
                )  # ADX > 25，趋势强度足够
                & (~dataframe["bearish_divergence"])  # 避免看跌背离时做多
                & (dataframe["rsi"] < self.sell_rsi_threshold.value)  # RSI < 70，避免超卖
            ),
            ["enter_long", "enter_tag"],
        ] = (1, "5m和1h MACD金叉，ADX强，无熊背离，RSI低，做多")

        # 开空仓条件：
        # 1. 5 分钟 MACD 快线小于慢线
        # 2. 1 小时 MACD 快线小于慢线
        # 3. 价格在 200 EMA 之下（趋势过滤）
        # 4. ADX > 25（趋势强度过滤）
        # 5. RSI > 30（避免超卖）
        dataframe.loc[
            (
                (dataframe["volume"] > dataframe["volume"].quantile(0.1))  # 过滤掉交易量较小的K线
                & (dataframe["macd"] < dataframe["macdsignal"])  # 5 分钟 MACD 快线小于慢线
                & (dataframe["macd_1h"] < dataframe["macdsignal_1h"])  # 1 小时 MACD 快线小于慢线
                & (dataframe["bearish_adx"] > self.sell_adx_threshold.value)  # ADX趋势强度足够
                & (~dataframe["bullish_divergence"])  # 避免看涨背离时做空
                & (dataframe["rsi"] > self.buy_rsi_threshold.value)  # RSI > 30，避免超买
            ),
            ["enter_short", "enter_tag"],
        ] = (1, "5m和1h MACD死叉，ADX强，无牛背离，RSI高，做空")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # # 判断金叉：当前 MACD 快线从下往上穿过 MACD 信号线
        # dataframe['golden_cross'] = (
        #     (dataframe['macd_1h'] > dataframe['macdsignal_1h']) &  # 当前快线在慢线之上
        #     (dataframe['macd_1h'].shift(1) <= dataframe['macdsignal_1h'].shift(1))  # 前一根快线在慢线之下 # noqa: E501
        # )

        # # 判断死叉：当前 MACD 快线从上往下穿过 MACD 信号线
        # dataframe['dead_cross'] = (
        #     (dataframe['macd_1h'] < dataframe['macdsignal_1h']) &  # 当前快线在慢线之下
        #     (dataframe['macd_1h'].shift(1) >= dataframe['macdsignal_1h'].shift(1))  # 前一根快线在慢线之上 # noqa: E501
        # )

        # # 1h出现死叉信号，平多单
        # dataframe.loc[
        #     (dataframe['dead_cross']),
        #     'exit_long'] = 1

        # # 1h出现金叉信号，平空单
        # dataframe.loc[
        #     (dataframe['golden_cross']),
        #     'exit_short'] = 1

        return dataframe

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        dataframe = self.dp.get_analyzed_dataframe(pair, self.timeframe)[0]
        last_candle = dataframe.iloc[-1]
        atr = last_candle["atr"]
        volatility = atr / last_candle["close"]
        if volatility < 0.01:
            return 3.0
        elif volatility < 0.02:
            return 2.0
        else:
            return 1.0

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> str | bool | None:
        """
        动态止盈止损逻辑
        """
        # ICU中的交易订单不需要处理
        icu_tag = trade.get_custom_data("icu_tag")
        if icu_tag is not None:
            return None

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        # 计算持仓时间（分钟）
        hold_time = (current_time - trade.open_date_utc).total_seconds() / 60

        # 获取当前趋势强度和波动性
        trend_strength = last_candle["trend_strength"]
        atr = last_candle["atr"]
        momentum = last_candle["momentum"]

        # === 动态止盈逻辑 ===
        if current_profit > 0:
            # 处理可能为 None 的 max_rate
            max_rate = trade.max_rate if trade.max_rate is not None else current_rate
            exit_reason = self._check_take_profit(
                trade, current_profit, trend_strength, momentum, current_rate, max_rate
            )
            if exit_reason:
                return exit_reason

        # === 动态止损逻辑 ===
        if current_profit < 0:
            exit_reason = self._check_stop_loss(trade, current_profit, hold_time, last_candle, atr)
            if exit_reason:
                return exit_reason

        # === 时间止损 ===
        # if hold_time > 3600:  # 60小时
        #     if current_profit > 0:
        #         return "timeout_with_profit"
        #     elif current_profit < -0.05:
        #         return "timeout_with_loss"

        return None

    def _check_take_profit(
        self,
        trade: Trade,
        current_profit: float,
        trend_strength: float,
        momentum: float,
        current_rate: float,
        max_rate: float,
    ) -> str | None:
        """检查动态止盈条件"""
        # 1. 超高利润保护（>15%）
        if current_profit >= 0.15:
            return "tp_over_15_percent"

        # 2. 高利润但趋势减弱
        if current_profit >= 0.10 and trend_strength < 2.0:
            return "tp_trend_weakening"

        # 3. 中等利润但动量反转
        if current_profit >= 0.05 and (
            (trade.entry_side == "long" and momentum < -0.01)
            or (trade.entry_side == "short" and momentum > 0.01)
        ):
            return "tp_momentum_reversal"

        # 4. 利润回撤保护
        profit_threshold = max(0.05, min(0.05, current_profit * 0.7))
        if max_rate > current_rate and (max_rate - current_rate) / max_rate > profit_threshold:
            return f"tp_drawdown_{profit_threshold:.0%}"

        return None

    def _check_stop_loss(
        self, trade: Trade, current_profit: float, hold_time: float, last_candle, atr: float
    ) -> str | None:
        """检查动态止损条件"""
        # 1. 快速止损（大幅亏损）
        # if current_profit <= -0.5:
        #     return "sl_emergency"

        # 2. 趋势反转止损
        if hold_time > 10 and (  # 至少持仓10分钟
            (trade.entry_side == "long" and last_candle["close"] < last_candle["trend"])
            or (trade.entry_side == "short" and last_candle["close"] > last_candle["trend"])
        ):
            return "sl_trend_reversal"

        return None

    def custom_entry_price(
        self,
        pair: str,
        trade: Trade | None,
        current_time: datetime,
        proposed_rate: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        """
        自定义入场价格回调函数，结合趋势指标和波动率指标判断。
        """
        if trade is None:
            # 如果 trade 为 None，返回默认值或记录日志
            logger.debug("Trade object is None in custom_entry_price. Returning proposed_rate.")
            return proposed_rate

        dataframe = self.dp.get_pair_dataframe(pair=trade.pair, timeframe=self.timeframe)

        # 计算 MACD
        macd, macdsignal, macdhist = ta.MACD(
            dataframe["close"], fastperiod=12, slowperiod=26, signalperiod=9
        )
        dataframe["macd"] = macd
        dataframe["macdsignal"] = macdsignal

        # 计算 ATR
        dataframe["atr"] = ta.ATR(
            dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=14
        )

        # 获取最近 5 根 K 线数据
        last_5_candles = dataframe.tail(5)

        # 动态调整涨跌幅阈值
        atr = dataframe["atr"].iloc[-1]
        threshold = 0.03 + atr * 0.01  # 基础阈值 + 波动率调整

        # 判断是否满足条件 - 根据市场条件调整入场价格
        if trade.is_short:  # 空单
            change_percent = (
                last_5_candles["close"].iloc[-1] - last_5_candles["close"].iloc[0]
            ) / last_5_candles["close"].iloc[0]
            if (
                all(last_5_candles["close"] < last_5_candles["open"]) & change_percent
                < -threshold & dataframe["macd"].iloc[-1]
                < dataframe["macdsignal"].iloc[-1]
            ):
                # 连续下跌且跌幅超过阈值，且 MACD 为空头信号
                # 对于空单，当市场下跌强劲时，可以设置略低的入场价格获得更好的入场点
                adjusted_rate = proposed_rate * 0.995  # 降低0.5%的入场价
                logger.info(
                    f"强势下跌市场检测到，调整空单入场价格从 {proposed_rate} 到 {adjusted_rate}"
                )
                return adjusted_rate
        else:  # 多单
            change_percent = (
                last_5_candles["close"].iloc[-1] - last_5_candles["close"].iloc[0]
            ) / last_5_candles["close"].iloc[0]
            if (
                all(last_5_candles["close"] > last_5_candles["open"]) & change_percent > threshold
                and dataframe["macd"].iloc[-1] > dataframe["macdsignal"].iloc[-1]
            ):
                # 连续上涨且涨幅超过阈值，且 MACD 为多头信号
                # 对于多单，当市场上涨强劲时，可以设置略高的入场价格以确保订单成交
                adjusted_rate = proposed_rate * 1.005  # 提高0.5%的入场价
                logger.info(
                    f"强势上涨市场检测到，调整多单入场价格从 {proposed_rate} 到 {adjusted_rate}"
                )
                return adjusted_rate

        # 默认情况下返回建议价格
        return proposed_rate

    # dca方法
    position_adjustment_enable = False
    # 开仓数量乘数
    mul_ = 1.0

    # 多头专用参数
    long_safety_trigger = -0.018  # 比空头更宽松的阈值(-1.8%)
    long_volume_scale = 2.0  # 更高的补仓倍率
    max_long_safety_orders = 3  # 允许更多补仓次数

    # 空头参数（保持原样）
    short_safety_trigger = -0.015
    short_volume_scale = 2.0
    max_short_safety_orders = 3

    # 通用参数
    max_pair_risk_ratio = 0.15
    min_stake = 10

    def get_open_stake_amount(self):
        total_stake_amount = self.wallets.get_total("USDT")
        proposed_stake = round(total_stake_amount / self.config["max_open_trades"])
        return proposed_stake * self.mul_

    # 获取持仓时间
    def get_pos_time(self, trade, current_time):
        open_order_side = "sell" if trade.is_short else "buy"
        filled_entries = trade.select_filled_orders(open_order_side)
        # 获取首个开仓的order
        first_open_order = filled_entries[0]
        # 获取最后开仓的order
        last_open_order = filled_entries[-1]
        current_time_utc = current_time.replace(tzinfo=_datetime.timezone.utc)
        first_open_order_time_utc = first_open_order.order_filled_utc
        last_open_order_time_utc = last_open_order.order_filled_utc
        # 计算首次开仓到现在的时间 分钟
        time_delta_first = int((current_time_utc - first_open_order_time_utc).total_seconds() / 60)
        # 计算最后一次开仓到现在的时间 分钟
        time_delta_last = int((current_time_utc - last_open_order_time_utc).total_seconds() / 60)
        return time_delta_first, time_delta_last

    def adjust_trade_position(
        self,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: float | None,
        max_stake: float,
        current_entry_rate: float,
        current_exit_rate: float,
        current_entry_profit: float,
        current_exit_profit: float,
        **kwargs: Any,
    ) -> float | tuple[float | None, str | None] | None:
        """主仓位调整逻辑（重构后）"""
        if self._should_skip_adjustment(trade):
            return None

        dataframe = self._get_validated_dataframe(trade.pair)
        if dataframe is None:
            return None

        signal = self._generate_trade_signals(dataframe)
        if not self._validate_addition_conditions(trade, dataframe, current_time):
            return None

        return self._process_position_adjustment(trade, current_rate, current_profit, signal)

    def _should_skip_adjustment(self, trade: Trade) -> bool:
        """判断是否需要跳过仓位调整"""
        return trade.get_custom_data("icu_tag") is not None

    def _get_validated_dataframe(self, pair: str) -> pd.DataFrame | None:
        """获取并验证数据有效性"""
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        return dataframe if len(dataframe) >= 2 else None

    def _generate_trade_signals(self, dataframe: pd.DataFrame) -> dict:
        """生成交易信号"""
        last = dataframe.iloc[-1].squeeze()
        prev = dataframe.iloc[-2].squeeze()
        return {
            "golden_cross": (
                (last["macd"] > last["macdsignal"]) & (prev["macd"] <= prev["macdsignal"])
            ),
            "death_cross": (
                (last["macd"] < last["macdsignal"]) & (prev["macd"] >= prev["macdsignal"])
            ),
            "price_down": last["close"] < prev["close"],
        }

    def _validate_addition_conditions(
        self, trade: Trade, dataframe: pd.DataFrame, current_time: datetime
    ) -> bool:
        """验证加仓基本条件"""
        last_candle = dataframe.iloc[-1].squeeze()
        prev_candle = dataframe.iloc[-2].squeeze()

        if last_candle["close"] < prev_candle["close"]:
            return False

        return self._is_addition_time_valid(trade, current_time)

    def _is_addition_time_valid(self, trade: Trade, current_time: datetime) -> bool:
        """验证加仓时间间隔"""
        time_rules = {1: (10, "禁止加仓 1-1"), 2: (15, "禁止加仓 1-2"), 3: (25, "禁止加仓 1-3")}

        entries = trade.nr_of_successful_entries
        time_delta = self.get_pos_time(trade, current_time)[1]

        if entries in time_rules:
            min_interval, message = time_rules[entries]
            if time_delta < min_interval:
                self._log_addition_denial(trade, message, time_delta)
                return False
        return True

    def _log_addition_denial(self, trade: Trade, message: str, time_delta: float):
        """记录加仓拒绝日志"""
        if self.dp.runmode.value in ("live", "dry"):
            logger.info(
                trade.pair,
                message,
                self.get_pos_time(trade, datetime.now())[0],
                time_delta,
                trade.stake_amount,
                trade.nr_of_successful_entries,
            )

    def _process_position_adjustment(
        self, trade: Trade, current_rate: float, current_profit: float, signal: dict
    ) -> float | None:
        """处理实际仓位调整"""
        if trade.is_short:
            return self._handle_short_position(trade, current_rate, current_profit, signal)
        return self._handle_long_position(trade, current_rate, current_profit, signal)

    def _handle_long_position(
        self, trade: Trade, rate: float, profit: float, signal: dict
    ) -> float | None:
        """处理多头加仓"""
        if not signal["golden_cross"]:
            return None
        self._record_addition(trade, self.calculate_long_addition_stake, rate)
        return self._process_long_addon(trade, rate, profit)

    def _handle_short_position(
        self, trade: Trade, rate: float, profit: float, signal: dict
    ) -> float | None:
        """处理空头加仓"""
        if not signal["death_cross"]:
            return None
        self._record_addition(trade, self.calculate_short_addition_stake, rate)
        return self._process_short_addon(trade, rate, profit)

    def _record_addition(self, trade: Trade, stake_calculator: Callable, rate: float):
        """记录加仓信息到交易数据"""
        additions = trade.get_custom_data("additions") or []
        if not additions:
            trade.set_custom_data("additions", additions)

        additions.append(
            {
                "stake": stake_calculator(trade.nr_of_successful_entries),
                "entry_price": rate,
                "target_profit": 0.01,
            }
        )

    def calculate_long_addition_stake(self, count: int) -> float:
        base_stake = self.get_open_stake_amount()
        scaled_stake = base_stake * (self.long_volume_scale ** (count - 1))
        return scaled_stake

    def calculate_short_addition_stake(self, count: int) -> float:
        base_stake = self.get_open_stake_amount()
        scaled_stake = base_stake * (self.short_volume_scale ** (count - 1))
        return scaled_stake

    def _process_long_addon(self, trade: Trade, rate: float, profit: float) -> float | None:
        """多头补仓专用处理模块（与空头相同标准）"""
        # 触发条件检测
        if profit > self.long_safety_trigger:
            logger.debug(f"多单 {trade.pair} 当前收益{profit * 100:.2f}% 未达补仓阈值")
            return None

        # 补仓次数验证
        count = trade.nr_of_successful_entries
        if not (1 <= count <= self.max_long_safety_orders):
            logger.debug(
                f"多单 {trade.pair} 已补仓{count}次，超出最大限制{self.max_long_safety_orders}"
            )
            return None

        try:
            # 获取当前所有开放交易
            open_trades = Trade.get_open_trades()
            # 动态仓位计算
            base_stake = self.get_open_stake_amount()

            # 多头专用缩放逻辑（指数增长）
            scaled_stake = base_stake * (self.long_volume_scale ** (count - 1))

            # 风险控制层
            if self.wallets:
                total_capital = self.wallets.get_total_stake_amount()
            max_per_pair = total_capital * self.max_pair_risk_ratio

            # 当前品种多头总投入
            current_investment = sum(
                t.stake_amount
                for t in open_trades
                if t and t.pair == trade.pair and not t.is_short and t.id != trade.id  # 排除自身
            )
            available = max(max_per_pair - current_investment, 0)

            final_stake = min(scaled_stake, available)
            final_stake = max(final_stake, self.min_stake)

            # 时间间隔验证
            if not self._check_time_interval(trade):
                logger.debug(f"多单 {trade.pair} 未满足补仓时间间隔要求")
                return None

            # 记录补仓日志
            logger.info(
                f"""多单补仓触发：
                交易对：{trade.pair}
                补仓次数：{count}/{self.max_long_safety_orders}
                理论仓位：{scaled_stake:.4f} {self.stake_currency}
                实际分配：{final_stake:.4f}
                当前投入：{current_investment:.2f}
                允许上限：{max_per_pair:.2f}"""
            )

            return final_stake

        except Exception as e:
            logger.error(f"多单补仓计算失败：{str(e)}", exc_info=True)
            return None

    def _process_short_addon(self, trade: Trade, rate: float, profit: float) -> float | None:
        """空头补仓专用处理模块"""
        # 触发条件检测
        if profit > self.short_safety_trigger:
            logger.debug(f"空单 {trade.pair} 当前收益{profit * 100:.2f}% 未达到补仓阈值")
            return None

        # 补仓次数验证
        count = trade.nr_of_successful_entries
        if not (1 <= count <= self.max_short_safety_orders):
            logger.debug(
                f"空单 {trade.pair} 已补仓{count}次，超出最大限制{self.max_short_safety_orders}"
            )
            return None

        try:
            # 获取当前所有开放交易
            open_trades = Trade.get_open_trades()
            # 动态仓位计算
            base_stake = self.get_open_stake_amount()

            # 空头专用缩放逻辑
            scaled_stake = base_stake * (self.short_volume_scale ** (count - 1))

            # 风险控制层
            if self.wallets:
                total_capital = self.wallets.get_total_stake_amount()
            max_per_pair = total_capital * self.max_pair_risk_ratio  # 单品种最大风险比例

            # 计算同品种空单总投入（排除自身）
            current_investment = sum(
                t.stake_amount
                for t in open_trades
                if t
                and t.pair == trade.pair  # 空值保护
                and t.is_short
                and t.id != trade.id  # 关键修正点
            )
            available = max(max_per_pair - current_investment, 0)

            final_stake = min(scaled_stake, available)
            final_stake = max(final_stake, self.min_stake)  # 保证最小仓位

            # 时间间隔验证
            if not self._check_time_interval(trade):
                logger.debug(f"空单 {trade.pair} 未满足补仓时间间隔要求")
                return None

            # 记录补仓日志
            logger.info(
                f"""空单补仓触发：
                交易对：{trade.pair}
                补仓次数：{count}/{self.max_short_safety_orders}
                理论仓位：{scaled_stake:.4f} {self.stake_currency}
                实际分配：{final_stake:.4f}
                当前投入：{current_investment:.2f}
                允许上限：{max_per_pair:.2f}"""
            )

            return final_stake

        except Exception as e:
            logger.error(f"空单补仓计算失败：{str(e)}", exc_info=True)
            return None

    def _check_time_interval(self, trade: Trade) -> bool:
        """补仓时间间隔控制系统"""
        pair = trade.pair

        # 从策略属性获取时间记录（首次运行初始化）
        if not hasattr(self, "last_addon_time"):
            self.last_addon_time: dict[str, datetime] = {}

        # 获取最近补仓时间
        last_time = self.last_addon_time.get(pair, None)

        # 时间间隔配置（可从参数中读取）
        min_interval = timedelta(minutes=240)  # 默认4小时

        # 时区处理（重要！）
        current_time = datetime.now(timezone.utc)

        if last_time:
            # 转换为带时区信息的时间
            last_time = (
                last_time.replace(tzinfo=timezone.utc) if not last_time.tzinfo else last_time
            )

            # 时间间隔检查
            if current_time - last_time < min_interval:
                logger.debug(f"交易对 {pair} 上次补仓时间 {last_time}，未达到最小间隔")
                return False

        # 更新最后补仓时间
        self.last_addon_time[pair] = current_time
        return True

    def custom_stoploss(
        self,
        pair: str,
        trade: "Trade",
        current_time: "datetime",
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ) -> float | None:
        # ICU中的交易订单不需要处理
        icu_tag = trade.get_custom_data("icu_tag")
        if icu_tag is not None:
            return None

        return self.stoploss

    def confirm_trade_exit(
        self,
        pair: str,
        trade: "Trade",
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        exit_reason: str,
        current_time: "datetime",
        **kwargs,
    ) -> bool:
        # ICU中的交易订单不需要处理
        icu_tag = trade.get_custom_data("icu_tag")
        if icu_tag is not None:
            return False  # 不确认退出

        return True

    # def custom_exit_price(self, pair: str, trade: 'Trade', current_time: 'datetime',
    #                     current_rate: float, proposed_rate: float, **kwargs) -> float | None:
    #     # ICU中的交易订单不需要处理
    #     icu_tag = trade.get_custom_data('icu_tag')
    #     if icu_tag is not None:
    #         return None

    #     return proposed_rate

    def custom_adjust_position(
        self, tradeid: int, quantity: float, exchange: "Exchange"
    ) -> str | None:
        """
        自定义补仓/减仓逻辑
        参数:
            tradeid: 交易 ID
            quantity: 补仓数量（正值为补仓，负值为减仓）
            exchange: 交易所实例
        返回:
            str: 操作结果消息
        """
        # 简单的锁机制防止并发操作（可选）
        position_adjust_lock = getattr(self, "_position_adjust_lock", {})
        if position_adjust_lock.get(tradeid, False):
            return "该交易正在被补单，请稍后再试"

        try:
            # 设置锁
            self._position_adjust_lock = position_adjust_lock.copy()
            self._position_adjust_lock[tradeid] = True

            # 获取交易并进行安全检查
            trade = Trade.get_trades([Trade.id == tradeid]).first()
            if not trade:
                return "交易不存在"

            # 验证补单数量
            if quantity == 0:
                return "补单数量不能为零"

            if quantity < 0 and abs(quantity) >= trade.amount:
                return f"减仓数量({abs(quantity)})不能大于或等于当前持仓({trade.amount})"

            # 获取交易所标准合约符号 (例如 DYDX/USDT:USDT → DYDXUSDT)
            market_info = exchange.markets[trade.pair]
            if not market_info:
                logger.error(f"无法获取市场信息: {trade.pair}")
            else:
                unified_symbol = market_info["id"]
                logger.info(f"统一合约符号: {unified_symbol}")

            # 确定操作方向
            is_increasing = quantity > 0
            order_side = (
                "buy"
                if (is_increasing and not trade.is_short) or (not is_increasing and trade.is_short)
                else "sell"
            )

            ticker = exchange.fetch_ticker(trade.pair)
            current_price = ticker.get("last", None)

            # 备用获取 bid 价格逻辑
            current_price_low = None
            if "bid" in ticker and ticker["bid"] is not None:
                current_price_low = ticker["bid"]
            else:
                logger.warning("Ticker 中 bid 为空，尝试从 Order Book 获取")
                current_price_low = current_price
            if not current_price:
                return "无法获取当前价格"

            # 处理数据库操作 - 避免使用 with session.begin()
            try:
                # 计算成本和费用
                abs_quantity = abs(quantity)
                fee_rate = exchange.get_fee(symbol=trade.pair, taker_or_maker="taker")
                raw_cost = abs_quantity * current_price
                fee = raw_cost * fee_rate
                final_cost = raw_cost + fee

                # 生成唯一订单ID
                timestamp = str(time.time())
                run_mode = "dry_run" if self.config["dry_run"] else "live"
                order_id = f"{run_mode}_{order_side}_{trade.pair}_{timestamp}"

                # 创建新订单
                new_order = Order(
                    ft_trade_id=trade.id,
                    ft_pair=trade.pair,
                    ft_order_side=order_side,
                    ft_amount=abs_quantity,
                    ft_price=current_price,
                    amount=abs_quantity,
                    filled=abs_quantity,
                    remaining=0,
                    cost=final_cost,
                    order_type="market",
                    price=current_price,
                    average=current_price,
                    status="closed",
                    symbol=trade.pair,
                    order_date=datetime.now(),
                    order_filled_date=datetime.now(),
                    order_update_date=datetime.now(),
                    side=order_side,
                    ft_order_tag=(
                        f"{'补单' if is_increasing else '减仓'}：{abs_quantity} @ {current_price}"
                    ),
                    ft_is_open=False,
                    order_id=order_id,
                )

                # 调整最高/最低价记录
                # 确保所有价格都已成功获取且有效 (非 None 且大于 0)
                if (
                    current_price
                    and current_price_low
                    and current_price > 0
                    and current_price_low > 0
                ):
                    # 使用所有必需的参数调用 adjust_min_max_rates
                    trade.adjust_min_max_rates(current_price, current_price_low)
                    logger.info(
                        f"已为 {trade.pair} 调整最低/最高费率: "
                        f"Price={current_price}, Low={current_price_low}"
                    )

                else:
                    # 如果缺少 bid 或 ask，可以考虑使用 'last' 作为备用，但这会降低准确性
                    # 或者记录警告并跳过调整
                    logger.warning(
                        f"无法为 {trade.pair} 获取有效的 'last'({current_price}) 和 "
                        f"'bid'({current_price_low}) 价格，跳过 adjust_min_max_rates。"
                    )

                Trade.session.add(new_order)
                Trade.session.commit()
                # 更新交易状态
                trade.recalc_trade_from_orders()
                # 保存交易
                Trade.session.commit()

            except Exception as e:
                # 回滚事务 - 不需要显式调用，因为 Freqtrade 会管理会话
                Trade.session.rollback()  # 如果需要手动回滚，取消这行的注释
                logger.error(f"回滚事务，仓位补单失败: {e}")
                raise

            # 返回成功消息
            operation_type = "补仓" if is_increasing else "减仓"
            return f"成功{operation_type}: {abs_quantity}，新平均价格: {trade.open_rate:.8f}"

        except Exception as e:
            logger.error(f"仓位补单失败: {e}", exc_info=True)
            return f"操作失败: {str(e)}"
        finally:
            # 释放锁
            if hasattr(self, "_position_adjust_lock") and tradeid in self._position_adjust_lock:
                self._position_adjust_lock[tradeid] = False
