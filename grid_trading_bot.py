import asyncio
import json
import logging
import os
import signal
import sys
from datetime import datetime
from functools import wraps
from typing import Dict, List, Set, Tuple

from bpx.bpx import *
from bpx.bpx_pub import *
from dotenv import load_dotenv
from requests.exceptions import ConnectionError, RequestException, Timeout

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("grid_trading.log"), logging.StreamHandler()],
)
logger = logging.getLogger("grid_trading")


running = True
trade_history = []
total_profit_loss = 0
total_fee = 0


def retry_on_network_error(max_retries=5, backoff_factor=1.5, max_backoff=60):
    """网络错误重试装饰器"""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            retries = 0
            backoff = 1

            while retries < max_retries:
                try:
                    return await func(*args, **kwargs)
                except (RequestException, ConnectionError, Timeout) as e:
                    retries += 1
                    if retries >= max_retries:
                        logger.error(f"达到最大重试次数 {max_retries}，操作失败: {e}")
                        raise

                    backoff = min(backoff * backoff_factor, max_backoff)
                    logger.warning(
                        f"网络错误: {e}. 将在 {backoff:.1f} 秒后重试 (尝试 {retries}/{max_retries})"
                    )
                    await asyncio.sleep(backoff)

            raise Exception("重试机制逻辑错误")

        return wrapper

    return decorator


class GridTradingBot:
    """网格交易机器人类"""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        symbol: str,
        lower_price: float,
        upper_price: float,
        num_grids: int,
        total_investment: float,
        pair_accuracy: int = 2,
        fee_rate: float = 0.0008,
    ):
        """
        初始化网格交易机器人

        Args:
            api_key: API密钥
            api_secret: API密钥
            symbol: 交易对，如 "SOL_USDC"
            lower_price: 网格下限价格
            upper_price: 网格上限价格
            num_grids: 网格数量
            total_investment: 总投资额(USDC)
            pair_accuracy: 交易对价格精度
            fee_rate: 手续费率
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.symbol = symbol
        self.lower_price = lower_price
        self.upper_price = upper_price
        self.num_grids = num_grids
        self.total_investment = total_investment
        self.pair_accuracy = pair_accuracy
        self.fee_rate = fee_rate

        self.base_currency = symbol.split("_")[0]
        self.quote_currency = symbol.split("_")[1]

        self.bpx = BpxClient()
        self.bpx.init(api_key=api_key, api_secret=api_secret)

        self.grid_prices = self.create_grid()

        self.quantity_per_grid = total_investment / num_grids

        self.active_orders: Dict[str, Dict] = {}  # orderId -> order_info
        self.order_grid_mapping: Dict[str, int] = {}  # orderId -> grid_index

        self.history_file = f"trade_history_{symbol}.json"
        self.load_trade_history()

        logger.info(
            f"网格交易机器人初始化完成: {symbol}, 价格范围: {lower_price}-{upper_price}, 网格数: {num_grids}"
        )

    def create_grid(self) -> List[float]:
        """创建价格网格"""
        grid_size = (self.upper_price - self.lower_price) / self.num_grids
        return [
            round(self.lower_price + i * grid_size, self.pair_accuracy)
            for i in range(self.num_grids + 1)
        ]

    @retry_on_network_error(max_retries=5, backoff_factor=1.5)
    async def get_market_price(self) -> float:
        """获取当前市场价格"""
        try:
            market_depth = Depth(self.symbol)
            current_price = round(
                (float(market_depth["asks"][0][0]) + float(market_depth["bids"][-1][0]))
                / 2,
                self.pair_accuracy,
            )
            return current_price
        except Exception as e:
            logger.error(f"获取市场价格失败: {e}")
            if hasattr(self, "last_price") and self.last_price:
                return self.last_price
            raise

    @retry_on_network_error(max_retries=5, backoff_factor=1.5)
    async def get_account_balance(self) -> Tuple[float, float]:
        """获取账户余额"""
        account_balance = self.bpx.balances()
        base_available = float(account_balance[self.base_currency]["available"])
        quote_available = float(account_balance[self.quote_currency]["available"])
        return base_available, quote_available

    async def place_grid_orders(self) -> None:
        """在网格上放置买卖订单"""
        await self.cancel_all_orders()

        current_price = await self.get_market_price()
        self.last_price = current_price
        logger.info(f"当前市场价格: {current_price}")

        current_grid_index = 0
        for i, price in enumerate(self.grid_prices):
            if price > current_price:
                current_grid_index = i - 1
                break

        logger.info(f"当前价格所在网格索引: {current_grid_index}")

        self.active_orders = {}
        self.order_grid_mapping = {}

        buy_tasks = []
        for i in range(current_grid_index, -1, -1):
            buy_tasks.append(self.place_buy_order(i))

        sell_tasks = []
        for i in range(current_grid_index + 1, len(self.grid_prices)):
            sell_tasks.append(self.place_sell_order(i))

        await asyncio.gather(*buy_tasks, *sell_tasks)

        active_orders = self.bpx.ordersQuery(self.symbol)
        if active_orders:
            logger.info(f"成功放置 {len(active_orders)} 个订单")
        else:
            logger.warning("未检测到活跃订单，可能下单失败")

    async def place_buy_order(self, grid_index: int) -> None:
        """放置买单"""
        price = self.grid_prices[grid_index]
        try:
            qty = self.calculate_quantity(price)

            response = self.bpx.ExeOrder(
                symbol=self.symbol,
                side="Bid",
                orderType="Limit",
                timeInForce="",
                quantity=qty,
                price=price,
            )

            if response and isinstance(response, dict) and "id" in response:
                order_id = response["id"]
                self.active_orders[order_id] = {
                    "price": price,
                    "quantity": qty,
                    "side": "Bid",
                    "grid_index": grid_index,
                }
                self.order_grid_mapping[order_id] = grid_index
                logger.info(f"买单已放置: 价格 {price}, 数量 {qty}, 订单ID: {order_id}")
            else:
                logger.warning(f"买单放置可能失败: {response}")
        except Exception as e:
            logger.error(f"放置买单失败: {e}")

    async def place_sell_order(self, grid_index: int) -> None:
        """放置卖单"""
        price = self.grid_prices[grid_index]
        try:
            qty = self.calculate_quantity(price)

            response = self.bpx.ExeOrder(
                symbol=self.symbol,
                side="Ask",
                orderType="Limit",
                timeInForce="",
                quantity=qty,
                price=price,
            )

            if response and isinstance(response, dict) and "id" in response:
                order_id = response["id"]
                self.active_orders[order_id] = {
                    "price": price,
                    "quantity": qty,
                    "side": "Ask",
                    "grid_index": grid_index,
                }
                self.order_grid_mapping[order_id] = grid_index
                logger.info(f"卖单已放置: 价格 {price}, 数量 {qty}, 订单ID: {order_id}")
            else:
                logger.warning(f"卖单放置可能失败: {response}")
        except Exception as e:
            logger.error(f"放置卖单失败: {e}")

    def calculate_quantity(self, price: float) -> float:
        """计算订单数量，确保格式正确"""
        qty = int(self.quantity_per_grid / price * 100) / 100
        return qty

    async def cancel_all_orders(self) -> None:
        """取消所有订单"""
        try:
            self.bpx.ordersCancel(self.symbol)
            logger.info(f"已取消所有 {self.symbol} 订单")
        except Exception as e:
            logger.error(f"取消订单失败: {e}")

    def load_trade_history(self) -> None:
        """加载历史交易记录"""
        global trade_history, total_profit_loss, total_fee

        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r") as f:
                    data = json.load(f)
                    trade_history = data.get("trades", [])
                    total_profit_loss = data.get("total_profit_loss", 0)
                    total_fee = data.get("total_fee", 0)
                    logger.info(f"已加载历史交易记录: {len(trade_history)}笔交易")
                    logger.info(
                        f"累计盈亏: {total_profit_loss:.4f} USDC, 累计手续费: {total_fee:.4f} USDC"
                    )
            except Exception as e:
                logger.error(f"加载历史交易记录失败: {e}")

    async def save_trade_history(self) -> None:
        """保存交易记录"""
        try:
            with open(self.history_file, "w") as f:
                json.dump(
                    {
                        "trades": trade_history,
                        "total_profit_loss": total_profit_loss,
                        "total_fee": total_fee,
                    },
                    f,
                    indent=2,
                )
            logger.info("交易记录已保存")
        except Exception as e:
            logger.error(f"保存交易记录失败: {e}")

    async def process_new_trades(self, executed_order_ids: Set[str]) -> None:
        """处理新成交的订单"""
        global trade_history, total_profit_loss, total_fee

        if not executed_order_ids:
            return

        try:
            for order_id in executed_order_ids:
                if order_id in self.active_orders:
                    order_info = self.active_orders[order_id]
                    side = order_info.get("side")
                    price = order_info.get("price", 0)
                    quantity = order_info.get("quantity", 0)
                    value = price * quantity
                    fee = value * self.fee_rate

                    profit_loss = 0

                    if side == "Bid":
                        profit_loss = 0
                    else:
                        grid_index = order_info.get("grid_index")
                        if grid_index > 0:
                            buy_price = self.grid_prices[grid_index - 1]
                            buy_value = buy_price * quantity
                            profit_loss = value - buy_value
                        else:
                            profit_loss = 0
                            logger.warning(
                                f"无法确定卖单 {order_id} 的买入成本，盈亏计为0"
                            )

                    trade_record = {
                        "trade_id": order_id,
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "side": side,
                        "price": price,
                        "quantity": quantity,
                        "value": value,
                        "fee": fee,
                        "profit_loss": profit_loss,
                    }

                    trade_history.append(trade_record)
                    total_fee += fee
                    total_profit_loss += profit_loss

                    logger.info(
                        f"检测到新成交: {side} {quantity} {self.base_currency} @ {price} {self.quote_currency}"
                    )
                    logger.info(
                        f"交易价值: {value:.4f} {self.quote_currency}, 手续费: {fee:.4f} {self.quote_currency}"
                    )

                    if side == "Ask" and profit_loss != 0:
                        logger.info(
                            f"本次交易盈亏: {profit_loss:.4f} {self.quote_currency} (扣除买入成本)"
                        )

            await self.save_trade_history()

            logger.info(
                f"累计盈亏: {total_profit_loss:.4f} {self.quote_currency} (含手续费: {total_fee:.4f} {self.quote_currency})"
            )
            logger.info(
                f"净盈亏: {(total_profit_loss - total_fee):.4f} {self.quote_currency}"
            )

        except Exception as e:
            logger.error(f"处理成交记录失败: {e}")

    async def rebalance_single_order(self, executed_order_id: str) -> None:
        """处理单个订单成交后的网格重新平衡"""
        try:
            if executed_order_id not in self.order_grid_mapping:
                logger.warning(
                    f"订单ID {executed_order_id} 不在网格映射中，无法精确重新平衡"
                )
                return

            grid_index = self.order_grid_mapping[executed_order_id]
            order_info = self.active_orders.get(executed_order_id, {})
            side = order_info.get("side")

            if not side:
                logger.warning(f"无法获取订单 {executed_order_id} 的方向信息")
                return

            current_price = await self.get_market_price()
            current_grid_index = 0
            for i, price in enumerate(self.grid_prices):
                if price > current_price:
                    current_grid_index = i - 1
                    break

            logger.info(
                f"订单成交: ID={executed_order_id}, 网格索引={grid_index}, 方向={side}"
            )

            if side == "Bid":
                new_grid_index = grid_index + 1
                if new_grid_index < len(self.grid_prices):
                    logger.info(f"买单成交，在网格 {new_grid_index} 放置新卖单")
                    await self.place_sell_order(new_grid_index)
                else:
                    logger.warning("买单成交，但已达到最高网格，无法放置新卖单")

                logger.info(f"在原网格 {grid_index} 重新放置买单")
                await self.place_buy_order(grid_index)

            elif side == "Ask":
                new_grid_index = grid_index - 1
                if new_grid_index >= 0:
                    logger.info(f"卖单成交，在网格 {new_grid_index} 放置新买单")
                    await self.place_buy_order(new_grid_index)
                else:
                    logger.warning("卖单成交，但已达到最低网格，无法放置新买单")

                logger.info(f"在原网格 {grid_index} 重新放置卖单")
                await self.place_sell_order(grid_index)

            if executed_order_id in self.active_orders:
                del self.active_orders[executed_order_id]
            if executed_order_id in self.order_grid_mapping:
                del self.order_grid_mapping[executed_order_id]

        except Exception as e:
            logger.error(f"重新平衡单个订单失败: {e}")

    async def monitor_and_rebalance(self) -> None:
        """监控订单执行情况并重新平衡网格"""
        previous_order_ids: Set[str] = set()

        self.initial_price = await self.get_market_price()

        if not await self.check_market_liquidity():
            logger.warning("市场流动性不足，请谨慎交易")

        while running:
            try:
                current_orders = self.bpx.ordersQuery(self.symbol) or []
                current_order_ids: Set[str] = set()

                for order in current_orders:
                    if "id" in order:
                        current_order_ids.add(order["id"])

                logger.info(f"当前活跃订单数量: {len(current_orders)}")

                if await self.implement_stop_loss():
                    logger.warning("已触发止损，停止交易")
                    break

                if hasattr(self, "last_grid_adjust_time"):
                    if time.time() - self.last_grid_adjust_time > 3600:
                        await self.adjust_grid_based_on_volatility()
                        self.last_grid_adjust_time = time.time()
                else:
                    self.last_grid_adjust_time = time.time()

                if previous_order_ids and previous_order_ids != current_order_ids:
                    executed_order_ids = previous_order_ids - current_order_ids
                    if executed_order_ids:
                        logger.info(f"检测到订单执行: {len(executed_order_ids)}个订单")
                        await self.process_new_trades(executed_order_ids)

                        rebalance_tasks = []
                        for order_id in executed_order_ids:
                            rebalance_tasks.append(
                                self.rebalance_single_order(order_id)
                            )

                        await asyncio.gather(*rebalance_tasks)

                        if len(current_orders) < self.num_grids * 0.7:
                            logger.warning(
                                f"订单数量明显不足，执行完全重新平衡: 当前{len(current_orders)}个，预期{self.num_grids}个"
                            )
                            await self.place_grid_orders()

                previous_order_ids = current_order_ids

                try:
                    base_available, quote_available = await self.get_account_balance()
                    logger.info(
                        f"当前余额: {base_available} {self.base_currency}, {quote_available} {self.quote_currency}"
                    )

                    current_price = await self.get_market_price()
                    total_value = base_available * current_price + quote_available
                    logger.info(
                        f"当前总资产价值: {total_value:.2f} {self.quote_currency}"
                    )
                except Exception as e:
                    logger.error(f"获取账户余额失败: {e}")

                monitor_interval = int(os.getenv("MONITOR_INTERVAL_SECONDS", "30"))
                await asyncio.sleep(monitor_interval)

            except Exception as e:
                logger.error(f"监控过程中出错: {e}")
                await asyncio.sleep(10)

    async def check_price_deviation(self) -> bool:
        """检查当前价格是否偏离网格范围过大"""
        try:
            current_price = await self.get_market_price()
            if (
                current_price < self.lower_price * 0.9
                or current_price > self.upper_price * 1.1
            ):
                logger.warning(
                    f"价格偏离网格范围过大: 当前价格{current_price}，网格范围{self.lower_price}-{self.upper_price}"
                )
                return True
            return False
        except Exception as e:
            logger.error(f"检查价格偏离失败: {e}")
            return False

    async def adjust_grid_range(self) -> None:
        """根据当前市场价格调整网格范围"""
        try:
            current_price = await self.get_market_price()

            range_percent = (self.upper_price - self.lower_price) / self.lower_price

            self.lower_price = round(
                current_price / (1 + range_percent / 2), self.pair_accuracy
            )
            self.upper_price = round(
                current_price * (1 + range_percent / 2), self.pair_accuracy
            )

            self.grid_prices = self.create_grid()

            logger.info(f"已调整网格范围: {self.lower_price}-{self.upper_price}")
        except Exception as e:
            logger.error(f"调整网格范围失败: {e}")

    def calculate_grid_profit(self, buy_price: float, sell_price: float) -> float:
        """计算网格利润"""

        quantity = self.calculate_quantity(buy_price)
        profit = quantity * (sell_price - buy_price)

        fee = (
            quantity * buy_price * self.fee_rate + quantity * sell_price * self.fee_rate
        )
        return profit - fee

    async def optimize_initial_allocation(self) -> None:
        """优化初始资产分配"""
        try:
            current_price = await self.get_market_price()

            base_available, quote_available = await self.get_account_balance()
            total_value = base_available * current_price + quote_available

            ideal_base_value = total_value / 2
            ideal_base_amount = ideal_base_value / current_price
            ideal_quote_amount = total_value / 2

            base_diff = ideal_base_amount - base_available

            if abs(base_diff) > 0.01 * ideal_base_amount:  # 如果差异超过1%
                logger.info("需要调整资产分配，当前分配不均衡")
                logger.info(f"当前市场价格: {current_price}")
                logger.info(
                    f"理想基础币数量: {ideal_base_amount}, 当前: {base_available}, 差额: {base_diff}"
                )

                if base_diff > 0:
                    buy_amount = round(base_diff, 2)  # 四舍五入到2位小数
                    if buy_amount > 0:
                        max_attempts = 60  # 最多尝试60次
                        attempt = 0
                        active_order_id = None

                        while attempt < max_attempts:
                            if active_order_id:
                                try:
                                    self.bpx.orderCancel(self.symbol, active_order_id)
                                    logger.info(
                                        f"已取消未成交的买入订单: {active_order_id}"
                                    )
                                    active_order_id = None
                                except Exception as e:
                                    logger.error(f"取消订单失败: {e}")

                            market_depth = Depth(self.symbol)
                            ask_price = float(market_depth["asks"][0][0])

                            buy_price = round(ask_price * 0.999, self.pair_accuracy)

                            logger.info(
                                f"尝试限价买入 (第{attempt + 1}次): {buy_amount} {self.base_currency} @ {buy_price}"
                            )

                            try:
                                response = self.bpx.ExeOrder(
                                    symbol=self.symbol,
                                    side="Bid",
                                    orderType="Limit",
                                    timeInForce="GTC",
                                    quantity=buy_amount,
                                    price=buy_price,
                                )

                                if (
                                    response
                                    and isinstance(response, dict)
                                    and "id" in response
                                ):
                                    active_order_id = response["id"]
                                    logger.info(f"买入订单已提交: {active_order_id}")

                                    await asyncio.sleep(1)

                                    orders = self.bpx.ordersQuery(self.symbol)
                                    if not any(
                                        order.get("id") == active_order_id
                                        for order in (orders or [])
                                    ):
                                        logger.info("买入订单已成交")
                                        break
                                else:
                                    logger.warning(f"买入订单提交失败: {response}")
                            except Exception as e:
                                logger.error(f"买入操作失败: {e}")

                            attempt += 1

                            await asyncio.sleep(1)

                        if active_order_id:
                            try:
                                self.bpx.orderCancel(self.symbol, active_order_id)
                                logger.info(
                                    f"已取消未成交的买入订单: {active_order_id}"
                                )
                            except Exception as e:
                                logger.error(f"取消订单失败: {e}")

                            logger.warning("无法在预期时间内完成买入操作")
                else:
                    sell_amount = round(abs(base_diff), 2)  # 四舍五入到2位小数
                    if sell_amount > 0:
                        max_attempts = 60  # 最多尝试60次
                        attempt = 0
                        active_order_id = None

                        while attempt < max_attempts:
                            if active_order_id:
                                try:
                                    self.bpx.orderCancel(self.symbol, active_order_id)
                                    logger.info(
                                        f"已取消未成交的卖出订单: {active_order_id}"
                                    )
                                    active_order_id = None
                                except Exception as e:
                                    logger.error(f"取消订单失败: {e}")

                            market_depth = Depth(self.symbol)
                            bid_price = float(market_depth["bids"][0][0])

                            sell_price = round(bid_price * 1.001, self.pair_accuracy)

                            logger.info(
                                f"尝试限价卖出 (第{attempt + 1}次): {sell_amount} {self.base_currency} @ {sell_price}"
                            )

                            try:
                                response = self.bpx.ExeOrder(
                                    symbol=self.symbol,
                                    side="Ask",
                                    orderType="Limit",
                                    timeInForce="GTC",
                                    quantity=sell_amount,
                                    price=sell_price,
                                )

                                if (
                                    response
                                    and isinstance(response, dict)
                                    and "id" in response
                                ):
                                    active_order_id = response["id"]
                                    logger.info(f"卖出订单已提交: {active_order_id}")

                                    await asyncio.sleep(1)

                                    orders = self.bpx.ordersQuery(self.symbol)
                                    if not any(
                                        order.get("id") == active_order_id
                                        for order in (orders or [])
                                    ):
                                        logger.info("卖出订单已成交")
                                        break
                                else:
                                    logger.warning(f"卖出订单提交失败: {response}")
                            except Exception as e:
                                logger.error(f"卖出操作失败: {e}")

                            attempt += 1

                            await asyncio.sleep(1)

                        if active_order_id:
                            try:
                                self.bpx.orderCancel(self.symbol, active_order_id)
                                logger.info(
                                    f"已取消未成交的卖出订单: {active_order_id}"
                                )
                            except Exception as e:
                                logger.error(f"取消订单失败: {e}")

                            logger.warning("无法在预期时间内完成卖出操作")

                new_base, new_quote = await self.get_account_balance()
                logger.info(
                    f"资产调整后余额: {new_base} {self.base_currency}, {new_quote} {self.quote_currency}"
                )

                current_price = await self.get_market_price()
                new_base_value = new_base * current_price
                new_total_value = new_base_value + new_quote
                base_percentage = (new_base_value / new_total_value) * 100
                quote_percentage = (new_quote / new_total_value) * 100

                logger.info(
                    f"资产分配比例: {base_percentage:.2f}% {self.base_currency}, {quote_percentage:.2f}% {self.quote_currency}"
                )
            else:
                logger.info("资产分配已经平衡，无需调整")
        except Exception as e:
            logger.error(f"优化资产分配失败: {e}")

    async def check_market_liquidity(self) -> bool:
        """检查市场流动性"""
        try:
            market_depth = Depth(self.symbol)

            bid_volume = sum(float(bid[1]) for bid in market_depth["bids"][:5])
            ask_volume = sum(float(ask[1]) for ask in market_depth["asks"][:5])

            avg_order_size = self.quantity_per_grid

            if avg_order_size > bid_volume * 0.1 or avg_order_size > ask_volume * 0.1:
                logger.warning(
                    f"市场流动性不足，可能造成滑点。平均订单大小: {avg_order_size}, 买盘深度: {bid_volume}, 卖盘深度: {ask_volume}"
                )
                return False
            return True
        except Exception as e:
            logger.error(f"检查市场流动性失败: {e}")
            return False

    async def implement_stop_loss(self) -> bool:
        """实现止损功能"""
        try:
            current_price = await self.get_market_price()

            stop_loss_percent = float(os.getenv("STOP_LOSS_PERCENT", "15"))
            initial_price = getattr(self, "initial_price", current_price)

            if current_price < initial_price * (1 - stop_loss_percent / 100):
                logger.warning(
                    f"触发止损: 当前价格 {current_price} 低于初始价格 {initial_price} 的 {stop_loss_percent}%"
                )

                await self.cancel_all_orders()
                return True
            return False
        except Exception as e:
            logger.error(f"检查止损失败: {e}")
            return False

    async def adjust_grid_based_on_volatility(self) -> None:
        """根据市场波动性调整网格间距"""
        try:
            if not hasattr(self, "price_history"):
                self.price_history = []

            current_price = await self.get_market_price()
            self.price_history.append(current_price)

            if len(self.price_history) > 20:
                self.price_history.pop(0)

            if len(self.price_history) >= 10:
                sma = sum(self.price_history) / len(self.price_history)

                volatility = sum(abs(p - sma) / sma for p in self.price_history) / len(
                    self.price_history
                )

                if volatility > 0.02:
                    new_num_grids = max(3, self.num_grids - 2)
                    logger.info(
                        f"检测到高波动性({volatility:.4f})，调整网格数量: {self.num_grids} -> {new_num_grids}"
                    )
                    self.num_grids = new_num_grids
                elif volatility < 0.005:
                    new_num_grids = min(20, self.num_grids + 2)
                    logger.info(
                        f"检测到低波动性({volatility:.4f})，调整网格数量: {self.num_grids} -> {new_num_grids}"
                    )
                    self.num_grids = new_num_grids

                self.grid_prices = self.create_grid()
                self.quantity_per_grid = self.total_investment / self.num_grids
        except Exception as e:
            logger.error(f"调整网格间距失败: {e}")

    async def run(self) -> None:
        """运行网格交易机器人"""
        try:
            base_available, quote_available = await self.get_account_balance()
            logger.info(
                f"初始余额: {base_available} {self.base_currency}, {quote_available} {self.quote_currency}"
            )

            current_price = await self.get_market_price()
            logger.info(f"当前市场价格: {current_price}")
            logger.info("优化初始资产分配...")
            await self.optimize_initial_allocation()

            if current_price < self.lower_price or current_price > self.upper_price:
                logger.warning(
                    f"当前价格({current_price})不在网格范围内({self.lower_price}-{self.upper_price})，自动调整网格范围"
                )
                await self.adjust_grid_range()

            total_value_in_quote = base_available * current_price + quote_available
            if total_value_in_quote < self.total_investment * 0.5:
                logger.warning(
                    f"账户余额不足: 当前价值{total_value_in_quote:.2f} {self.quote_currency}，所需{self.total_investment} {self.quote_currency}"
                )

                self.total_investment = total_value_in_quote * 0.95
                self.quantity_per_grid = self.total_investment / self.num_grids
                logger.info(
                    f"已调整投资金额为: {self.total_investment:.2f} {self.quote_currency}"
                )

            logger.info("开始放置网格订单...")
            await self.place_grid_orders()

            logger.info("开始监控订单执行情况...")
            await self.monitor_and_rebalance()

        except Exception as e:
            logger.error(f"运行网格交易机器人失败: {e}")
            import traceback

            traceback.print_exc()


def signal_handler(sig, frame):
    """处理退出信号"""
    global running
    logger.info("接收到退出信号，正在退出...")
    running = False

    sys.exit(0)


async def main():
    """主函数"""

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    load_dotenv()

    api_key = os.getenv("API_KEY")
    api_secret = os.getenv("API_SECRET")

    if not api_key or not api_secret:
        logger.error("错误: API密钥未正确加载，请检查.env文件")
        sys.exit(1)

    symbol = os.getenv("TRADING_PAIR", "SOL_USDC")
    pair_accuracy = int(os.getenv("PAIR_ACCURACY", "2"))

    try:
        bpx_temp = BpxClient()
        bpx_temp.init(api_key=api_key, api_secret=api_secret)

        test_balance = bpx_temp.balances()
        if not test_balance:
            logger.error("错误: 无法获取账户余额，API连接失败")
            sys.exit(1)

        market_depth = Depth(symbol)
        current_price = round(
            (float(market_depth["asks"][0][0]) + float(market_depth["bids"][-1][0]))
            / 2,
            pair_accuracy,
        )
        logger.info(f"当前市场价格: {current_price}")

        grid_range_percent = float(os.getenv("GRID_RANGE_PERCENT", "5"))
        lower_price = round(
            current_price * (1 - grid_range_percent / 100), pair_accuracy
        )
        upper_price = round(
            current_price * (1 + grid_range_percent / 100), pair_accuracy
        )

    except Exception as e:
        logger.error(f"获取市场价格失败: {e}")

        lower_price = float(os.getenv("GRID_LOWER_PRICE", "60"))
        upper_price = float(os.getenv("GRID_UPPER_PRICE", "65"))

    num_grids = int(os.getenv("GRID_NUM", "5"))
    total_investment = float(os.getenv("GRID_INVESTMENT", "100"))
    fee_rate = float(os.getenv("FEE_RATE", "0.0008"))

    logger.info(f"交易对: {symbol}")
    logger.info(f"网格范围: {lower_price}-{upper_price}")
    logger.info(f"网格数量: {num_grids}")
    logger.info(f"投资金额: {total_investment}")
    logger.info(f"手续费率: {fee_rate * 100}%")

    bot = GridTradingBot(
        api_key=api_key,
        api_secret=api_secret,
        symbol=symbol,
        lower_price=lower_price,
        upper_price=upper_price,
        num_grids=num_grids,
        total_investment=total_investment,
        pair_accuracy=pair_accuracy,
        fee_rate=fee_rate,
    )

    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())
