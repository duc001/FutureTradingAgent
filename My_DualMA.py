# encoding utf-8
import tbpy
import sys
import datetime

class My_DualMA(tbpy.IStrategy):
    def __init__(self, fast_length, slow_length, account_id, symbol):
        super().__init__('My_DualMA')
        self._fast_length = fast_length
        self._slow_length = slow_length
        self._account_id = account_id
        self._symbol = symbol
        self._account = None
        self._tick = None
        self._pos = None
        self._order_dict = {}
        self._timer_id = 0
        pass

    def __del__(self):
        pass

    def on_init(self, context):
        ret = context.subscribe_tick(symbol=self._symbol)
        if ret is not None:
            print(ret)
            tbpy.exit()
        ret = context.subscribe_bar(symbol=self._symbol, frequency='10s', begin_time=datetime.datetime.now()-datetime.timedelta(minutes=self._slow_length), sliding_window=self._slow_length)
        if ret is not None:
            print(ret)
            tbpy.exit()
        self._account = context.subscribe_account(account_id=self._account_id)
        if self._account is None:
            print(tbpy.get_last_err())
            tbpy.exit()
        self._timer_id = context.create_timer(interval_millsecs=5000)
        self._tick = tbpy.get_current_tick(symbol=self._symbol)
        self._pos = self._account.get_position(symbol=self._symbol)
        print('on_init success.')

    def on_bar(self, context, bars, symbol, flag):
        print(bars[len(bars)-1])
        # flag=0 历史数据；flag=1 实时数据
        if flag != 1:
            return
        if self._tick is None:
            return
        if self._account.get_status() != tbpy.AccountStatus.OnService:
            return
        fast_avg = sum(bar.close for bar in bars[(-self._fast_length - 1):-1]) / self._fast_length
        slow_avg = sum(bar.close for bar in bars[(-self._slow_length - 1):-1]) / self._slow_length
        if fast_avg > slow_avg:
            if self._pos is None:
                self.push_order_id(self._account.buy(symbol=symbol, volume=1, price=self._tick.last))
            else:
                if self._pos.s_can_cover_volume > 0:
                    self.push_order_id(self._account.buy2cover(symbol=symbol, volume=self._pos.s_can_cover_volume, price=self._tick.last))
                if self._pos.l_current_volume + self._pos.l_active_volume + self._pos.l_active_close_volume < 2:
                    self.push_order_id(self._account.buy(symbol=symbol, volume=1, price=self._tick.last))
        elif fast_avg < slow_avg:
            if self._pos is None:
                self.push_order_id(self._account.sell2short(symbol=symbol, volume=1, price=self._tick.last))
            else:
                if self._pos.l_can_sell_volume > 0:
                    self.push_order_id(self._account.sell(symbol=symbol, volume=self._pos.l_can_sell_volume, price=self._tick.last))
                if self._pos.s_current_volume + self._pos.s_active_volume + self._pos.s_active_close_volume < 2:
                    self.push_order_id(self._account.sell2short(symbol=symbol, volume=1, price=self._tick.last))

    def on_tick(self, context, tick):
        # print(tick)
        self._tick = tick

    def on_position(self, context, pos):
        print(pos)
        self._pos = pos

    def on_order(self, context, order):
        print(order)
        if order.status == tbpy.OrderStatus.NewReject or order.status == tbpy.OrderStatus.AllFill or \
                        order.status == tbpy.OrderStatus.Canceled or order.status == tbpy.OrderStatus.CanceledFill:
            self._order_dict.pop(order.order_id)

    def on_fill(self, context, fill):
        print(fill)
        pass

    def on_timer(self, context, id, millsecs):
        now_time = datetime.datetime.now()
        for key, value in self._order_dict.items():
            if (now_time - value).seconds >= 10:
                self._account.cancel_order(order_id=key)

    def push_order_id(self, order_id_list):
        send_time = datetime.datetime.now()
        for id in order_id_list:
            self._order_dict[id] = send_time

if __name__ == '__main__':
    ret = tbpy.init('tbquant3')
    if ret is False:
        print('init fail.')
        sys.exit()

    rb_main = tbpy.get_main_instrument(underlying_symbol='rb.SHFE')
    if rb_main is None:
        sys.exit()
    strategy = My_DualMA(5, 10, '66113172', rb_main.symbol)
    tbpy.exe()

