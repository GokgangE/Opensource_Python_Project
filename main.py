import sys
import pandas as pd
import pyqtgraph as pg
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QTableWidget, 
                             QTableWidgetItem, QHeaderView, QInputDialog, 
                             QMessageBox, QSpinBox, QGroupBox, QGridLayout)
from PyQt6.QtCore import Qt, QObject, pyqtSignal, QThread, pyqtSlot, QRectF
from PyQt6.QtGui import QColor, QPicture, QPainter

# ★ stock_api.py 필요
import stock_api

# =============================================================================
# [1] 데이터 관리 & 백엔드 클래스들 (Portfolio, Client, Loader)
# =============================================================================

class Portfolio:
    """사용자 자산 및 보유 주식 관리 클래스"""
    def __init__(self):
        self.cash = 10000000.0  # 예수금 (초기 천만원)
        self.holdings = {}      # { 'AAPL': {'qty': 10, 'avg': 150.0}, ... }

    @property
    def total_invested(self):
        """총 매수 금액"""
        return sum(h['qty'] * h['avg'] for h in self.holdings.values())

    def get_valuation(self, current_prices):
        """총 평가 금액 (현재가 기준)"""
        stock_val = 0
        for ticker, info in self.holdings.items():
            # 현재가가 없으면 평단가로 계산
            price = current_prices.get(ticker, info['avg'])
            stock_val += price * info['qty']
        return stock_val

    def buy(self, ticker, price, qty):
        cost = price * qty
        if cost > self.cash:
            return False, "예수금이 부족합니다."
        
        self.cash -= cost
        
        if ticker in self.holdings:
            old = self.holdings[ticker]
            # 평단가 갱신
            new_avg = ((old['qty'] * old['avg']) + cost) / (old['qty'] + qty)
            old['qty'] += qty
            old['avg'] = new_avg
        else:
            self.holdings[ticker] = {'qty': qty, 'avg': price}
        return True, "매수 체결 완료"

    def sell(self, ticker, price, qty):
        if ticker not in self.holdings or self.holdings[ticker]['qty'] < qty:
            return False, "보유 수량이 부족합니다."
        
        earnings = price * qty
        self.cash += earnings
        
        self.holdings[ticker]['qty'] -= qty
        if self.holdings[ticker]['qty'] == 0:
            del self.holdings[ticker]
        return True, "매도 체결 완료"


class QtStockClient(stock_api.StockWebSocket, QObject):
    """실시간 가격 수신용 (매매 버튼 활성화 핵심)"""
    data_received_signal = pyqtSignal(dict)

    def __init__(self, tickers):
        QObject.__init__(self)
        stock_api.StockWebSocket.__init__(self, tickers)

    def _on_message(self, message):
        try:
            if isinstance(message, str):
                import json
                data = json.loads(message)
            else:
                data = message
            
            ticker = data.get('id')
            price = data.get('price')

            if price is not None:
                self.data_received_signal.emit({"id": ticker, "price": price})
        except Exception:
            pass


class DataLoader(QThread):
    """차트 및 재무정보 로딩용 (비동기)"""
    data_loaded = pyqtSignal(dict) 

    def __init__(self, ticker, data_type, **kwargs):
        super().__init__()
        self.ticker = ticker
        self.data_type = data_type
        self.kwargs = kwargs 

    def run(self):
        result = {}
        try:
            if self.data_type == "history":
                raw_data = stock_api.get_historical_price_data(self.ticker, **self.kwargs)
                result = {"type": "history", "data": raw_data}
            elif self.data_type == "fundamental":
                raw_data = stock_api.get_fundamental_data(self.ticker)
                result = {"type": "fundamental", "data": raw_data}
        except Exception as e:
            result = {"error": str(e)}
        self.data_loaded.emit(result)


class CandlestickItem(pg.GraphicsObject):
    """캔들 차트 아이템 (수정된 버전)"""
    def __init__(self, data):
        pg.GraphicsObject.__init__(self)
        self.data = data  # [(time, open, close, low, high), ...]
        self.generatePicture()

    def generatePicture(self):
        self.picture = QPicture()
        p = QPainter(self.picture)
        p.setPen(pg.mkPen('w')) 
        
        if not self.data:
            p.end()
            return

        if len(self.data) > 1:
            times = [d[0] for d in self.data]
            gaps = [(times[i+1] - times[i]) for i in range(len(times)-1)]
            if gaps:
                min_gap = min(gaps) 
                w = min_gap * 0.4 
            else:
                w = 1.0 
        else:
            w = 1.0 

        for (t, open, close, low, high) in self.data:
            if open > close: # 하락 (파랑)
                p.setBrush(pg.mkBrush((0, 0, 255)))
                p.setPen(pg.mkPen((0, 0, 255)))
            else: # 상승 (빨강)
                p.setBrush(pg.mkBrush((255, 0, 0)))
                p.setPen(pg.mkPen((255, 0, 0)))
            
            p.drawLine(int(t), int(low), int(t), int(high))
            p.drawRect(QRectF(t - w, open, w * 2, close - open))
        p.end()

    def paint(self, p, *args):
        self.picture.play(p)

    def boundingRect(self):
        return QRectF(self.picture.boundingRect())

# =============================================================================
# [2] 메인 GUI (모든 기능 통합)
# =============================================================================
class TradingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Python AI Trader - Integrated System")
        self.resize(1600, 900)

        # 1. 데이터 관리 객체 초기화
        self.portfolio = Portfolio()      # 포트폴리오 (매수/매도 로직)
        self.current_prices = {}          # 현재가 저장소
        self.target_ticker = None         # 현재 선택된 종목
        self.client = None                # 실시간 시세 수신 클라이언트

        # 2. UI 레이아웃 설정
        central = QWidget()
        self.setCentralWidget(central)
        self.main_layout = QHBoxLayout(central)

        self.init_chart_panel()       # 좌측: 차트
        self.init_fundamental_panel() # 중앙: 재무정보
        self.init_user_panel()        # 우측: 주문 및 잔고

        self.main_layout.setStretch(0, 5)
        self.main_layout.setStretch(1, 2)
        self.main_layout.setStretch(2, 3)

        # 초기 대시보드 갱신
        self.update_dashboard()

    def init_chart_panel(self):
        panel = QWidget()
        panel.setStyleSheet("background-color: #121212;")
        layout = QVBoxLayout(panel)

        self.lbl_ticker = QLabel("종목을 검색해주세요")
        self.lbl_ticker.setStyleSheet("color: white; font-size: 24px; font-weight: bold;")
        layout.addWidget(self.lbl_ticker)

        # 기간 설정 버튼
        btn_layout = QHBoxLayout()
        periods = [("1일 (5분봉)", "1d", "5m"), ("3개월 (일봉)", "3mo", "1d"), ("10년 (월봉)", "10y", "1mo")]
        for name, p, i in periods:
            btn = QPushButton(name)
            btn.setStyleSheet("background-color: #333; color: white;")
            btn.clicked.connect(lambda _, p=p, i=i: self.load_history_data(p, i))
            btn_layout.addWidget(btn)
        layout.addLayout(btn_layout)

        self.chart_widget = pg.PlotWidget()
        self.chart_widget.setBackground('#121212')
        self.chart_widget.showGrid(x=True, y=True, alpha=0.3)
        self.date_axis = pg.DateAxisItem(orientation='bottom')
        self.chart_widget.setAxisItems({'bottom': self.date_axis})
        layout.addWidget(self.chart_widget)
        self.main_layout.addWidget(panel)

    def init_fundamental_panel(self):
        panel = QWidget()
        panel.setStyleSheet("background-color: #1e1e1e; border-right: 1px solid #444;")
        layout = QVBoxLayout(panel)
        
        title = QLabel("🏢 기업 재무 정보")
        title.setStyleSheet("color: #FFD700; font-size: 18px; font-weight: bold; margin-bottom: 15px;")
        layout.addWidget(title)

        self.fund_labels = {}
        items = {"shortName": "기업명", "marketCap": "시가총액", "trailingPE": "PER", "trailingEps": "EPS",
                 "totalRevenue": "매출액", "grossProfits": "매출총이익", "netIncomeToCommon": "당기순이익", "ebitda": "EBITDA"}

        form_grid = QGridLayout()
        row = 0
        for key, name in items.items():
            lbl_name = QLabel(name)
            lbl_name.setStyleSheet("color: #aaa; font-weight: bold;")
            lbl_value = QLabel("-")
            lbl_value.setStyleSheet("color: white;")
            lbl_value.setWordWrap(True)
            form_grid.addWidget(lbl_name, row, 0)
            form_grid.addWidget(lbl_value, row, 1)
            self.fund_labels[key] = lbl_value 
            row += 1

        layout.addLayout(form_grid)
        layout.addStretch() 
        self.main_layout.addWidget(panel)

    def init_user_panel(self):
        panel = QWidget()
        panel.setStyleSheet("background-color: #f5f5f5;")
        layout = QVBoxLayout(panel)

        # [A] 내 계좌 현황
        grp_user = QGroupBox("👤 내 계좌 현황")
        user_layout = QGridLayout()
        
        # 라벨 변수 저장 (update_dashboard에서 쓰기 위함)
        self.val_cash = QLabel("-")
        self.val_invested = QLabel("-")
        self.val_total = QLabel("-")
        self.val_profit = QLabel("-")

        for lbl in [self.val_cash, self.val_invested, self.val_total, self.val_profit]:
            lbl.setStyleSheet("font-size: 15px; font-weight: bold; color: #333;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignRight)

        user_layout.addWidget(QLabel("예수금:"), 0, 0)
        user_layout.addWidget(self.val_cash, 0, 1)
        user_layout.addWidget(QLabel("총 매수금:"), 1, 0)
        user_layout.addWidget(self.val_invested, 1, 1)
        user_layout.addWidget(QLabel("총 자산:"), 2, 0)
        user_layout.addWidget(self.val_total, 2, 1)
        user_layout.addWidget(QLabel("수익률:"), 3, 0)
        user_layout.addWidget(self.val_profit, 3, 1)
        grp_user.setLayout(user_layout)
        layout.addWidget(grp_user)

        # [B] 주문창
        grp_order = QGroupBox("⚡ 간편 주문")
        order_layout = QVBoxLayout()
        self.spin_qty = QSpinBox()
        self.spin_qty.setRange(1, 100000)
        
        row_qty = QHBoxLayout()
        row_qty.addWidget(QLabel("수량:"))
        row_qty.addWidget(self.spin_qty)
        order_layout.addLayout(row_qty)
        
        btn_box = QHBoxLayout()
        btn_buy = QPushButton("매수 (Buy)")
        btn_buy.setStyleSheet("background-color: #ff4444; color: white; padding: 10px; font-weight: bold;")
        btn_buy.clicked.connect(lambda: self.execute_trade('buy')) # 매수 연결
        
        btn_sell = QPushButton("매도 (Sell)")
        btn_sell.setStyleSheet("background-color: #4444ff; color: white; padding: 10px; font-weight: bold;")
        btn_sell.clicked.connect(lambda: self.execute_trade('sell')) # 매도 연결

        btn_box.addWidget(btn_buy)
        btn_box.addWidget(btn_sell)
        order_layout.addLayout(btn_box)
        grp_order.setLayout(order_layout)
        layout.addWidget(grp_order)

        # [C] 보유 종목 테이블
        layout.addWidget(QLabel("보유 종목"))
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["종목", "수량", "평단가", "수익률"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table)

        # [D] 검색 버튼
        btn_search = QPushButton("🔍 종목 검색 / 변경")
        btn_search.setStyleSheet("padding: 15px; background-color: #222; color: white; font-weight: bold;")
        btn_search.clicked.connect(self.open_search_dialog)
        layout.addWidget(btn_search)

        self.main_layout.addWidget(panel)

    # =========================================================================
    # [로직] 데이터 요청 및 매매 처리
    # =========================================================================
    def open_search_dialog(self):
        text, ok = QInputDialog.getText(self, "종목 검색", "Ticker 입력 (예: TSLA, AAPL):")
        if ok and text:
            ticker = text.strip().upper()
            self.target_ticker = ticker
            self.lbl_ticker.setText(f"{ticker} 데이터 수신 중...")
            
            # 1. 과거 데이터(차트) 및 펀더멘탈 요청 (DataLoader)
            self.loader_fund = DataLoader(ticker, "fundamental")
            self.loader_fund.data_loaded.connect(self.update_fundamental_ui)
            self.loader_fund.start()
            self.load_history_data("3mo", "1d") # 기본 차트

            # 2. 실시간 시세 수신 시작 (QtStockClient) -> 이게 있어야 매매 가능!
            if self.client: self.client.stop()
            self.client = QtStockClient([ticker])
            self.client.data_received_signal.connect(self.on_realtime_data)
            self.client.start()

    def load_history_data(self, period, interval):
        if not self.target_ticker: return
        self.chart_widget.clear()
        self.loader_hist = DataLoader(self.target_ticker, "history", period=period, interval=interval)
        self.loader_hist.data_loaded.connect(self.update_chart_ui)
        self.loader_hist.start()

    @pyqtSlot(dict)
    def on_realtime_data(self, data):
        """실시간 가격 수신 -> 현재가 저장 -> 대시보드 갱신"""
        ticker = data['id']
        price = data['price']
        self.current_prices[ticker] = price # ★ 핵심: 현재가 저장
        
        # 선택된 종목이면 라벨 업데이트
        if ticker == self.target_ticker:
            self.lbl_ticker.setText(f"{ticker} : ${price:,.2f}")
        
        # 대시보드(수익률) 갱신
        self.update_dashboard()

    def execute_trade(self, action):
        """매수/매도 버튼 클릭 시 실행"""
        if not self.target_ticker or self.target_ticker not in self.current_prices:
            QMessageBox.warning(self, "주문 실패", "현재가 정보를 수신 중입니다. 잠시 후 다시 시도해주세요.")
            return

        price = self.current_prices[self.target_ticker]
        qty = self.spin_qty.value()

        if action == 'buy':
            ok, msg = self.portfolio.buy(self.target_ticker, price, qty)
        else:
            ok, msg = self.portfolio.sell(self.target_ticker, price, qty)
            
        if ok:
            QMessageBox.information(self, "체결 성공", f"{msg}\n가격: ${price}\n수량: {qty}")
            self.update_dashboard()
        else:
            QMessageBox.warning(self, "주문 거부", msg)

    def update_dashboard(self):
        """사용자 자산 정보 및 보유 종목 테이블 갱신"""
        invested = self.portfolio.total_invested
        valuation = self.portfolio.get_valuation(self.current_prices)
        total_profit = valuation - invested
        profit_rate = (total_profit / invested * 100) if invested > 0 else 0.0

        # 라벨 갱신
        self.val_cash.setText(f"${self.portfolio.cash:,.0f}")
        self.val_invested.setText(f"${invested:,.0f}")
        self.val_total.setText(f"${self.portfolio.cash + valuation:,.0f}")
        
        color = "red" if total_profit > 0 else "blue" if total_profit < 0 else "black"
        self.val_profit.setText(f"${total_profit:,.0f} ({profit_rate:+.2f}%)")
        self.val_profit.setStyleSheet(f"color: {color}; font-size: 15px; font-weight: bold;")

        # 테이블 갱신
        self.table.setRowCount(0)
        for ticker, info in self.portfolio.holdings.items():
            row = self.table.rowCount()
            self.table.insertRow(row)
            
            curr_p = self.current_prices.get(ticker, info['avg'])
            p_rate = ((curr_p - info['avg']) / info['avg']) * 100
            
            item_profit = QTableWidgetItem(f"{p_rate:+.2f}%")
            if p_rate > 0: item_profit.setForeground(Qt.GlobalColor.red)
            elif p_rate < 0: item_profit.setForeground(Qt.GlobalColor.blue)

            self.table.setItem(row, 0, QTableWidgetItem(ticker))
            self.table.setItem(row, 1, QTableWidgetItem(str(info['qty'])))
            self.table.setItem(row, 2, QTableWidgetItem(f"${info['avg']:,.2f}"))
            self.table.setItem(row, 3, item_profit)

    # (차트/재무정보 업데이트 함수는 동일함)
    @pyqtSlot(dict)
    def update_fundamental_ui(self, result):
        if "error" in result: return
        data = result['data'].get(self.target_ticker, {}).get('fundamental', {})
        if not data: return
        for key, label in self.fund_labels.items():
            val = data.get(key)
            display_text = "N/A"
            if val is not None and isinstance(val, (int, float)):
                if val > 1e12: display_text = f"{val/1e12:.2f}T"
                elif val > 1e9: display_text = f"{val/1e9:.2f}B"
                else: display_text = f"{val:,.0f}"
            elif val is not None: display_text = str(val)
            label.setText(display_text)

    @pyqtSlot(dict)
    def update_chart_ui(self, result):
        if "error" in result: return
        hist = result['data'].get(self.target_ticker, {}).get("history")
        if not hist: return

        # Timestamp 변환 (여기까지는 numpy int64 형태입니다)
        times = pd.to_datetime(hist['Time']).astype('int64') // 10**9
        
        candle_data = []
        for i in range(len(times)):
            # ★ [수정 핵심] times[i]는 numpy.int64이므로 int()로 감싸서 표준 정수형으로 변환해야 합니다.
            # 나머지 가격 데이터(Open, Close 등)는 float()로 감싸주는 것이 안전합니다.
            candle_data.append((
                int(times[i]),      # <--- 여기 수정 (int로 변환)
                float(hist['Open'][i]), 
                float(hist['Close'][i]), 
                float(hist['Low'][i]), 
                float(hist['High'][i])
            ))
        
        item = CandlestickItem(candle_data)
        self.chart_widget.addItem(item)
        
        # 줌 설정
        if len(times) > 0:
            min_x, max_x = int(times[0]), int(times[-1]) # 여기도 int 변환
            min_y, max_y = min(hist['Low']), max(hist['High'])
            view_box = self.chart_widget.getViewBox()
            view_box.setLimits(xMin=min_x, xMax=max_x, yMin=min_y*0.9, yMax=max_y*1.1)
            view_box.autoRange()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TradingApp()
    window.show()
    sys.exit(app.exec())