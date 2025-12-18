import yfinance as yf
import pandas as pd
import threading
import time
import argparse
import websockets.exceptions
import json
import datetime
from typing import Any, Union

# 문자열 객체 날짜로 변환
def parse_date(s):
    try:
        return datetime.datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        msg = f"Not a valid date: '{s}'"
        raise argparse.ArgumentTypeError(msg)

class StockWebSocket:
    def __init__(self, tickers: Union[str, list[str]]): # 생성자
        if isinstance(tickers, str):
            tickers = [tickers]
        self.tickers = tickers
        self.ws = None
        self._thread = None #스레드 저장변수

    def _on_message(self, message):
        try:
            # 1. 입력 타입에 따라 데이터 준비
            if isinstance(message, str):
                data = json.loads(message) # 웹소켓은 문자열로 옴 -> 파싱
            else:
                data = message # 폴링은 이미 딕셔너리로 줌 -> 그대로 사용

            # 2. 필요한 정보 추출 (웹소켓 데이터 구조인 'id'와 'price' 기준)
            ticker = data.get('id')
            price = data.get('price')

            # 3. 가격 정보가 있을 때만 출력
            if price is not None:
                clean_output = {
                    "id": ticker,
                    "price": price
                }
                print(json.dumps(clean_output, ensure_ascii=False))
                
        except Exception as e:
            # print({"error": f"Message processing error: {str(e)}"}) 파싱 에러 시
            pass

    def start(self): #웹소켓 시작
        if self._thread is not None and self._thread.is_alive():
            return

        # run_listener 함수가 웹소켓 실패 시 폴링으로 대체하도록 변경
        def run_listener():
            try:
                # 1. 웹소켓 연결 시도
                self.ws = yf.WebSocket()
                self.ws.subscribe(self.tickers)

                # 디버그 메세지 출력
                print(f"[DEBUG] WebSocket connected. Listening for {self.tickers}")

                # 2. listen()은 블로킹(blocking) 메소드 (정상 작동 시 여기서 계속 대기)
                self.ws.listen(self._on_message)

            # 정상 종료
            except websockets.exceptions.ConnectionClosedOK:
                print("[DEBUG] WebSocket connection closed cleanly.")
                return
            
            # 연결 실패
            except websockets.exceptions.ConnectionClosedError as e:
                print(f"[DEBUG] WebSocket connection lost ({e}). Falling back to polling.")
                pass

            # 그 외 모든 오류
            except Exception as e:
                print(f"[DEBUG] WebSocket failed ({e}). Falling back to 10-second polling for {self.tickers}.")

            # --- 4. 폴링 루프 ---
            try:
                while True:
                    # 5. 'latest' 가격 조회 함수(get_latest_price) 호출
                    latest_prices_data = get_latest_price(self.tickers)

                    for ticker, data in latest_prices_data.items():
                        if "price" in data:
                            simulated_message = {
                                "id": ticker,
                                "price": data["price"]
                            }
                            # 6. 출력
                            self._on_message(simulated_message)
                        else:
                            print(f"[DEBUG] Polling error for {ticker}: {data.get('error')}")
                    # 7. 10초 대기
                    time.sleep(10)

            except (KeyboardInterrupt, SystemExit):
                pass # Ctrl+C -> 종료
            except Exception as poll_e:
                pass # 오류로 인한 폴링 루프 종료

            finally:
                if self.ws:
                    self.ws.close()
                print(f"[DEBUG] Listener thread for {self.tickers} stopped.")

        self._thread = threading.Thread(target=run_listener, daemon=True)
        self._thread.start()

    def stop(self): #연결 종료
        if self.ws:
            self.ws.close()
        if self._thread is not None and self._thread.is_alive():
             self._thread.join(timeout=1) # 스레드가 종료될 때까지 잠시 대기
        print("Websocket service disconnected")

# 현재가
def get_latest_price(tickers: Union[str, list[str]]) -> dict[str, dict[str, Any]]:
    if isinstance(tickers, str):
        tickers = [tickers]
    output: dict[str, dict[str, Any]] = {}
    for ticker in tickers:
        try:
            stock_info = yf.Ticker(ticker).fast_info
            price = stock_info.get("lastPrice")
            if price is not None:
                output[ticker] = {"price": round(float(price), 2)}
            else:
                # [수정] lastPrice가 없으면 closing price(previousClose)라도 가져오기
                price = stock_info.get("previousClose")
                if price is not None:
                     output[ticker] = {"price": round(float(price), 2)}
                else:
                    output[ticker] = {"error": "Price not available"}
        except Exception as e:
            output[ticker] = {"error": "Invalid Ticker or Network Error"}
    return output

# 과거기록_dictioinary 반환
def get_historical_price_data(tickers: Union[str, list[str]], period: str = "1mo", interval: str = "1d", num: int = 0) -> dict[str, dict[str, Any]]:
    if isinstance(tickers, str):
        tickers = [tickers]
    output: dict[str, dict[str, Any]] = {ticker: {} for ticker in tickers}
    short_interval = ["1m", "5m", "15m", "30m", "60m", "90m", "1h"]
    Time = "TimeStamp" if interval in short_interval else "Datetime"
    try:
        hist_df_multi = yf.download(
            tickers=tickers,
            period=period,
            interval=interval,
            auto_adjust=False,
            group_by='ticker',
            progress=False
        )
        for ticker in tickers:
            output_data = {"history": None}
            if isinstance(hist_df_multi.columns, pd.MultiIndex) and ticker in hist_df_multi.columns.levels[0]:
                hist_df = hist_df_multi[ticker]
            elif not isinstance(hist_df_multi.columns, pd.MultiIndex) and len(tickers) == 1:
                hist_df = hist_df_multi
            else:
                output[ticker] = {"error": "Failed to download data for this ticker.", "history": None}
                continue
            if hist_df.empty or hist_df.isnull().all().all():
                output_data["error"] = f"No data found for ticker {ticker}."
            else:
                hist_df_processing = hist_df.iloc[-num:].copy() if num is not None and num > 0 else hist_df.copy()
                hist_df_processing.reset_index(inplace=True)
                hist_df_processing.rename(columns={hist_df_processing.columns[0]: Time}, inplace=True)
                hist_df_processing["Time"] = hist_df_processing[Time].apply(lambda x: x.isoformat())
                price_cols_to_round = ['High', 'Low', 'Close', 'Open']
                for col in price_cols_to_round:
                    if col in hist_df_processing.columns and pd.api.types.is_numeric_dtype(hist_df_processing[col]):
                        hist_df_processing[col] = hist_df_processing[col].round(2)
                required_data_columns = ['Time', 'High', 'Low', 'Close', 'Open']
                final_columns = [col for col in required_data_columns if col in hist_df_processing.columns]
                if final_columns:
                    output_data["history"] = hist_df_processing[final_columns].to_dict(orient='list')
                    output_data["history"].update({"period": period, "interval": interval})
                else:
                    output_data["error"] = f"No usable data columns found for {ticker}."
            output[ticker] = output_data
    except Exception as e:
        for ticker in tickers:
            if ticker not in output:
                output[ticker] = {"error": str(e), "history": None}
    return output

# 과거기록_dataframe 반환
def get_history_df(ticker, start, end, num):
    if start is None:
        start = "2022-01-01"  # 시작 날짜 기본값
    if end is None:
        end = datetime.date.today().strftime("%Y-%m-%d") # 종료 날짜 기본값 (오늘)
    
    try:
        history_df =  yf.download(
            tickers=ticker,
            start=start,
            end=end,
            auto_adjust=False,
            group_by='column',
            progress=False
        )
        if history_df.empty:
            print(f"Warning: {ticker} 데이터가 비어있습니다.")
            return None

        if num > 0:
            if len(history_df) > num:
                history_df = history_df.iloc[-num:] 
        return history_df

    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

# 펀더멘탈
def get_fundamental_data(tickers: Union[str, list[str]]) -> dict[str, dict[str, Any]]:
    if isinstance(tickers, str):
        tickers = [tickers]
    output: dict[str, dict[str, Any]] = {}
    extract_data = {"shortName", "marketCap", "totalRevenue", "grossProfits",
                    "netIncomeToCommon", "trailingPE", "trailingEps", "ebitda",
                    "currency"}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            stock_info = stock.info
            filtered_data = {
                key: stock_info.get(key, None) for key in extract_data
            }
            output[ticker] = {"fundamental": filtered_data}
        except Exception as e:
            output[ticker] = {"error": str(e)}
    return output

# 메인
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch stock data from Yahoo Finance.")
    parser.add_argument(
        "data_type",
        type=str,
        choices=["latest", "history", "history_df", "fundamental", "websocket"],
        help="Type of data to fetch: 'latest' price, 'history' , or 'fundamental' info."
    )
    parser.add_argument(
        "--tickers",
        type=str,
        required=True,
        help="Comma-separated stock ticker symbols (e.g., AAPL,MSFT,005930.KS)"
    )
    # ... (parser.add_argument for period, interval, num remains the same) ...
    parser.add_argument(
        "--period",
        type=str,
        default="1mo",
        help="Period for historical data (e.g., 1d, 5d, 1mo, 1y, max). Used only with 'history' type."
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="1d",
        help="Data interval (e.g., 1m, 5m, 1h, 1d, 1wk, 1mo). Used only with 'history' type."
    )
    parser.add_argument(
        "--num",
        type=int,
        default=0,
        help="Fetch data size"
    )
    parser.add_argument(
        "--start",
        type=parse_date, # 날짜 객체
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end",
        type=parse_date, # 날짜 객체
        help="End date (YYYY-MM-DD)"
    )

    args = parser.parse_args()
    tickers_list = [ticker.strip() for ticker in args.tickers.split(',')]
    final_output = {}
    try:
        if args.data_type == "websocket":
            if not tickers_list:
                print({"error": "No tickers provided for websocket."})
                exit(1)
            ws = StockWebSocket(tickers_list)
            try:
                ws.start()
                print("Press CTRL+C to disconnect websocket")
                while True:
                    time.sleep(10)
            except KeyboardInterrupt:
                pass # 종료
            except Exception as e:
                print({"error": f"Websocket main loop error: {str(e)}"})
            finally:
                ws.stop()
        else:
            final_output = {}
            if args.data_type == "latest":
                final_output = get_latest_price(tickers_list)
            elif args.data_type == "history":
                final_output = get_historical_price_data(tickers_list, period=args.period, interval=args.interval, num=args.num)
            elif args.data_type == "fundamental":
                final_output = get_fundamental_data(tickers_list)
            elif args.data_type == "history_df":
                result_df = get_history_df(tickers_list, start=args.start, end=args.end, num=args.num)
                # 2. 결과가 진짜 DataFrame인지, 그리고 비어있지 않은지 확인합니다.
                if isinstance(result_df, pd.DataFrame) and not result_df.empty:
                    filename = "_".join(tickers_list) + ".csv"
                    result_df.to_csv(filename)
                    final_output = {"success": f"Saved to {tickers_list}.csv", "rows": len(result_df)}
                else:
                    # 데이터가 없거나 에러가 난 경우
                    print({"error": "Failed to fetch dataframe. Result is None or empty."})
            else:
                final_output = {"error": "Invalid data_type specified."}
            print(final_output)
    except Exception as e:
        final_output = {
            "error": f"An unexpected error occurred in main execution: {str(e)}",
            "tickers": tickers_list,
        }
        print(final_output)