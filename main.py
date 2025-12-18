import numpy as np
import pandas as pd
import stock_api as api
import tensorflow as tf
import argparse
import sys
import datetime
from sklearn.preprocessing import MinMaxScaler

def model(ticker, num, start, end):
    window_size = 50
    if isinstance(ticker, list):
        ticker = ticker[0] # 여러 개의 티커를 받을 경우 하나만 처리

    # 1. 주가 데이터 가져오기
    try:
        df = api.get_history_df(ticker, start, end, num)
    except Exception as e:
        print(f"API Error: {e}")
        return

    if df is None or df.empty:
        sys.exit(1)

    # MultiIndex 컬럼 처리
    if isinstance(df.columns, pd.MultiIndex):
        target_level = None
        for i in range(df.columns.nlevels):
            level_values = df.columns.get_level_values(i)
            if 'Close' in level_values or 'Open' in level_values:
                target_level = i
                break
        if target_level is not None:
            df.columns = df.columns.get_level_values(target_level)
        else:
            df.columns = df.columns.get_level_values(-1)

    # 2. 전처리
    # (1) 이평선 추가
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    
    # (3) 결측치(NaN) 제거
    df.dropna(inplace=True)

    # 3. 데이터 분할
    feature_cols = ['Open', 'High', 'Low', 'Close', 'MA5', 'MA20']
    target_col = ['Close']

    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]

    # 4. 스케일링
    scaler_x = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))

    # Train 데이터로만 fit 수행
    scaler_x.fit(train_df[feature_cols])
    scaler_y.fit(train_df[target_col])

    # Transform은 각각 수행
    scaled_train_x = scaler_x.transform(train_df[feature_cols])
    scaled_train_y = scaler_y.transform(train_df[target_col])
    
    scaled_test_x = scaler_x.transform(test_df[feature_cols])
    scaled_test_y = scaler_y.transform(test_df[target_col])

    # 5. Sliding Window 생성을 위해 연결 (경계선 처리)
    total_scaled_x = np.vstack((scaled_train_x, scaled_test_x))
    total_scaled_y = np.vstack((scaled_train_y, scaled_test_y))

    X, y = [], []
    for i in range(len(total_scaled_x) - window_size):
        X.append(total_scaled_x[i : i + window_size])
        y.append(total_scaled_y[i + window_size])

    X, y = np.array(X), np.array(y)
    num_features = X.shape[2]

    # 다시 Train/Test 분리 (윈도우 생성 후 개수 기반)
    # 실제 학습에 쓸 데이터 개수 계산 (앞서 계산한 비율 유지)
    real_train_len = len(scaled_train_x) - window_size # Train 데이터 내부에서 만들 수 있는 윈도우 수
    if real_train_len < 0: real_train_len = 0 # 예외처리

    # 엄밀한 분리를 위해 재조정 (Train 데이터 끝부분 ~ Test 데이터 시작부분)
    split_idx = int(len(X) * 0.9)
    
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # 6. 모델 구성
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(units=64, return_sequences=True, input_shape=(window_size, num_features)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(units=64, return_sequences=False))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(units=32))
    model.add(tf.keras.layers.Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    # EarlyStopping: Test 데이터(val_loss)가 10번(patience) 동안 좋아지지 않으면 학습 중단
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1)

    # ModelCheckpoint: Test 성적이 가장 좋았을 때의 모델을 파일로 저장
    checkpoint = tf.keras.callbacks.ModelCheckpoint(f'best_model_{ticker}.keras', monitor='val_loss', save_best_only=True, verbose=1)

    # 7. 학습 (validation_data에 X_test, y_test를 넣어줌으로써 실시간 평가)
    print(f"\n[학습 시작] Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    history = model.fit(
        X_train, y_train, 
        validation_data=(X_test, y_test), # 학습 중에 계속 테스트 데이터를 훔쳐보며 성능 체크
        batch_size=32, 
        epochs=100, # 넉넉하게 100번 주되, EarlyStopping이 알아서 멈춤
        callbacks=[early_stop, checkpoint], # 콜백 등록
        verbose=1
    )

    # 8. 예측 (가장 좋았던 모델을 다시 불러와서 예측)
    best_model = tf.keras.models.load_model(f'best_model_{ticker}.keras')

    # 8. 예측 및 결과 확인
    if len(X_test) > 0:
        pred = best_model.predict(X_test)
        real_pred = scaler_y.inverse_transform(pred)
        real_actual = scaler_y.inverse_transform(y_test)
        
        # [수정] 최근 결과(마지막 5일)를 출력하도록 변경
        print("\n=== 최근 예측 결과 (마지막 5일) ===")
        # 최근 5일치만 가져오기
        recent_pred = real_pred[-5:]
        recent_actual = real_actual[-5:]
        
        for i in range(len(recent_pred)):
            print(f"D-{5-i}: 예측 {recent_pred[i][0]:.2f} / 실제 {recent_actual[i][0]:.2f}")
    
    # 9. 내일 주가 예측
    last_window = total_scaled_x[-window_size:] 
    last_window = np.reshape(last_window, (1, window_size, num_features))
    
    tomorrow_pred_scaled = best_model.predict(last_window)
    tomorrow_price = scaler_y.inverse_transform(tomorrow_pred_scaled)

    print("\n" + "="*40)
    print(f"[{ticker}] 내일 예측 종가: {tomorrow_price[0][0]:,.0f}")
    print("="*40 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stock Price Prediction AI")
    
    # 1. Ticker (필수 입력)
    parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g., 005930.KS)"
    )
    
    # 2. Start Date (선택 입력, 기본값 처리)
    parser.add_argument(
        "--start",
        type=api.parse_date, # 문자열을 날짜 객체로 변환
        default="2022-01-01",
        help="Start date (YYYY-MM-DD)"
    )
    
    # 3. End Date (선택 입력, 오늘 날짜 기본값)
    parser.add_argument(
        "--end",
        type=api.parse_date,
        default=datetime.date.today().strftime("%Y-%m-%d"),
        help="End date (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--num",
        type=int,
        default=0,
        help="Fetch data size limit"
    )

    args = parser.parse_args()

    if not args.ticker or args.ticker.strip() == "":
        print("Error: Ticker must be provided.")
        sys.exit(1)

    # 실행
    model(ticker=args.ticker, start=args.start, end=args.end, num=args.num)