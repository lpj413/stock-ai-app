import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from datetime import datetime
import pytz

# 1. 網頁頁面設定
st.set_page_config(page_title="AI 雙價金戰略導航", page_icon="📈", layout="centered")

STRATEGY_MAP = {
    "2330.TW": {"adr": "TSM", "index": "^SOX", "name": "台積電"},
    "2317.TW": {"adr": "AAPL", "index": "^IXIC", "name": "鴻海"},
    "2454.TW": {"adr": "NVDA", "index": "^SOX", "name": "聯發科"},
    "2303.TW": {"adr": "UMC", "index": "^SOX", "name": "聯電"},
    "3711.TW": {"adr": "ASX", "index": "^SOX", "name": "日月光"},
    "2324.TW": {"adr": "HPQ", "index": "^IXIC", "name": "仁寶"},
}

st.title("🍎 股市投資小幫手")
st.subheader("AI 雙價位分析系統 (強健修正版)")

tw_tz = pytz.timezone('Asia/Taipei')
now_tw = datetime.now(tw_tz).strftime('%Y-%m-%d %H:%M:%S')
st.caption(f"📅 系統偵測時間：{now_tw} (台北)")

target = st.text_input("輸入台股代號 (例如: 2324.TW)", value="2330.TW").upper().strip()
analyze_btn = st.button("執行深度戰略分析", type="primary")

if analyze_btn:
    config = STRATEGY_MAP.get(target, {"adr": "^IXIC", "index": "^SOX", "name": "一般個股"})
    
    with st.spinner('正在同步全球雙軌數據...'):
        try:
            # A. 數據抓取
            raw_tw = yf.download(target, period="2y", auto_adjust=False, progress=False)['Close']
            adj_tw = yf.download(target, period="2y", auto_adjust=True, progress=False)['Close']
            df_adr = yf.download(config['adr'], period="2y", auto_adjust=True, progress=False)['Close']
            df_idx = yf.download(config['index'], period="2y", auto_adjust=True, progress=False)['Close']

            # B. 強健合併邏輯 (修正 Scalar Value 錯誤)
            # 確保所有數據都是 Series 格式並對齊索引
            df = pd.DataFrame(index=raw_tw.index)
            df['TW_Raw'] = raw_tw
            df['TW_Adj'] = adj_tw
            df['ADR'] = df_adr
            df['IDX'] = df_idx

            # 清洗數據：保留台股交易日並補齊美股缺失值
            df = df.dropna(subset=['TW_Raw'])
            df = df.ffill().dropna()

            # C. 取得最新數值
            curr_raw = float(df['TW_Raw'].iloc[-1])
            curr_adj = float(df['TW_Adj'].iloc[-1])
            
            # D. 技術指標 (用還原價計算)
            ma5 = df['TW_Adj'].rolling(5).mean().iloc[-1]
            ma20 = df['TW_Adj'].rolling(20).mean().iloc[-1]
            ma60 = df['TW_Adj'].rolling(60).mean().iloc[-1]
            std20 = df['TW_Adj'].rolling(20).std().iloc[-1]
            
            # 轉換回掛牌價比例
            ratio = curr_raw / curr_adj
            upper_band = (ma20 + (2 * std20)) * ratio
            lower_band = (ma20 - (2 * std20)) * ratio

            # E. AI 預測
            df['ADR_Ret'] = df['ADR'].pct_change().shift(1)
            df['IDX_Ret'] = df['IDX'].pct_change().shift(1)
            df['Target_Pct'] = df['TW_Adj'].pct_change().shift(-1)
            df['Target_Cls'] = (df['TW_Adj'].shift(-1) > df['TW_Adj']).astype(int)
            
            final_df = df.dropna()
            X = final_df[['ADR_Ret', 'IDX_Ret']]
            
            clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X[:-1], final_df['Target_Cls'][:-1])
            prob_up = clf.predict_proba(X.tail(1))[0][1]
            
            regr = RandomForestRegressor(n_estimators=100, random_state=42).fit(X[:-1], final_df['Target_Pct'][:-1])
            pred_pct = float(regr.predict(X.tail(1))[0])
            pred_price_raw = curr_raw * (1 + pred_pct)

            # F. 介面呈現
            st.divider()
            is_long_bull = ma5 > ma20 > ma60
            
            if is_long_bull:
                st.success(f"✅【推薦】{config['name']} 長線趨勢強勁。")
            else:
                st.error(f"❌【避開】{config['name']} 目前走勢偏弱。")

            c1, c2, c3 = st.columns(3)
            c1.metric("市場掛牌價", f"{curr_raw:.2f}")
            
            trend_icon = "📈" if prob_up > 0.5 else "📉"
            c2.metric(f"預測明日 (掛牌)", f"{pred_price_raw:.2f}", f"{pred_pct*100:+.2f}%")
            c3.metric("方向信心度", f"{max(prob_up, 1-prob_up)*100:.0f}%")

            st.write("### 🚩 實戰買賣價格參考 (掛牌價)")
            col_a, col_b = st.columns(2)
            col_a.info(f"📍 **建議撿貨價 (地板)：{lower_band:.2f}**")
            col_b.error(f"📍 **建議獲利價 (天花板)：{upper_band:.2f}**")

            with st.expander("🔍 查看數據細節"):
                st.write(f"**市場現價：** {curr_raw:.2f}")
                st.write(f"**還原參考價：** {curr_adj:.2f}")
                st.write(f"**價差(含股利)：** {curr_raw - curr_adj:.2f}")

        except Exception as e:
            st.error(f"分析發生錯誤，可能是 yfinance 暫時連線不穩。請點擊按鈕重試。錯誤訊息：{e}")
