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
    "2324.TW": {"adr": "HPQ", "index": "^IXIC", "name": "仁寶"}, # 增加仁寶對照組
}

st.title("🍎 股市投資小幫手")
st.subheader("AI 雙價位戰略分析 (掛牌價 + 還原價)")

tw_tz = pytz.timezone('Asia/Taipei')
now_tw = datetime.now(tw_tz).strftime('%Y-%m-%d %H:%M:%S')
st.caption(f"📅 系統偵測時間：{now_tw} (台北)")

target = st.text_input("輸入台股代號 (例如: 2324.TW)", value="2330.TW").upper().strip()
analyze_btn = st.button("執行深度戰略分析", type="primary")

if analyze_btn:
    config = STRATEGY_MAP.get(target, {"adr": "^IXIC", "index": "^SOX", "name": "一般個股"})
    
    with st.spinner('正在同步全球雙軌數據...'):
        try:
            # A. 抓取數據：同時抓取「掛牌價」與「還原價」
            raw_data = yf.download(target, period="2y", auto_adjust=False, progress=False)
            adj_data = yf.download(target, period="2y", auto_adjust=True, progress=False)
            
            # 美股連動數據 (維持還原價以利 AI 學習趨勢)
            df_adr = yf.download(config['adr'], period="2y", auto_adjust=True, progress=False)['Close']
            df_idx = yf.download(config['index'], period="2y", auto_adjust=True, progress=False)['Close']

            # 建立合併表格
            df = pd.DataFrame({
                'TW_Raw': raw_data['Close'],   # 市場掛牌價 (27.45)
                'TW_Adj': adj_data['Close'],   # 還原股價 (26.95)
                'ADR': df_adr,
                'IDX': df_idx
            })

            df = df.dropna(subset=['TW_Raw']).ffill().dropna()

            # B. 取得最新數值
            curr_raw = float(df['TW_Raw'].iloc[-1])
            curr_adj = float(df['TW_Adj'].iloc[-1])
            
            # C. 技術指標與 AI (使用還原價計算，避免除息缺口干擾)
            ma5 = df['TW_Adj'].rolling(5).mean().iloc[-1]
            ma20 = df['TW_Adj'].rolling(20).mean().iloc[-1]
            ma60 = df['TW_Adj'].rolling(60).mean().iloc[-1]
            std20 = df['TW_Adj'].rolling(20).std().iloc[-1]
            
            # 轉換回掛牌價比例的地板與天花板 (方便下單)
            ratio = curr_raw / curr_adj
            upper_band = (ma20 + (2 * std20)) * ratio
            lower_band = (ma20 - (2 * std20)) * ratio

            # D. AI 訓練 (使用還原百分比)
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
            pred_price_raw = curr_raw * (1 + pred_pct) # 以掛牌價顯示預測結果

            # E. 介面呈現
            st.divider()
            
            # 戰略判斷
            is_long_bull = ma5 > ma20 > ma60
            if is_long_bull:
                st.success(f"✅【強力推薦】{config['name']} 長線趨勢強勁。")
            else:
                st.error(f"❌【避開風險】{config['name']} 目前走勢偏弱。")

            # 核心數據卡片
            c1, c2, c3 = st.columns(3)
            # 今日收盤：主顯示掛牌價，副顯示還原價
            c1.metric("市場掛牌價", f"{curr_raw:.2f}")
            st.sidebar.write(f"💡 還原參考價: {curr_adj:.2f}") # 放在側邊或小字
            
            trend_icon = "📈" if prob_up > 0.5 else "📉"
            c2.metric(f"明日預測 (掛牌)", f"{pred_price_raw:.2f}", f"{pred_pct*100:+.2f}%")
            c3.metric("方向信心度", f"{max(prob_up, 1-prob_up)*100:.0f}%")

            # 買賣區間 (直接給掛牌價，方便下單)
            st.write("### 🚩 實戰下單價格參考 (掛牌價)")
            col_a, col_b = st.columns(2)
            col_a.info(f"📍 **分批撿貨價 (地板)：{lower_band:.2f}**")
            col_b.error(f"📍 **建議獲利價 (天花板)：{upper_band:.2f}**")

            with st.expander("🔍 查看還原價與技術細節"):
                st.write(f"**今日市場價：** {curr_raw:.2f}")
                st.write(f"**今日還原價：** {curr_adj:.2f}")
                st.write(f"**兩者價差：** {curr_raw - curr_adj:.2f} (包含已領取股利/權利)")
                st.caption("AI 運算時會自動考慮除權息缺口，確保預測不失真。")

        except Exception as e:
            st.error(f"分析發生錯誤：{e}")
