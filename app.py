import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

# A. 網頁頁面設定 (手機優化)
st.set_page_config(page_title="AI 股市導航", page_icon="📈")

# B. 股票對照表
STRATEGY_MAP = {
    "2330.TW": {"adr": "TSM", "index": "^SOX", "name": "台積電"},
    "2317.TW": {"adr": "AAPL", "index": "^IXIC", "name": "鴻海"},
    "2454.TW": {"adr": "NVDA", "index": "^SOX", "name": "聯發科"},
    "2303.TW": {"adr": "UMC", "index": "^SOX", "name": "聯電"},
    "3711.TW": {"adr": "ASX", "index": "^SOX", "name": "日月光"},
}

# C. 網頁標題與輸入
st.title("🍎 股市投資小幫手")
st.subheader("AI 全方位戰略分析 (行動版)")

target = st.text_input("輸入台股代號", value="2330.TW").upper().strip()
analyze_btn = st.button("開始執行戰略分析", type="primary")

if analyze_btn:
    config = STRATEGY_MAP.get(target, {"adr": "^IXIC", "index": "^SOX", "name": "一般股票"})
    
    with st.spinner('正在同步全球數據...'):
        try:
            # 1. 抓取數據
            df_all = yf.download(target, period="3y", progress=False)['Close']
            us_adr = yf.download(config['adr'], period="3y", progress=False)['Close']
            us_idx = yf.download(config['index'], period="3y", progress=False)['Close']

            df = pd.concat([df_all, us_adr, us_idx], axis=1)
            df.columns = ['TW', 'ADR', 'IDX']
            df = df.ffill().dropna()

            # 2. 計算指標
            ma5 = df['TW'].rolling(5).mean().iloc[-1]
            ma20 = df['TW'].rolling(20).mean().iloc[-1]
            ma60 = df['TW'].rolling(60).mean().iloc[-1]
            std20 = df['TW'].rolling(20).std().iloc[-1]
            
            upper_band = ma20 + (2 * std20)
            lower_band = ma20 - (2 * std20)
            curr = df['TW'].iloc[-1]

            # 3. AI 預測
            df['ADR_Ret'] = df['ADR'].pct_change().shift(1)
            df['IDX_Ret'] = df['IDX'].pct_change().shift(1)
            df['Target'] = (df['TW'].shift(-1) > df['TW']).astype(int)
            final_df = df.dropna()
            X = final_df[['ADR_Ret', 'IDX_Ret']]
            clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X[:-1], final_df['Target'][:-1])
            prob = clf.predict_proba(X.tail(1))[0][1]

            # 4. 顯示結果 (使用網頁排版)
            st.divider()
            
            # 戰略顏色判定
            is_long_bull = ma5 > ma20 > ma60
            is_short_bull = prob > 0.55
            
            if is_long_bull and is_short_bull:
                st.success("✅【推薦買入】趨勢強勁且還有上漲空間。")
            elif is_long_bull:
                st.warning("⏳【建議觀望】長線沒壞，但明天可能回檔。")
            elif is_short_bull:
                st.info("⚡【短線價差】只是小反彈，不要長抱。")
            else:
                st.error("❌【暫不進場】目前路況不佳，風險較高。")

            # 數據卡片
            col1, col2 = st.columns(2)
            col1.metric("目前價格", f"{curr:.2f}")
            col2.metric("明日上漲信心", f"{prob*100:.0f}%")

            # 交易區間
            st.write("### 🚩 買賣門票區間")
            st.info(f"📍 **地板價（適合撿）：{lower_band:.2f} 元**")
            st.error(f"📍 **天花板（記得賣）：{upper_band:.2f} 元**")

            # 詳細細節
            with st.expander("查看完整診斷報告"):
                st.write(f"**長線趨勢：** {'🌟 多頭排列' if is_long_bull else '⚠️ 正在塞車'}")
                st.write(f"**市場氣氛：** {'大家都在賺錢' if curr > ma60 else '大家都在賠錢'}")
                st.write(f"**操作提醒：** 接近 {upper_band:.1f} 元時記得獲利了結。")

        except Exception as e:
            st.error(f"分析失敗：{e}")