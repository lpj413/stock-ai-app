import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from datetime import datetime
import pytz

# 1. 網頁頁面優化設定 (行動版優化)
st.set_page_config(page_title="AI 股市戰略導航儀", page_icon="📈", layout="centered")

# 2. 定義連動標的
STRATEGY_MAP = {
    "2330.TW": {"adr": "TSM", "index": "^SOX", "name": "台積電"},
    "2317.TW": {"adr": "AAPL", "index": "^IXIC", "name": "鴻海"},
    "2454.TW": {"adr": "NVDA", "index": "^SOX", "name": "聯發科"},
    "2303.TW": {"adr": "UMC", "index": "^SOX", "name": "聯電"},
    "3711.TW": {"adr": "ASX", "index": "^SOX", "name": "日月光"},
}

# 3. 標題與時間顯示
st.title("🍎 股市投資小幫手")
st.subheader("AI 全方位戰略分析系統 (穩定版)")

# 設定台灣時區
tw_tz = pytz.timezone('Asia/Taipei')
now_tw = datetime.now(tw_tz).strftime('%Y-%m-%d %H:%M:%S')
st.caption(f"📅 系統偵測時間：{now_tw} (台北)")

# 用戶輸入區
target = st.text_input("輸入台股代號 (例如: 2330.TW)", value="2330.TW").upper().strip()
analyze_btn = st.button("執行深度戰略分析", type="primary")

if analyze_btn:
    config = STRATEGY_MAP.get(target, {"adr": "^IXIC", "index": "^SOX", "name": "一般個股"})
    
    with st.spinner('正在從全球資料庫同步數據...'):
        try:
            # A. 數據抓取：改用 Adj Close 並確保數據完整性
            # 抓取 2 年資料以平衡運算速度與精準度
            df_tw = yf.download(target, period="2y", progress=False)['Adj Close']
            df_adr = yf.download(config['adr'], period="2y", progress=False)['Adj Close']
            df_idx = yf.download(config['index'], period="2y", progress=False)['Adj Close']

            # 合併數據並剔除台股未開盤的日子
            df = pd.concat([df_tw, df_adr, df_idx], axis=1)
            df.columns = ['TW', 'ADR', 'IDX']
            # 先剔除台股為空的行，再用前一交易日補齊美股（處理時差）
            df = df.dropna(subset=['TW'])
            df = df.ffill().dropna()

            # B. 技術指標計算 (布林通道與均線)
            curr_price = df['TW'].iloc[-1]
            ma5 = df['TW'].rolling(5).mean().iloc[-1]
            ma20 = df['TW'].rolling(20).mean().iloc[-1]
            ma60 = df['TW'].rolling(60).mean().iloc[-1]
            std20 = df['TW'].rolling(20).std().iloc[-1]
            
            upper_band = ma20 + (2 * std20) # 天花板
            lower_band = ma20 - (2 * std20) # 地板價

            # C. AI 特徵工程 (使用百分比變動率)
            df['ADR_Ret'] = df['ADR'].pct_change().shift(1)
            df['IDX_Ret'] = df['IDX'].pct_change().shift(1)
            # 預測目標：明天的漲跌幅 %
            df['Target_Pct'] = df['TW'].pct_change().shift(-1)
            # 分類目標：明天是否上漲
            df['Target_Cls'] = (df['TW'].shift(-1) > df['TW']).astype(int)
            
            final_df = df.dropna()
            X = final_df[['ADR_Ret', 'IDX_Ret']]
            
            # --- AI 訓練與預測 ---
            # 1. 信心度模型 (隨機森林分類)
            clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X[:-1], final_df['Target_Cls'][:-1])
            prob_up = clf.predict_proba(X.tail(1))[0][1]
            prob_down = clf.predict_proba(X.tail(1))[0][0]
            
            # 2. 價格預測模型 (隨機森林回歸)
            regr = RandomForestRegressor(n_estimators=100, random_state=42).fit(X[:-1], final_df['Target_Pct'][:-1])
            pred_pct = regr.predict(X.tail(1))[0]
            pred_price = curr_price * (1 + pred_pct)

            # D. 介面呈現
            st.divider()
            
            # 長短線戰略判定
            is_long_bull = ma5 > ma20 > ma60
            is_strong_up = prob_up > 0.55
            
            # 綜合建議顯示
            if is_long_bull and is_strong_up:
                st.success("✅【強力推薦】長線趨勢強勁，短線 AI 看好，適合分批買進。")
                advice_msg = "目前路況極佳，建議分批加碼。若漲到天花板可適度減碼。"
            elif is_long_bull and not is_strong_up:
                st.warning("⏳【長多短空】長線趨勢未壞，但短線偵測到回檔壓力。")
                advice_msg = "不用急著賣出長線部位，但現在不適合追高，等跌回地板價再買。"
            elif not is_long_bull and is_strong_up:
                st.info("⚡【短線反彈】長線走勢偏弱，目前僅為短線小反彈。")
                advice_msg = "這只是短暫天晴，賺了就跑，千萬不要長抱，這不是翻轉趨勢。"
            else:
                st.error("❌【避開風險】長線下行中，且短線 AI 極度看空。")
                advice_msg = "目前路況極差，現在進場容易套牢，請握緊現金觀察地板價支撐。"

            # 核心數據卡片
            c1, c2, c3 = st.columns(3)
            c1.metric("今日收盤價", f"{curr_price:.2f}")
            
            trend_icon = "📈" if prob_up > prob_down else "📉"
            final_prob = prob_up if prob_up > prob_down else prob_down
            c2.metric(f"AI 預估明天 {trend_icon}", f"{pred_price:.2f}", f"{pred_pct*100:+.2f}%")
            c3.metric("方向信心度", f"{final_prob*100:.0f}%")

            # 買賣區間導航
            st.write("### 🚩 實戰買賣價格參考")
            col_a, col_b = st.columns(2)
            col_a.info(f"📍 **建議撿貨價 (地板)：{lower_band:.2f}**")
            col_b.error(f"📍 **建議獲利價 (天花板)：{upper_band:.2f}**")

            # 深度分析報告
            with st.expander("🔍 檢視詳細診斷報告"):
                st.write(f"**標的名稱：** {config['name']} ({target})")
                st.write(f"**長線體質：** {'🌟 多頭排列 (強勢形態)' if is_long_bull else '⚠️ 走勢偏弱 (整理形態)'}")
                st.write(f"**短線預報：** {'☀️ AI 預報陽光普照' if is_strong_up else '🌧️ AI 預報可能有雨'}")
                st.markdown(f"**💡 戰術操作：**\n{advice_msg}")
                st.caption("註：分析結果於美股收盤後（台北時間早上 7 點）數據對齊最精準。")

        except Exception as e:
            st.error(f"發生錯誤！請檢查代號是否正確。錯誤訊息：{e}")
