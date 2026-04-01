import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from datetime import datetime
import pytz

# 1. 網頁頁面優化設定
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
st.subheader("AI 全方位戰略分析系統")

# 設定台灣時區顯示時間
tw_tz = pytz.timezone('Asia/Taipei')
now_tw = datetime.now(tw_tz).strftime('%Y-%m-%d %H:%M:%S')
st.caption(f"📅 系統偵測時間：{now_tw} (台北)")

# 用戶輸入區
target = st.text_input("輸入台股代號 (例如: 2330.TW)", value="2330.TW").upper().strip()
analyze_btn = st.button("執行深度戰略分析", type="primary")

if analyze_btn:
    config = STRATEGY_MAP.get(target, {"adr": "^IXIC", "index": "^SOX", "name": "一般個股"})
    
    with st.spinner('正在同步全球數據並校正預測模型...'):
        try:
            # A. 下載歷史數據 (3年以確保均線準確)
            df_all = yf.download(target, period="3y", progress=False)['Close']
            us_adr = yf.download(config['adr'], period="3y", progress=False)['Close']
            us_idx = yf.download(config['index'], period="3y", progress=False)['Close']

            df = pd.concat([df_all, us_adr, us_idx], axis=1)
            df.columns = ['TW', 'ADR', 'IDX']
            df = df.ffill().dropna()

            # B. 技術指標計算
            curr_price = df['TW'].iloc[-1]
            ma5 = df['TW'].rolling(5).mean().iloc[-1]
            ma20 = df['TW'].rolling(20).mean().iloc[-1]
            ma60 = df['TW'].rolling(60).mean().iloc[-1]
            std20 = df['TW'].rolling(20).std().iloc[-1]
            
            upper_band = ma20 + (2 * std20) # 天花板
            lower_band = ma20 - (2 * std20) # 地板價

            # C. AI 特徵工程 (使用變動率百分比)
            df['ADR_Ret'] = df['ADR'].pct_change().shift(1)
            df['IDX_Ret'] = df['IDX'].pct_change().shift(1)
            df['Target_Cls'] = (df['TW'].shift(-1) > df['TW']).astype(int)
            df['Target_Pct'] = df['TW'].pct_change().shift(-1)
            
            final_df = df.dropna()
            X = final_df[['ADR_Ret', 'IDX_Ret']]
            
            # --- AI 訓練與預測 ---
            # 1. 預測信心度
            clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X[:-1], final_df['Target_Cls'][:-1])
            prob_up = clf.predict_proba(X.tail(1))[0][1]
            prob_down = clf.predict_proba(X.tail(1))[0][0]
            
            # 2. 預測具體價格
            regr = RandomForestRegressor(n_estimators=100, random_state=42).fit(X[:-1], final_df['Target_Pct'][:-1])
            pred_pct = regr.predict(X.tail(1))[0]
            pred_price = curr_price * (1 + pred_pct)

            # D. 介面呈現
            st.divider()
            
            # 決定狀態與長短期建議
            is_long_bull = ma5 > ma20 > ma60
            is_strong_up = prob_up > 0.55
            
            # 核心建議顯示
            if is_long_bull and is_strong_up:
                st.success("✅【強力推薦】長線趨勢強勁，短線 AI 看好，適合順風買進。")
                advice_msg = "目前路況極佳，建議分批加碼。若漲到天花板可適度減碼。"
            elif is_long_bull and not is_strong_up:
                st.warning("⏳【長多短空】長線趨勢未壞，但短線 AI 偵測到回檔壓力。")
                advice_msg = "不用急著賣出長線部位，但現在不適合追高，等跌回地板價再買。"
            elif not is_long_bull and is_strong_up:
                st.info("⚡【短線反彈】長線仍在塞車，但短線有小機率反彈。")
                advice_msg = "這只是小天晴，賺了就跑，千萬不要長抱，這不是翻轉趨勢。"
            else:
                st.error("❌【避開風險】長線下行中，且短線 AI 極度看空。")
                advice_msg = "路況極差，現在進場容易套牢，請握緊現金觀察地板價支撐。"

            # 主要數據卡片
            c1, c2, c3 = st.columns(3)
            c1.metric("今日收盤", f"{curr_price:.2f}")
            
            trend_icon = "📈" if prob_up > prob_down else "📉"
            final_prob = prob_up if prob_up > prob_down else prob_down
            c2.metric(f"AI 預測明日 {trend_icon}", f"{pred_price:.2f}", f"{pred_pct*100:+.2f}%")
            c3.metric("預測信心度", f"{final_prob*100:.0f}%")

            # 買賣區間提示
            st.write("### 🚩 實戰買賣價格參考")
            col_a, col_b = st.columns(2)
            col_a.info(f"📍 **建議撿貨價 (地板)：{lower_band:.2f}**")
            col_b.error(f"📍 **建議獲利價 (天花板)：{upper_band:.2f}**")

            # 詳細診斷報告
            with st.expander("🔍 深度戰略報告 (長短線整合)"):
                st.write(f"**標的名稱：** {config['name']} ({target})")
                st.write(f"**長線體質：** {'🌟 多頭排列 (高速公路)' if is_long_bull else '⚠️ 走勢偏弱 (慢速車道)'}")
                st.write(f"**短線天氣：** {'☀️ AI 預報陽光普照' if is_strong_up else '🌧️ AI 預報可能有雨'}")
                st.markdown(f"**💡 具體戰術建議：**\n{advice_msg}")
                st.caption("註：分析結果於美股收盤後（台北時間早上 7 點）參考價值最高。")

        except Exception as e:
            st.error(f"發生錯誤！請檢查代號是否正確。錯誤訊息：{e}")
