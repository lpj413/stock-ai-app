import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from datetime import datetime
import pytz

# 1. 網頁頁面設定
st.set_page_config(page_title="AI 股市戰略導航儀", page_icon="📈", layout="centered")

# 2. 定義連動標的
STRATEGY_MAP = {
    "2330.TW": {"adr": "TSM", "index": "^SOX", "name": "台積電"},
    "2317.TW": {"adr": "AAPL", "index": "^IXIC", "name": "鴻海"},
    "2454.TW": {"adr": "NVDA", "index": "^SOX", "name": "聯發科"},
    "2303.TW": {"adr": "UMC", "index": "^SOX", "name": "聯電"},
    "3711.TW": {"adr": "ASX", "index": "^SOX", "name": "日月光"},
    "2324.TW": {"adr": "HPQ", "index": "^IXIC", "name": "仁寶"},
}

# 3. 標題與時間判定
st.title("🍎 股市投資小幫手")
st.subheader("AI 雙價位 & 即時戰略系統")

# 設定台北時區
tw_tz = pytz.timezone('Asia/Taipei')
now_tw = datetime.now(tw_tz)
now_str = now_tw.strftime('%Y-%m-%d %H:%M:%S')

# 判定文字：15:00 為界
if now_tw.hour >= 15:
    price_label = "今日收盤價"
    status_msg = "🔴 台股已收盤，目前顯示今日結算數據"
else:
    price_label = "昨日收盤價"
    status_msg = "🔵 盤中/清晨時段，目前顯示昨日結算數據"

st.caption(f"📅 系統偵測時間：{now_str} (台北)")
st.info(status_msg)

target = st.text_input("輸入台股代號 (例如: 2324.TW)", value="2324.TW").upper().strip()
analyze_btn = st.button("執行深度戰略分析", type="primary")

if analyze_btn:
    config = STRATEGY_MAP.get(target, {"adr": "^IXIC", "index": "^SOX", "name": "一般個股"})
    
    with st.spinner('正在強制校準即時數據...'):
        try:
            # A. 數據抓取：使用 Ticker 獲取即時價格
            tk = yf.Ticker(target)
            
            # 抓取歷史數據包 (auto_adjust=False 確保 Raw 與 Adj 獨立)
            hist = tk.history(period="2y", auto_adjust=False)
            adj_hist = tk.history(period="2y", auto_adjust=True)
            
            # 強制校準：嘗試獲取最後成交價 (解決 26.95 延遲問題)
            try:
                live_price = tk.fast_info['last_price']
            except:
                live_price = hist['Close'].iloc[-1]

            # B. 合併與清洗
            df = pd.DataFrame(index=hist.index)
            df['TW_Raw'] = hist['Close']
            df['TW_Adj'] = adj_hist['Close']
            
            # 補償機制：如果 yfinance 的歷史紀錄還沒更新到今天的 27.45
            # 我們手動將最後一筆數據替換為即時抓到的價格
            curr_raw = live_price if live_price > 0 else float(df['TW_Raw'].iloc[-1])
            curr_adj = float(df['TW_Adj'].iloc[-1])
            
            # 如果 Raw 已經更新但 Adj (還原價) 還沒跟上，依比例手動補償
            if curr_raw != df['TW_Raw'].iloc[-1]:
                price_ratio = df['TW_Raw'].iloc[-1] / df['TW_Adj'].iloc[-1]
                curr_adj = curr_raw / price_ratio

            # 下載美股連動數據
            df_adr = yf.download(config['adr'], period="2y", auto_adjust=True, progress=False)['Close']
            df_idx = yf.download(config['index'], period="2y", auto_adjust=True, progress=False)['Close']
            
            df['ADR'] = df_adr
            df['IDX'] = df_idx
            df = df.dropna(subset=['TW_Raw']).ffill().dropna()

            # C. 技術指標 (基於還原價計算趨勢)
            ma5 = df['TW_Adj'].rolling(5).mean().iloc[-1]
            ma20 = df['TW_Adj'].rolling(20).mean().iloc[-1]
            ma60 = df['TW_Adj'].rolling(60).mean().iloc[-1]
            std20 = df['TW_Adj'].rolling(20).std().iloc[-1]
            
            # 換算回掛牌價比例的地板天花板
            raw_to_adj_ratio = curr_raw / curr_adj
            upper_band = (ma20 + (2 * std20)) * raw_to_adj_ratio
            lower_band = (ma20 - (2 * std20)) * raw_to_adj_ratio

            # D. AI 預測邏輯
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

            # E. 介面呈現
            st.divider()
            is_long_bull = ma5 > ma20 > ma60
            is_strong_up = prob_up > 0.55
            
            # 長短線建議整合
            if is_long_bull and is_strong_up:
                st.success(f"✅【強力推薦】{config['name']} 長線趨勢強勁，短線 AI 看好。")
                advice_msg = "目前路況極佳，建議分批加碼。若漲到天花板可適度減碼。"
            elif is_long_bull and not is_strong_up:
                st.warning(f"⏳【長多短空】{config['name']} 趨勢未壞，但短線有回檔壓力。")
                advice_msg = "不用急著賣出，但現在不適合追高，等跌回地板價再買。"
            elif not is_long_bull and is_strong_up:
                st.info(f"⚡【短線反彈】{config['name']} 長線偏弱，目前僅為短暫反彈。")
                advice_msg = "這只是短暫天晴，賺了就跑，千萬不要長抱。"
            else:
                st.error(f"❌【避開風險】{config['name']} 長線下行中，且 AI 極度看空。")
                advice_msg = "目前路況極差，建議握緊現金觀察地板價支撐。"

            # 數據卡片
            c1, c2, c3 = st.columns(3)
            c1.metric(price_label, f"{curr_raw:.2f}")
            
            trend_icon = "📈" if prob_up > 0.5 else "📉"
            c2.metric(f"預測明日 (掛牌)", f"{pred_price_raw:.2f}", f"{pred_pct*100:+.2f}%")
            c3.metric("方向信心度", f"{max(prob_up, 1-prob_up)*100:.0f}%")

            # 實戰區間
            st.write(f"### 🚩 實戰買賣參考 (掛牌價)")
            col_a, col_b = st.columns(2)
            col_a.info(f"📍 **建議撿貨價 (地板)：{lower_band:.2f}**")
            col_b.error(f"📍 **建議獲利價 (天花板)：{upper_band:.2f}**")

            with st.expander("🔍 數據診斷與還原細節"):
                st.write(f"**市場掛牌現價：** {curr_raw:.2f}")
                st.write(f"**還原參考價格：** {curr_adj:.2f}")
                st.write(f"**累計除息價差：** {curr_raw - curr_adj:.2f}")
                st.write(f"**長線體質判斷：** {'🌟 多頭' if is_long_bull else '⚠️ 弱勢'}")
                st.markdown(f"**💡 操作建議細節：**\n{advice_msg}")

        except Exception as e:
            st.error(f"分析發生錯誤，可能是數據源延遲。請稍後重試。錯誤：{e}")
