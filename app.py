import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from datetime import datetime
import pytz

# 1. 網頁頁面設定
st.set_page_config(page_title="AI 股市戰略導航儀", page_icon="📈", layout="centered")

# 2. 定義連動標項
STRATEGY_MAP = {
    "2330.TW": {"adr": "TSM", "index": "^SOX", "name": "台積電"},
    "2317.TW": {"adr": "AAPL", "index": "^IXIC", "name": "鴻海"},
    "2454.TW": {"adr": "NVDA", "index": "^SOX", "name": "聯發科"},
    "2303.TW": {"adr": "UMC", "index": "^SOX", "name": "聯電"},
    "3711.TW": {"adr": "ASX", "index": "^SOX", "name": "日月光"},
    "2324.TW": {"adr": "HPQ", "index": "^IXIC", "name": "仁寶"},
}

# 3. 核心時間邏輯：自動切換標籤
tw_tz = pytz.timezone('Asia/Taipei')
now_tw = datetime.now(tw_tz)
now_str = now_tw.strftime('%Y-%m-%d %H:%M:%S')

# 判定顯示文案
if now_tw.hour >= 15:
    # 下午三點後到午夜前
    price_label = "今日收盤價"
    pred_label = "預測明日 (掛牌)"
    status_msg = "🔴 台股已收盤，顯示今日結算數據"
elif 0 <= now_tw.hour < 9:
    # 午夜過後到開盤前
    price_label = "昨日收盤價"
    pred_label = "預測今日 (掛牌)"
    status_msg = "🌙 午夜時段，正在為今日開盤做預測"
else:
    # 早上九點到下午三點 (盤中)
    price_label = "最後成交價"
    pred_label = "預測今日收盤 (掛牌)"
    status_msg = "⚡ 盤中即時分析，數據隨時變動"

st.title("🍎 股市投資小幫手")
st.subheader("AI 雙軌全方位戰略系統")
st.caption(f"📅 台北時間：{now_str}")
st.info(status_msg)

target = st.text_input("輸入台股代號 (例如: 2324.TW)", value="2324.TW").upper().strip()
analyze_btn = st.button("執行深度戰略分析", type="primary")

if analyze_btn:
    config = STRATEGY_MAP.get(target, {"adr": "^IXIC", "index": "^SOX", "name": "一般個股"})
    
    with st.spinner('正在同步數據並計算長短線建議...'):
        try:
            # A. 數據抓取與時區抹除
            tk = yf.Ticker(target)
            hist = tk.history(period="2y", auto_adjust=False)
            adj_hist = tk.history(period="2y", auto_adjust=True)
            
            # 抹除時區資訊以利對齊
            hist.index = hist.index.tz_localize(None)
            adj_hist.index = adj_hist.index.tz_localize(None)

            # B. 即時價格抓取 (確保 27.45 正確)
            try:
                live_price = float(tk.fast_info['last_price'])
            except:
                live_price = float(hist['Close'].iloc[-1])

            # C. 下載美股連動
            df_adr = yf.download(config['adr'], period="2y", auto_adjust=True, progress=False)['Close']
            df_idx = yf.download(config['index'], period="2y", auto_adjust=True, progress=False)['Close']
            df_adr.index = df_adr.index.tz_localize(None)
            df_idx.index = df_idx.index.tz_localize(None)

            # D. 寬容數據對齊
            main_df = pd.DataFrame(index=hist.index)
            main_df['TW_Raw'] = hist['Close']
            main_df['TW_Adj'] = adj_hist['Close']
            main_df['ADR'] = df_adr.reindex(main_df.index, method='ffill')
            main_df['IDX'] = df_idx.reindex(main_df.index, method='ffill')

            df = main_df.dropna(subset=['TW_Raw']).ffill().dropna()

            # E. 數值與除息校準
            curr_raw = live_price if live_price > 0 else float(df['TW_Raw'].iloc[-1])
            curr_adj = float(df['TW_Adj'].iloc[-1])
            
            # 手動計算還原比例 (若最新資料未入庫)
            if abs(curr_raw - df['TW_Raw'].iloc[-1]) > 0.05:
                stable_ratio = df['TW_Raw'].iloc[-2] / df['TW_Adj'].iloc[-2]
                curr_adj = curr_raw / stable_ratio

            # F. 技術指標 (均線)
            ma5 = df['TW_Adj'].rolling(5).mean().iloc[-1]
            ma20 = df['TW_Adj'].rolling(20).mean().iloc[-1]
            ma60 = df['TW_Adj'].rolling(60).mean().iloc[-1]
            std20 = df['TW_Adj'].rolling(20).std().iloc[-1]
            
            b_ratio = curr_raw / curr_adj
            upper_band = (ma20 + (2 * std20)) * b_ratio
            lower_band = (ma20 - (2 * std20)) * b_ratio

            # G. AI 訓練
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

            # H. 長短期建議邏輯 (修正回補)
            st.divider()
            is_long_bull = ma5 > ma20 > ma60
            is_strong_up = prob_up > 0.55
            
            if is_long_bull and is_strong_up:
                st.success(f"✅【強力推薦】{config['name']} 長線趨勢強勁，短線 AI 看好。")
                advice_msg = "目前路況極佳，建議分批加碼。若漲到天花板可適度減碼。"
            elif is_long_bull and not is_strong_up:
                st.warning(f"⏳【長多短空】{config['name']} 趨勢未壞，但短線偵測到回檔壓力。")
                advice_msg = "不用急著賣出，但現在不適合追高，等跌回地板價再買。"
            elif not is_long_bull and is_strong_up:
                st.info(f"⚡【短線反彈】{config['name']} 長線走勢偏弱，目前僅為短暫反彈。")
                advice_msg = "這只是短暫天晴，賺了就跑，千萬不要長抱，趨勢尚未翻轉。"
            else:
                st.error(f"❌【避開風險】{config['name']} 長線下行中，且短線 AI 看空。")
                advice_msg = "目前路況極差，建議握緊現金觀察地板價支撐，切勿輕易攤平。"

            # 數據卡片
            c1, c2, c3 = st.columns(3)
            c1.metric(price_label, f"{curr_raw:.2f}")
            c2.metric(pred_label, f"{pred_price_raw:.2f}", f"{pred_pct*100:+.2f}%")
            c3.metric("AI 信心度", f"{max(prob_up, 1-prob_up)*100:.0f}%")

            # 實戰區間
            st.write("### 🚩 實戰參考價 (掛牌)")
            col_a, col_b = st.columns(2)
            col_a.info(f"📍 **建議撿貨價 (地板)：{lower_band:.2f}**")
            col_b.error(f"📍 **建議獲利價 (天花板)：{upper_band:.2f}**")

            with st.expander("🔍 數據診斷與還原細節"):
                st.write(f"當前掛牌價: {curr_raw:.2f} / 還原價: {curr_adj:.2f}")
                st.write(f"價差(股利): {curr_raw - curr_adj:.2f}")
                st.write(f"長線體質: {'多頭排列' if is_long_bull else '偏弱震盪'}")
                st.markdown(f"**💡 操作建議：**\n{advice_msg}")

        except Exception as e:
            st.error(f"分析失敗，這可能是 Yahoo 數據源尚未結算完成。請重試。({e})")
