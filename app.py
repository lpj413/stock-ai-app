import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from datetime import datetime
import pytz

# 1. 網頁頁面設定
st.set_page_config(page_title="AI 股市戰略導航儀 Pro", page_icon="🚀", layout="centered")

# 2. 定義連動標項
STRATEGY_MAP = {
    "2330.TW": {"adr": "TSM", "index": "^SOX", "name": "台積電"},
    "2317.TW": {"adr": "AAPL", "index": "^IXIC", "name": "鴻海"},
    "2454.TW": {"adr": "NVDA", "index": "^SOX", "name": "聯發科"},
    "2303.TW": {"adr": "UMC", "index": "^SOX", "name": "聯電"},
    "3711.TW": {"adr": "ASX", "index": "^SOX", "name": "日月光"},
    "2324.TW": {"adr": "HPQ", "index": "^IXIC", "name": "仁寶"},
}

# 3. 三段式時間自動切換標籤
tw_tz = pytz.timezone('Asia/Taipei')
now_tw = datetime.now(tw_tz)
now_str = now_tw.strftime('%Y-%m-%d %H:%M:%S')

if now_tw.hour >= 15:
    price_label, pred_label = "今日收盤價", "預測明日 (掛牌)"
    status_msg = "🔴 台股已收盤，顯示今日結算數據"
elif 0 <= now_tw.hour < 9:
    price_label, pred_label = "昨日收盤價", "預測今日 (掛牌)"
    status_msg = "🌙 午夜時段，正在為今日開盤做預測"
else:
    price_label, pred_label = "最後成交價", "預測今日收盤 (掛牌)"
    status_msg = "⚡ 盤中即時分析，數據隨時變動"

st.title("🍎 股市投資小幫手 Pro")
st.subheader("量價結合 AI 戰略系統")
st.caption(f"📅 台北時間：{now_str}")
st.info(status_msg)

target = st.text_input("輸入台股代號", value="2324.TW").upper().strip()
analyze_btn = st.button("執行量價深度分析", type="primary")

if analyze_btn:
    config = STRATEGY_MAP.get(target, {"adr": "^IXIC", "index": "^SOX", "name": "一般個股"})
    
    with st.spinner('正在計算量價特徵與 AI 信心度...'):
        try:
            # A. 數據抓取
            tk = yf.Ticker(target)
            hist = tk.history(period="2y", auto_adjust=False)
            adj_hist = tk.history(period="2y", auto_adjust=True)
            
            hist.index = hist.index.tz_localize(None)
            adj_hist.index = adj_hist.index.tz_localize(None)

            # B. 即時價格與成交量
            try:
                live_price = float(tk.fast_info['last_price'])
            except:
                live_price = float(hist['Close'].iloc[-1])

            # C. 下載美股與大盤
            df_adr = yf.download(config['adr'], period="2y", auto_adjust=True, progress=False)['Close']
            df_idx = yf.download(config['index'], period="2y", auto_adjust=True, progress=False)['Close']
            df_adr.index = df_adr.index.tz_localize(None)
            df_idx.index = df_idx.index.tz_localize(None)

            # D. 寬容對齊
            df = pd.DataFrame(index=hist.index)
            df['TW_Raw'] = hist['Close']
            df['TW_Adj'] = adj_hist['Close']
            df['Volume'] = hist['Volume'] # 引入成交量
            df['ADR'] = df_adr.reindex(df.index, method='ffill')
            df['IDX'] = df_idx.reindex(df.index, method='ffill')
            df = df.dropna(subset=['TW_Raw']).ffill().dropna()

            # E. 數值與比例校準
            curr_raw = live_price if live_price > 0 else float(df['TW_Raw'].iloc[-1])
            curr_adj = float(df['TW_Adj'].iloc[-1])
            if abs(curr_raw - df['TW_Raw'].iloc[-1]) > 0.05:
                ratio = df['TW_Raw'].iloc[-2] / df['TW_Adj'].iloc[-2]
                curr_adj = curr_raw / ratio

            # F. 技術指標 (均線與布林)
            ma5 = df['TW_Adj'].rolling(5).mean().iloc[-1]
            ma20 = df['TW_Adj'].rolling(20).mean().iloc[-1]
            ma60 = df['TW_Adj'].rolling(60).mean().iloc[-1]
            std20 = df['TW_Adj'].rolling(20).std().iloc[-1]
            
            b_ratio = curr_raw / curr_adj
            upper_band = (ma20 + (2 * std20)) * b_ratio
            lower_band = (ma20 - (2 * std20)) * b_ratio

            # G. AI 訓練：加入【成交量變化】特徵
            df['ADR_Ret'] = df['ADR'].pct_change().shift(1)
            df['IDX_Ret'] = df['IDX'].pct_change().shift(1)
            df['Vol_Ret'] = df['Volume'].pct_change().shift(1) # 成交量特徵
            df['Target_Pct'] = df['TW_Adj'].pct_change().shift(-1)
            df['Target_Cls'] = (df['TW_Adj'].shift(-1) > df['TW_Adj']).astype(int)
            
            final_df = df.dropna()
            # 訓練特徵包含：美股漲跌、大盤漲跌、成交量變化
            features = ['ADR_Ret', 'IDX_Ret', 'Vol_Ret']
            X = final_df[features]
            
            clf = RandomForestClassifier(n_estimators=200, random_state=42).fit(X[:-1], final_df['Target_Cls'][:-1])
            prob_up = clf.predict_proba(X.tail(1))[0][1]
            
            regr = RandomForestRegressor(n_estimators=200, random_state=42).fit(X[:-1], final_df['Target_Pct'][:-1])
            pred_pct = float(regr.predict(X.tail(1))[0])
            pred_price_raw = curr_raw * (1 + pred_pct)

            # H. 顯示長短線戰略建議
            st.divider()
            is_long_bull = ma5 > ma20 > ma60
            is_strong_up = prob_up > 0.55
            
            if is_long_bull and is_strong_up:
                st.success(f"✅【強力推薦】{config['name']} 長線趨勢強勁，且量價配合 AI 看好。")
                advice_msg = "路況極佳，建議分批加碼。若漲到天花板可適度減碼。"
            elif is_long_bull:
                st.warning(f"⏳【長多短空】{config['name']} 趨勢未壞，但短線量能轉弱。")
                advice_msg = "不用急著賣出，但現在不適合追高，等跌回地板價再買。"
            elif is_strong_up:
                st.info(f"⚡【短線反彈】{config['name']} 長線走勢偏弱，目前僅為短暫噴量反彈。")
                advice_msg = "這只是短暫天晴，賺了就跑，千萬不要長抱。"
            else:
                st.error(f"❌【避開風險】{config['name']} 長線下行且無量。")
                advice_msg = "目前路況極差，建議握緊現金，切勿輕易攤平。"

            # 數據卡片
            c1, c2, c3 = st.columns(3)
            c1.metric(price_label, f"{curr_raw:.2f}")
            c2.metric(pred_label, f"{pred_price_raw:.2f}", f"{pred_pct*100:+.2f}%")
            c3.metric("AI 信心度", f"{max(prob_up, 1-prob_up)*100:.0f}%")

            st.write("### 🚩 實戰參考價 (掛牌價)")
            col_a, col_b = st.columns(2)
            col_a.info(f"📍 **建議撿貨價 (地板)：{lower_band:.2f}**")
            col_b.error(f"📍 **建議獲利價 (天花板)：{upper_band:.2f}**")

            with st.expander("🔍 數據診斷細節"):
                st.write(f"當前掛牌價: {curr_raw:.2f} / 還原價: {curr_adj:.2f}")
                st.write(f"成交量狀態: {'放量' if df['Vol_Ret'].iloc[-1] > 0 else '縮量'}")
                st.markdown(f"**💡 操作建議：**\n{advice_msg}")

        except Exception as e:
            st.error(f"分析失敗，請稍後重試。({e})")
