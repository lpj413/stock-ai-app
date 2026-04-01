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

# 邏輯：00:00-09:00 預測今日開盤 / 09:00-15:00 預測今日收盤 / 15:00-24:00 預測明日
if 0 <= now_tw.hour < 9:
    p_label, f_label = "昨日收盤價", "預測今日 (掛牌)"
    status_msg = "🌙 午夜時段：正在為今日開盤進行 AI 建模"
elif 9 <= now_tw.hour < 15:
    p_label, f_label = "最後成交價", "預測今日收盤 (掛牌)"
    status_msg = "⚡ 盤中時段：數據隨時變動，請注意風險"
else:
    p_label, f_label = "今日收盤價", "預測明日 (掛牌)"
    status_msg = "🔴 收盤時段：今日數據已結算"

st.title("🍎 股市投資小幫手 Pro")
st.subheader("AI 雙軌全方位戰略系統")
st.caption(f"📅 台北時間：{now_str}")
st.info(status_msg)

target = st.text_input("輸入台股代號 (如: 2324.TW)", value="2324.TW").upper().strip()
analyze_btn = st.button("執行量價深度分析", type="primary")

if analyze_btn:
    config = STRATEGY_MAP.get(target, {"adr": "^IXIC", "index": "^SOX", "name": "一般個股"})
    
    with st.spinner('正在計算長短線趨勢與量價特徵...'):
        try:
            # A. 數據抓取與時區處理
            tk = yf.Ticker(target)
            hist = tk.history(period="2y", auto_adjust=False)
            adj_hist = tk.history(period="2y", auto_adjust=True)
            hist.index = hist.index.tz_localize(None)
            adj_hist.index = adj_hist.index.tz_localize(None)

            # B. 取得即時價格 (確保顯示 27.45)
            try:
                live_price = float(tk.fast_info['last_price'])
            except:
                live_price = float(hist['Close'].iloc[-1])

            # C. 下載連動指標並統一時區
            df_adr = yf.download(config['adr'], period="2y", auto_adjust=True, progress=False)['Close']
            df_idx = yf.download(config['index'], period="2y", auto_adjust=True, progress=False)['Close']
            df_adr.index = df_adr.index.tz_localize(None)
            df_idx.index = df_idx.index.tz_localize(None)

            # D. 強制數據對齊
            df = pd.DataFrame(index=hist.index)
            df['TW_Raw'] = hist['Close']
            df['TW_Adj'] = adj_hist['Close']
            df['Volume'] = hist['Volume']
            df['ADR'] = df_adr.reindex(df.index, method='ffill')
            df['IDX'] = df_idx.reindex(df.index, method='ffill')
            df = df.dropna(subset=['TW_Raw']).ffill().dropna()

            # E. 除息與價格比例校準
            curr_raw = live_price if live_price > 0 else float(df['TW_Raw'].iloc[-1])
            curr_adj = float(df['TW_Adj'].iloc[-1])
            if abs(curr_raw - df['TW_Raw'].iloc[-1]) > 0.05:
                # 使用前一日穩定比例
                ratio = df['TW_Raw'].iloc[-2] / df['TW_Adj'].iloc[-2]
                curr_adj = curr_raw / ratio

            # F. 技術指標 (均線)
            ma5 = df['TW_Adj'].rolling(5).mean().iloc[-1]
            ma20 = df['TW_Adj'].rolling(20).mean().iloc[-1]
            ma60 = df['TW_Adj'].rolling(60).mean().iloc[-1]
            std20 = df['TW_Adj'].rolling(20).std().iloc[-1]
            
            # 地板天花板換算
            b_ratio = curr_raw / curr_adj
            upper_band = (ma20 + (2 * std20)) * b_ratio
            lower_band = (ma20 - (2 * std20)) * b_ratio

            # G. AI 模型訓練 (加入量價特徵)
            df['ADR_Ret'] = df['ADR'].pct_change().shift(1)
            df['IDX_Ret'] = df['IDX'].pct_change().shift(1)
            df['Vol_Ret'] = df['Volume'].pct_change().shift(1)
            df['Target_Pct'] = df['TW_Adj'].pct_change().shift(-1)
            df['Target_Cls'] = (df['TW_Adj'].shift(-1) > df['TW_Adj']).astype(int)
            
            f_df = df.dropna()
            X = f_df[['ADR_Ret', 'IDX_Ret', 'Vol_Ret']]
            
            clf = RandomForestClassifier(n_estimators=200, random_state=42).fit(X[:-1], f_df['Target_Cls'][:-1])
            prob_up = clf.predict_proba(X.tail(1))[0][1]
            
            regr = RandomForestRegressor(n_estimators=200, random_state=42).fit(X[:-1], f_df['Target_Pct'][:-1])
            pred_pct = float(regr.predict(X.tail(1))[0])
            pred_price_raw = curr_raw * (1 + pred_pct)

            # H. 【核心鎖定】長短線戰略建議模組
            st.divider()
            is_long_bull = ma5 > ma20 > ma60  # 長線多頭排列
            is_strong_up = prob_up > 0.55     # 短線 AI 看好
            
            if is_long_bull and is_strong_up:
                st.success(f"💎 **戰略建議：強力推薦**")
                advice = f"目前的 {config['name']} 處於長線多頭排列，且短線量價動能強勁。路況極佳，建議分批加碼，持股續抱。若觸及天花板價 ({upper_band:.2f}) 可考慮調節部分獲利。"
            elif is_long_bull:
                st.warning(f"⚖️ **戰略建議：長多短空**")
                advice = f"目前的 {config['name']} 長線趨勢仍穩，但短線 AI 偵測到動能減弱或量能不足。不建議現在追高，持股者可續抱，空手者請靜待回檔至地板價 ({lower_band:.2f}) 附近再布局。"
            elif is_strong_up:
                st.info(f"⚡ **戰略建議：短線反彈**")
                advice = f"目前的 {config['name']} 長線走勢偏弱（尚未翻轉），目前僅為短暫的量能噴發反彈。策略應以「搶短」為主，賺了就跑，切勿長抱，需嚴格執行停損停利。"
            else:
                st.error(f"⚠️ **戰略建議：避開風險**")
                advice = f"目前的 {config['name']} 長線趨勢下行，且短線 AI 信心度極低。目前路況極差，建議握緊現金，不宜隨意攤平，靜待趨勢落底企穩再觀察。"

            # 數據儀表板
            c1, c2, c3 = st.columns(3)
            c1.metric(p_label, f"{curr_raw:.2f}")
            c2.metric(f_label, f"{pred_price_raw:.2f}", f"{pred_pct*100:+.2f}%")
            c3.metric("AI 信心度", f"{max(prob_up, 1-prob_up)*100:.0f}%")

            # 實戰區間
            st.write("### 🚩 實戰參考價 (掛牌價)")
            col_a, col_b = st.columns(2)
            col_a.info(f"📍 **建議撿貨價 (地板)：{lower_band:.2f}**")
            col_b.error(f"📍 **建議獲利價 (天花板)：{upper_band:.2f}**")

            with st.expander("🔍 數據診斷與細節"):
                st.write(f"當前掛牌價: {curr_raw:.2f} / 還原價: {curr_adj:.2f}")
                st.write(f"長線狀態: {'🌟 多頭排列' if is_long_bull else '⚠️ 偏弱震盪'}")
                st.write(f"成交量狀態: {'📈 放量' if df['Vol_Ret'].iloc[-1] > 0 else '📉 縮量'}")
                st.markdown(f"--- \n **💡 詳細分析：** \n {advice}")

        except Exception as e:
            st.error(f"分析失敗，這可能是 Yahoo 數據源尚未結算完成。請重試。({e})")
