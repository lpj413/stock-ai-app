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
st.subheader("AI 雙價位全方位分析 (強韌版)")

tw_tz = pytz.timezone('Asia/Taipei')
now_tw = datetime.now(tw_tz)
now_str = now_tw.strftime('%Y-%m-%d %H:%M:%S')

# 判定文字：15:00 為界
if now_tw.hour >= 15:
    price_label = "今日收盤價"
    status_msg = "🔴 台股已收盤，目前顯示今日結算數據"
else:
    price_label = "最後成交價"
    status_msg = "🔵 盤中/清晨時段，顯示最後已知成交數據"

st.caption(f"📅 系統偵測時間：{now_str} (台北)")
st.info(status_msg)

target = st.text_input("輸入台股代號 (例如: 2324.TW)", value="2324.TW").upper().strip()
analyze_btn = st.button("執行深度戰略分析", type="primary")

if analyze_btn:
    config = STRATEGY_MAP.get(target, {"adr": "^IXIC", "index": "^SOX", "name": "一般個股"})
    
    with st.spinner('數據同步中，請稍候...'):
        try:
            tk = yf.Ticker(target)
            
            # A. 數據抓取：增加緩衝，確保一定有資料
            hist = tk.history(period="2y", auto_adjust=False)
            adj_hist = tk.history(period="2y", auto_adjust=True)
            
            if hist.empty or len(hist) < 20:
                st.error("此代號數據不足，請檢查代號是否正確。")
                st.stop()

            # B. 取得即時價格 (強效防錯邏輯)
            try:
                # 優先抓取即時最後成交價
                live_price = float(tk.fast_info['last_price'])
            except:
                # 如果 fast_info 報錯，抓取 history 最後一筆收盤價
                live_price = float(hist['Close'].iloc[-1])

            # C. 合併與數據校準
            df = pd.DataFrame(index=hist.index)
            df['TW_Raw'] = hist['Close']
            df['TW_Adj'] = adj_hist['Close']
            
            # 下載連動指標
            df_adr = yf.download(config['adr'], period="2y", auto_adjust=True, progress=False)['Close']
            df_idx = yf.download(config['index'], period="2y", auto_adjust=True, progress=False)['Close']
            
            df['ADR'] = df_adr
            df['IDX'] = df_idx
            df = df.dropna(subset=['TW_Raw']).ffill().dropna()

            # 確保最後一筆價格跟最新掛牌價同步 (27.45 關鍵)
            curr_raw = live_price if live_price > 0 else float(df['TW_Raw'].iloc[-1])
            curr_adj = float(df['TW_Adj'].iloc[-1])
            
            # D. 技術指標計算 (修正 index out-of-bounds 關鍵)
            # 確保 rolling windows 內有足夠數據
            ma5 = df['TW_Adj'].rolling(5).mean().iloc[-1]
            ma20 = df['TW_Adj'].rolling(20).mean().iloc[-1]
            ma60 = df['TW_Adj'].rolling(60).mean().iloc[-1]
            std20 = df['TW_Adj'].rolling(20).std().iloc[-1]
            
            # 比例換算
            ratio = curr_raw / curr_adj if curr_adj != 0 else 1
            upper_band = (ma20 + (2 * std20)) * ratio
            lower_band = (ma20 - (2 * std20)) * ratio

            # E. AI 預測模型
            df['ADR_Ret'] = df['ADR'].pct_change().shift(1)
            df['IDX_Ret'] = df['IDX'].pct_change().shift(1)
            df['Target_Pct'] = df['TW_Adj'].pct_change().shift(-1)
            df['Target_Cls'] = (df['TW_Adj'].shift(-1) > df['TW_Adj']).astype(int)
            
            final_df = df.dropna()
            if len(final_df) < 10:
                st.warning("當前數據對齊不完全，AI 預測可能略有偏差。")
            
            X = final_df[['ADR_Ret', 'IDX_Ret']]
            clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X[:-1], final_df['Target_Cls'][:-1])
            prob_up = clf.predict_proba(X.tail(1))[0][1]
            
            regr = RandomForestRegressor(n_estimators=100, random_state=42).fit(X[:-1], final_df['Target_Pct'][:-1])
            pred_pct = float(regr.predict(X.tail(1))[0])
            pred_price_raw = curr_raw * (1 + pred_pct)

            # F. 介面呈現
            st.divider()
            is_long_bull = ma5 > ma20 > ma60
            is_strong_up = prob_up > 0.55
            
            if is_long_bull and is_strong_up:
                st.success(f"✅【強力推薦】{config['name']} 趨勢極佳。")
                advice_msg = "目前路況極佳，建議分批加碼。若漲到天花板可適度減碼。"
            elif is_long_bull and not is_strong_up:
                st.warning(f"⏳【長多短空】{config['name']} 趨勢未壞，但有短線壓力。")
                advice_msg = "不用急著賣出，但現在不適合追高，等跌回地板價。"
            elif not is_long_bull and is_strong_up:
                st.info(f"⚡【短線反彈】{config['name']} 僅為短暫反彈。")
                advice_msg = "這只是短暫天晴，賺了就跑，不要長抱。"
            else:
                st.error(f"❌【避開風險】{config['name']} 長短線皆弱。")
                advice_msg = "建議握緊現金觀察地板價支撐。"

            c1, c2, c3 = st.columns(3)
            c1.metric(price_label, f"{curr_raw:.2f}")
            c2.metric("預測明日", f"{pred_price_raw:.2f}", f"{pred_pct*100:+.2f}%")
            c3.metric("信心度", f"{max(prob_up, 1-prob_up)*100:.0f}%")

            st.write("### 🚩 實戰參考 (掛牌價)")
            col_a, col_b = st.columns(2)
            col_a.info(f"📍 **分批撿貨價：{lower_band:.2f}**")
            col_b.error(f"📍 **建議獲利價：{upper_band:.2f}**")

            with st.expander("🔍 診斷細節"):
                st.write(f"市場價: {curr_raw:.2f} / 還原價: {curr_adj:.2f}")
                st.write(f"長線體質: {'多頭' if is_long_bull else '偏弱'}")
                st.markdown(f"**建議：**\n{advice_msg}")

        except Exception as e:
            st.error(f"⚠️ 數據源在結算期間不穩定，請 30 秒後點擊按鈕重試。 (Error: {e})")
