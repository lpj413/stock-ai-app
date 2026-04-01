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

# 3. 標題與時間判定
st.title("🍎 股市投資小幫手")
st.subheader("AI 雙價位全方位分析系統")

# 設定台北時區
tw_tz = pytz.timezone('Asia/Taipei')
now_tw = datetime.now(tw_tz)
now_str = now_tw.strftime('%Y-%m-%d %H:%M:%S')

# 判定文字：15:00 為界
if now_tw.hour >= 15:
    price_label = "今日收盤價"
    status_msg = "🔴 台股已收盤，顯示今日結算數據"
else:
    price_label = "最後成交價"
    status_msg = "🔵 盤中/清晨時段，顯示最後已知數據"

st.caption(f"📅 系統偵測時間：{now_str} (台北)")
st.info(status_msg)

target = st.text_input("輸入台股代號 (例如: 2324.TW)", value="2324.TW").upper().strip()
analyze_btn = st.button("執行深度戰略分析", type="primary")

if analyze_btn:
    config = STRATEGY_MAP.get(target, {"adr": "^IXIC", "index": "^SOX", "name": "一般個股"})
    
    with st.spinner('數據同步與 AI 邏輯計算中...'):
        try:
            # A. 數據抓取
            tk = yf.Ticker(target)
            hist = tk.history(period="2y", auto_adjust=False)
            adj_hist = tk.history(period="2y", auto_adjust=True)
            
            if hist.empty:
                st.error("找不到該代號數據，請確認輸入格式。")
                st.stop()

            # B. 抓取掛牌價 (解決 27.45 問題)
            try:
                # 優先抓取 fast_info，這通常是最新掛牌價
                live_price = float(tk.fast_info['last_price'])
            except:
                live_price = float(hist['Close'].iloc[-1])

            # C. 下載美股指標
            df_adr = yf.download(config['adr'], period="2y", auto_adjust=True, progress=False)['Close']
            df_idx = yf.download(config['index'], period="2y", auto_adjust=True, progress=False)['Close']

            # D. 強制對齊數據 (預防 out-of-bounds 錯誤)
            df = pd.DataFrame(index=hist.index)
            df['TW_Raw'] = hist['Close']
            df['TW_Adj'] = adj_hist['Close']
            df['ADR'] = df_adr
            df['IDX'] = df_idx

            # 補齊美股因時差產生的空值，並確保台股有開盤
            df = df.ffill().dropna(subset=['TW_Raw']).dropna()

            if len(df) < 20:
                st.error("對齊數據量不足，請稍後重試。")
                st.stop()

            # E. 最新數值校正
            curr_raw = live_price if live_price > 0 else float(df['TW_Raw'].iloc[-1])
            curr_adj = float(df['TW_Adj'].iloc[-1])
            
            # 若 Raw 有更新但 Adj 沒跟上，維持比例補償
            if curr_raw != df['TW_Raw'].iloc[-1]:
                p_ratio = df['TW_Raw'].iloc[-1] / df['TW_Adj'].iloc[-1]
                curr_adj = curr_raw / p_ratio

            # F. 技術指標 (還原價為準)
            ma5 = df['TW_Adj'].rolling(5).mean().iloc[-1]
            ma20 = df['TW_Adj'].rolling(20).mean().iloc[-1]
            ma60 = df['TW_Adj'].rolling(60).mean().iloc[-1]
            std20 = df['TW_Adj'].rolling(20).std().iloc[-1]
            
            # 地板天花板換算回掛牌價比例
            band_ratio = curr_raw / curr_adj
            upper_band = (ma20 + (2 * std20)) * band_ratio
            lower_band = (ma20 - (2 * std20)) * band_ratio

            # G. AI 訓練與預測
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

            # H. 呈現區面
            st.divider()
            is_long_bull = ma5 > ma20 > ma60
            is_strong_up = prob_up > 0.55
            
            if is_long_bull and is_strong_up:
                st.success(f"✅【強力推薦】{config['name']} 趨勢極佳。")
                advice_msg = "建議分批加碼。若漲到天花板可適度減碼。"
            elif is_long_bull and not is_strong_up:
                st.warning(f"⏳【長多短空】{config['name']} 趨勢未壞，短線有壓。")
                advice_msg = "不用急著賣出，但現在不適合追高，等跌回地板價。"
            elif not is_long_bull and is_strong_up:
                st.info(f"⚡【短線反彈】{config['name']} 僅為短暫反彈。")
                advice_msg = "這只是短暫天晴，賺了就跑，不要長抱。"
            else:
                st.error(f"❌【避開風險】{config['name']} 長短線走勢皆弱。")
                advice_msg = "目前路況極差，建議握緊現金觀察。"

            c1, c2, c3 = st.columns(3)
            c1.metric(price_label, f"{curr_raw:.2f}")
            c2.metric("預測明日", f"{pred_price_raw:.2f}", f"{pred_pct*100:+.2f}%")
            c3.metric("信心度", f"{max(prob_up, 1-prob_up)*100:.0f}%")

            st.write("### 🚩 實戰買賣區間 (市場掛牌價)")
            col_a, col_b = st.columns(2)
            col_a.info(f"📍 **建議下單 (地板)：{lower_band:.2f}**")
            col_b.error(f"📍 **建議獲利 (天花板)：{upper_band:.2f}**")

            with st.expander("🔍 數據明細"):
                st.write(f"掛牌價: {curr_raw:.2f} / 還原價: {curr_adj:.2f}")
                st.write(f"今日價差: {curr_raw - curr_adj:.2f}")
                st.markdown(f"**詳細建議：**\n{advice_msg}")

        except Exception as e:
            st.error(f"分析發生錯誤，這通常是數據暫時無法對齊。請按按鈕重試。({e})")
