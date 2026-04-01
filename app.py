import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from datetime import datetime
import pytz

# 1. 網頁頁面設定
st.set_page_config(page_title="AI 雙價位戰略導航儀", page_icon="📈", layout="centered")

# 2. 定義連動標的
STRATEGY_MAP = {
    "2330.TW": {"adr": "TSM", "index": "^SOX", "name": "台積電"},
    "2317.TW": {"adr": "AAPL", "index": "^IXIC", "name": "鴻海"},
    "2454.TW": {"adr": "NVDA", "index": "^SOX", "name": "聯發科"},
    "2303.TW": {"adr": "UMC", "index": "^SOX", "name": "聯電"},
    "3711.TW": {"adr": "ASX", "index": "^SOX", "name": "日月光"},
    "2324.TW": {"adr": "HPQ", "index": "^IXIC", "name": "仁寶"},
}

# 3. 標題與時間顯示
st.title("🍎 股市投資小幫手")
st.subheader("AI 雙價位全方位戰略系統")

tw_tz = pytz.timezone('Asia/Taipei')
now_tw = datetime.now(tw_tz).strftime('%Y-%m-%d %H:%M:%S')
st.caption(f"📅 系統偵測時間：{now_tw} (台北)")

target = st.text_input("輸入台股代號 (例如: 2324.TW)", value="2330.TW").upper().strip()
analyze_btn = st.button("執行深度戰略分析", type="primary")

if analyze_btn:
    config = STRATEGY_MAP.get(target, {"adr": "^IXIC", "index": "^SOX", "name": "一般個股"})
    
    with st.spinner('正在同步全球數據並交叉計算...'):
        try:
            # A. 數據抓取 (雙軌制)
            raw_tw = yf.download(target, period="2y", auto_adjust=False, progress=False)['Close']
            adj_tw = yf.download(target, period="2y", auto_adjust=True, progress=False)['Close']
            df_adr = yf.download(config['adr'], period="2y", auto_adjust=True, progress=False)['Close']
            df_idx = yf.download(config['index'], period="2y", auto_adjust=True, progress=False)['Close']

            # B. 強健合併邏輯
            df = pd.DataFrame(index=raw_tw.index)
            df['TW_Raw'] = raw_tw
            df['TW_Adj'] = adj_tw
            df['ADR'] = df_adr
            df['IDX'] = df_idx

            df = df.dropna(subset=['TW_Raw']).ffill().dropna()

            # C. 取得最新數值
            curr_raw = float(df['TW_Raw'].iloc[-1])
            curr_adj = float(df['TW_Adj'].iloc[-1])
            
            # D. 技術指標 (以還原價計算確保趨勢正確)
            ma5 = df['TW_Adj'].rolling(5).mean().iloc[-1]
            ma20 = df['TW_Adj'].rolling(20).mean().iloc[-1]
            ma60 = df['TW_Adj'].rolling(60).mean().iloc[-1]
            std20 = df['TW_Adj'].rolling(20).std().iloc[-1]
            
            # 換算回掛牌價比例
            ratio = curr_raw / curr_adj
            upper_band = (ma20 + (2 * std20)) * ratio
            lower_band = (ma20 - (2 * std20)) * ratio

            # E. AI 預測模型
            df['ADR_Ret'] = df['ADR'].pct_change().shift(1)
            df['IDX_Ret'] = df['IDX'].pct_change().shift(1)
            df['Target_Pct'] = df['TW_Adj'].pct_change().shift(-1)
            df['Target_Cls'] = (df['TW_Adj'].shift(-1) > df['TW_Adj']).astype(int)
            
            final_df = df.dropna()
            X = final_df[['ADR_Ret', 'IDX_Ret']]
            
            # 分類：預測漲跌方向
            clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X[:-1], final_df['Target_Cls'][:-1])
            prob_up = clf.predict_proba(X.tail(1))[0][1]
            
            # 回歸：預測漲跌幅度
            regr = RandomForestRegressor(n_estimators=100, random_state=42).fit(X[:-1], final_df['Target_Pct'][:-1])
            pred_pct = float(regr.predict(X.tail(1))[0])
            pred_price_raw = curr_raw * (1 + pred_pct)

            # F. 戰略建議邏輯 (整合回歸)
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
                st.error(f"❌【避開風險】{config['name']} 長線下行中，且短線 AI 極度看空。")
                advice_msg = "目前路況極差，建議握緊現金觀察地板價支撐，切勿輕易攤平。"

            # G. 核心數據卡片
            c1, c2, c3 = st.columns(3)
            c1.metric("市場掛牌價", f"{curr_raw:.2f}")
            
            trend_icon = "📈" if prob_up > 0.5 else "📉"
            final_prob = max(prob_up, 1-prob_up)
            c2.metric(f"預測明日 (掛牌)", f"{pred_price_raw:.2f}", f"{pred_pct*100:+.2f}%")
            c3.metric("方向信心度", f"{final_prob*100:.0f}%")

            # H. 買賣區間提示
            st.write("### 🚩 實戰買賣價格參考 (市場掛牌價)")
            col_a, col_b = st.columns(2)
            col_a.info(f"📍 **建議撿貨價 (地板)：{lower_band:.2f}**")
            col_b.error(f"📍 **建議獲利價 (天花板)：{upper_band:.2f}**")

            # I. 詳細數據報告
            with st.expander("🔍 檢視詳細診斷與還原價細節"):
                st.write(f"**標的名稱：** {config['name']} ({target})")
                st.write(f"**市場現價：** {curr_raw:.2f}")
                st.write(f"**還原參考價：** {curr_adj:.2f}")
                st.write(f"**累計價差(股利)：** {curr_raw - curr_adj:.2f}")
                st.write(f"**長線體質：** {'🌟 多頭排列' if is_long_bull else '⚠️ 走勢偏弱'}")
                st.markdown(f"**💡 具體操作建議：**\n{advice_msg}")
                st.caption("註：分析結果於美股收盤後（台北時間早上 7 點）數據對齊最精準。")

        except Exception as e:
            st.error(f"分析發生錯誤：{e}")
