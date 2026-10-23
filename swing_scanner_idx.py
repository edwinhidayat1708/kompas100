import pandas as pd
import numpy as np
import yfinance as yf
import xgboost as xgb
import os
import requests

def get_kompas100_tickers():
    tickers = [
        "ACES", "ADRO", "AKRA", "ANTM", "ARTO", "ASII", "BBCA", "BBNI", "BBRI", "BBTN",
        "BMRI", "BRPT", "BUKA", "CPIN", "EMTK", "EXCL", "GOTO", "HRUM", "ICBP", "INCO",
        "INDF", "INKP", "ITMG", "JSMR", "KLBF", "MDKA", "MEDC", "PGAS", "PTBA", "SMGR",
        "SRTG", "TLKM", "UNTR", "UNVR", "MAPI", "ADMR", "AMRT", "ARCI", "ASSA", "AUTO",
        "BFIN", "BSDE", "BUVA", "CHIP", "CPRO", "CTRA", "DSSA", "ELSA", "ENRG", "ESSA",
        "FREN", "GJTL", "HEAL", "JKON", "JPFA", "KAIF", "KIJA", "MAPA", "MIKA",
        "MNCN", "MPPA", "MTEL", "MYOR", "NCKL", "PANI", "PTPP", "PWON", "RAJA",
        "SCMA", "SIDO", "SMDR", "SMRA", "TBIG", "TINS", "TKIM", "TOWR", "UNIQ", "WIFI",
        "WIKA", "WSKT", "BIRD", "BRIS", "BUKK", "CLEO", "DEWA", "DMAS", "DRMA", "HAIS",
        "HELI", "HERO", "HOME", "IPCM", "KEEN", "LPPF", "MAYA", "MLIA", "MPXL", "NZIA"
    ]
    cleaned = [t.strip().upper() + ".JK" for t in tickers]
    return list(set(cleaned))

def fetch_macro_and_market_data():
    """Menarik data makro dengan koreksi Time-Zone (Shift 1 hari untuk S&P 500) dan Big 4 Momentum"""
    try:
        ihsg = yf.download("^JKSE", period="7y", interval="1d", progress=False)['Close']
        ihsg_vol = yf.download("^JKSE", period="7y", interval="1d", progress=False)['Volume']
        
        sp500 = yf.download("^GSPC", period="7y", interval="1d", progress=False)['Close']
        idr = yf.download("USDIDR=X", period="7y", interval="1d", progress=False)['Close']
        
        # Big 4 Banks untuk mengukur Cross-Asset Momentum Divergence
        big4 = yf.download(["BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK"], period="7y", interval="1d", progress=False)['Close']
        if isinstance(big4, pd.DataFrame):
            big4_proxy = big4.mean(axis=1)
        else:
            big4_proxy = big4
            
        for df_item in [ihsg, ihsg_vol, sp500, idr, big4_proxy]:
            if isinstance(df_item, pd.DataFrame):
                df_item = df_item.iloc[:, 0]
        
        macro_df = pd.DataFrame({
            'IHSG': ihsg,
            'IHSG_Vol': ihsg_vol,
            'SP500': sp500, 
            'USDIDR': idr,
            'Big4': big4_proxy
        }).dropna()
        
        # KOREKSI TIME-ZONE: Shift S&P 500 sebesar 1 hari ke depan agar merefleksikan Wall Street semalam
        macro_df['SP500_Return_5d'] = macro_df['SP500'].pct_change(5).shift(1)
        macro_df['USDIDR_Return_5d'] = macro_df['USDIDR'].pct_change(5).shift(1)
        
        macro_df['IHSG_Return_5d'] = macro_df['IHSG'].pct_change(5)
        macro_df['Big4_Return_5d'] = macro_df['Big4'].pct_change(5)
        
        # Cross-Asset Momentum Divergence: Selisih momentum Big 4 dengan IHSG keseluruhan
        macro_df['Cross_Asset_Divergence'] = macro_df['Big4_Return_5d'] - macro_df['IHSG_Return_5d']
        
        return macro_df
    except Exception as e:
        print(f"[!] Gagal menarik data makro: {e}")
        return None

def engineer_features(df, macro_df):
    df = df.copy()
    
    # 1. Tren & Jarak ke MA200
    df['MA200'] = df['Close'].rolling(window=200).mean()
    df['Dist_MA200'] = (df['Close'] - df['MA200']) / df['MA200']
    
    # 2. Leading Indicator: Range Expansion & Volume Surge
    df['Daily_Range'] = (df['High'] - df['Low']) / df['Close']
    df['Range_MA5'] = df['Daily_Range'].rolling(window=5).mean()
    df['Range_Expansion'] = df['Daily_Range'] / df['Range_MA5']
    
    df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()
    df['Vol_Ratio'] = df['Volume'] / df['Vol_MA20']
    
    # 3. Volatilitas ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Low'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR14'] = tr.rolling(14).mean()
    df['ATR_Pct'] = df['ATR14'] / df['Close']
    
    # Gabungkan dengan data makro & market
    if macro_df is not None:
        df = df.join(macro_df, how='left').ffill()
        # 4. Beta Liquidity Ratio: Rasio Volume Saham dibagi Volume IHSG
        df['Liquidity_Ratio'] = df['Volume'] / df['IHSG_Vol']
        df['Liquidity_Ratio_MA5'] = df['Liquidity_Ratio'].rolling(window=5).mean()
        df['Liquidity_Surge'] = df['Liquidity_Ratio'] / df['Liquidity_Ratio_MA5']
        
    return df

def create_target_labels(df, forward_days=5, target_pct=0.07):
    labels = []
    close_prices = df['Close'].values
    high_prices = df['High'].values
    low_prices = df['Low'].values
    
    for i in range(len(df) - forward_days):
        entry_price = close_prices[i]
        future_highs = high_prices[i+1:i+1+forward_days]
        future_lows = low_prices[i+1:i+1+forward_days]
        
        max_gain = (future_highs.max() - entry_price) / entry_price
        max_loss = (future_lows.min() - entry_price) / entry_price
        
        if max_gain >= target_pct and max_loss > -0.035:
            labels.append(1)
        else:
            labels.append(0)
            
    labels.extend([0] * forward_days)
    df['Target_Success'] = labels
    return df

def send_telegram_message(message):
    token = os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    try:
        response = requests.post(url, json=payload)
        return response.status_code == 200
    except:
        return False

def run_scanner():
    print("[*] Mengambil data makro V7 (Time-Zone Corrected + Liquidity Ratio)...")
    macro_df = fetch_macro_and_market_data()
    
    tickers = get_kompas100_tickers()
    rekomendasi_list = []
    
    print(f"[*] Memulai Quantitative Swing Scan V7 pada {len(tickers)} emiten...")

    for ticker in tickers:
        try:
            df = yf.download(ticker, period="7y", interval="1d", progress=False)
            if df.empty or len(df) < 500:
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = engineer_features(df, macro_df)
            df = create_target_labels(df)
            
            current_close = df['Close'].iloc[-1]
            current_ma200 = df['MA200'].iloc[-1]
            
            if pd.isna(current_ma200) or current_close <= current_ma200:
                continue  # Filter Wajib: Harga di atas MA 200

            # Fitur V7: Beta Liquidity Ratio + Cross-Asset Divergence + Time-Shifted Macro
            features = [
                'Dist_MA200', 'Range_Expansion', 'Vol_Ratio', 'ATR_Pct', 
                'Liquidity_Surge', 'Cross_Asset_Divergence', 
                'SP500_Return_5d', 'USDIDR_Return_5d'
            ]
            
            model_df = df.dropna(subset=features + ['Target_Success'])
            if len(model_df) < 200:
                continue

            X = model_df[features]
            y = model_df['Target_Success']
            
            model = xgb.XGBClassifier(
                n_estimators=150, max_depth=4, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='logloss'
            )
            model.fit(X, y)
            
            latest_features = X.iloc[[-1]]
            prob_success = model.predict_proba(latest_features)[0][1]
            
            if prob_success >= 0.65:
                atr = df['ATR14'].iloc[-1]
                stop_loss = current_close - (1.5 * atr)
                target_7pct = current_close * 1.07
                
                rekomendasi_list.append({
                    'Ticker': ticker,
                    'Close': round(current_close, 2),
                    'Probability_%': round(prob_success * 100, 1),
                    'Suggested_StopLoss': round(stop_loss, 2),
                    'Target_7Pct': round(target_7pct, 2)
                })
        except Exception as e:
            print(f"[!] Error memproses {ticker}: {e}")

    if rekomendasi_list:
        df_log = pd.DataFrame(rekomendasi_list)
        df_log = df_log.sort_values(by='Probability_%', ascending=False)
        df_log.to_csv("log_rekomendasi_swing_v3.csv", index=False)
        
        msg = "🚨 *QUANT SWING SCANNER V7 (INSTITUTIONAL GRADE)* 🚨\n"
        msg += "Sinyal Probabilitas Sukses $\\ge 65\\%$ (Target +7% / 5 Hari):\n\n"
        for idx, row in df_log.iterrows():
            msg += f"🔹 *{row['Ticker']}*\n"
            msg += f"   • Close: `Rp {row['Close']}`\n"
            msg += f"   • Probabilitas: *{row['Probability_%']}%*\n"
            msg += f"   • Target (+7%): `Rp {row['Target_7Pct']}`\n"
            msg += f"   • Stop Loss: `Rp {row['Suggested_StopLoss']}`\n\n"
        
        send_telegram_message(msg)
        print(f"[+] Ditemukan {len(df_log)} sinyal valid & terkirim ke Telegram.")
    else:
        print("[-] Selesai. Tidak ada emiten yang lolos kriteria probabilitas tinggi hari ini.")
        send_telegram_message("ℹ️ *Quant Swing Scanner V7*:\nScan selesai. Tidak ada emiten KOMPAS100 yang memenuhi syarat probabilitas $\\ge 65\\%$ hari ini.")
        if os.path.exists("log_rekomendasi_swing_v3.csv"):
            os.remove("log_rekomendasi_swing_v3.csv")

if __name__ == "__main__":
    run_scanner()
