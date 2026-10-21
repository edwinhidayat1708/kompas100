import pandas as pd
import numpy as np
import yfinance as yf
import xgboost as xgb
import os
import requests

# ==========================================
# 1. 100 KONSTITUEN LENGKAP INDEKS KOMPAS100
# ==========================================
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

# ==========================================
# 2. FEATURE ENGINEERING & LABELING (Target >=7% dalam 5 Hari)
# ==========================================
def engineer_features(df):
    df = df.copy()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    df['Return_1d'] = df['Close'].pct_change(1)
    df['Return_5d'] = df['Close'].pct_change(5)
    
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR14'] = tr.rolling(14).mean()
    df['ATR_Pct'] = df['ATR14'] / df['Close']
    
    df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()
    df['Vol_Ratio'] = df['Volume'] / df['Vol_MA20']
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

# ==========================================
# 3. KIRIM NOTIFIKASI KE TELEGRAM
# ==========================================
def send_telegram_message(message):
    token = os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("[!] Token atau Chat ID Telegram tidak ditemukan di Environment.")
        return
    
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("[+] Notifikasi Telegram berhasil terkirim.")
        else:
            print(f"[!] Gagal kirim Telegram: {response.text}")
    except Exception as e:
        print(f"[!] Error koneksi Telegram: {e}")

# ==========================================
# 4. MAIN SCANNER ENGINE
# ==========================================
def run_scanner():
    tickers = get_kompas100_tickers()
    rekomendasi_list = []
    
    print(f"[*] Memulai Quantitative Swing Scan pada {len(tickers)} emiten KOMPAS100...")

    for ticker in tickers:
        try:
            df = yf.download(ticker, period="2 yr", interval="1d", progress=False)
            if df.empty or len(df) < 250:
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = engineer_features(df)
            df = create_target_labels(df)
            
            current_close = df['Close'].iloc[-1]
            current_ma200 = df['MA200'].iloc[-1]
            
            if pd.isna(current_ma200) or current_close <= current_ma200:
                continue  # Filter Wajib: Harga harus di atas MA 200

            features = ['Return_1d', 'Return_5d', 'ATR_Pct', 'Vol_Ratio']
            model_df = df.dropna(subset=features + ['Target_Success'])
            
            if len(model_df) < 100:
                continue

            X = model_df[features]
            y = model_df['Target_Success']
            
            model = xgb.XGBClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='logloss'
            )
            model.fit(X, y)
            
            latest_features = X.iloc[[-1]]
            prob_success = model.predict_proba(latest_features)[0][1]
            
            if prob_success >= 0.65: # Filter Probabilitas Tinggi >= 65%
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
        
        # Format Pesan Telegram
        msg = "🚨 *DAILY SWING SCANNER (KOMPAS100)* 🚨\n"
        msg += "Model ML mendeteksi peluang profit $\\ge 7\\%$ dalam 5 hari:\n\n"
        for idx, row in df_log.iterrows():
            msg += f"🔹 *{row['Ticker']}*\n"
            msg += f"   • Close: `Rp {row['Close']}`\n"
            msg += f"   • Probabilitas Sukses: *{row['Probability_%']}%*\n"
            msg += f"   • Target (+7%): `Rp {row['Target_7Pct']}`\n"
            msg += f"   • Stop Loss: `Rp {row['Suggested_StopLoss']}`\n\n"
        
        send_telegram_message(msg)
        print(f"[+] Ditemukan {len(df_log)} sinyal valid & terkirim ke Telegram.")
    else:
        print("[-] Tidak ada sinyal yang memenuhi kriteria hari ini.")
        if os.path.exists("log_rekomendasi_swing_v3.csv"):
            os.remove("log_rekomendasi_swing_v3.csv")

if __name__ == "__main__":
    run_scanner()
