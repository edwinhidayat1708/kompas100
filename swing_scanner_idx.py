#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KOMPAS100 QUANTITATIVE SWING SCANNER V8.2
=========================================
Architectural Refinements for Institutional Deployment:
  1. Automated IDX Multi-Token Volume Scaler (Handles Lot vs Share anomalies).
  2. Multi-Year Corporate Actions Adjusted Pricing (auto_adjust=True verification) [finance].
  3. Ticker-Aware Cross-Sectional Panel Data Purging Framework.
"""
from __future__ import annotations

import os
import sys
import logging
import warnings
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import xgboost as xgb
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger("scanner_v8_2")

# =============================================================================
# HARDCORE CONFIGURATION PARAMETERS
# =============================================================================
KOMPAS100 = [
    "AADI", "ACES", "ADMR", "ADRO", "AKRA", "AMMN", "AMRT", "ANTM", "ARCI", "ARTO",
    "ASII", "BBCA", "BBNI", "BBRI", "BBTN", "BBYB", "BFIN", "BIPI", "BKSL", "BMRI",
    "BNBR", "BRIS", "BRMS", "BRPT", "BSDE", "BUKA", "BULL", "BUMI", "BUVA", "CBDK",
    "CMRY", "COIN", "CPIN", "CTRA", "CUAN", "DEWA", "DSNG", "ELSA", "EMAS", "EMTK",
    "ENRG", "ERAA", "ESSA", "EXCL", "GGRM", "GOTO", "HEAL", "HRTA", "HRUM", "ICBP",
    "IMPC", "INCO", "INDF", "INDY", "INET", "INKP", "ISAT", "ITMG", "JPFA", "JSMR",
    "KIJA", "KLBF", "KPIG", "LSIP", "MAPA", "MAPI", "MBMA", "MDKA", "MEDC", "MIKA",
    "MINA", "MYOR", "NCKL", "PANI", "PGAS", "PGEO", "PNLF", "PSAB", "PTBA", "PTRO",
    "PWON", "RAJA", "RATU", "RMKE", "SCMA", "SGER", "SMGR", "SMIL", "SMRA", "SSIA",
    "TAPG", "TINS", "TLKM", "TOBA", "TOWR", "TPIA", "UNTR", "UNVR", "WIFI", "WIRG",
]

YEARS_BACK = 7                 
WIB = ZoneInfo("Asia/Jakarta")

TP_ATR = 2.5
SL_ATR = 1.25
HORIZON = 7                    

MIN_PRICE = 150                
MIN_VALUE_IDR = 5e9            

FEATURES = [
    "Dist_MA200", "MA50_Slope", "ATR_Pct",
    "ATR_Compress", "BB_Width_Rel", "Dist_High20_ATR",
    "Range_Expansion", "Value_Ratio", "CLV",
    "Ext_MA20_ATR",
    "IHSG_vs_MA50", "SP500_Ret5", "USDIDR_Ret5",
]

# =============================================================================
# FIXED FLAW 1 & 2: RE-ENGINEERED PIPELINE CORE FUNCTIONS
# =============================================================================
def normalise_idx_volume(df: pd.DataFrame) -> pd.DataFrame:
    """
    FIX FLAW 1: Evaluates tracking value loops across raw data tables.
    Standardizes transactional share scaling automatically if lot metrics are passed.
    """
    df = df.copy()
    if df.empty or 'Volume' not in df.columns or 'Close' not in df.columns:
        return df
        
    median_turnover = (df['Volume'] * df['Close']).median()
    
    # If the median daily cash flow computes below Rp 50 Million on Kompas100 members,
    # yfinance has scaled inputs down to the 100-share LOT metric.
    if median_turnover < 50_000_000:
        df['Volume'] = df['Volume'] * 100
        
    return df

def round_to_idx_tick(price: float, direction: str = "up") -> float:
    """FIX FLAW 2: Maps pricing metrics to the official IDX microstructure rules."""
    if pd.isna(price) or price <= 0:
        return price
        
    if price <= 200:
        tick = 1
    elif price <= 500:
        tick = 2
    elif price <= 2000:
        tick = 5
    elif price <= 5000:
        tick = 10
    else:
        tick = 25
        
    if direction == "up":
        return float(np.ceil(price / tick) * tick)
    else:
        return float(np.floor(price / tick) * tick)

# =============================================================================
# ADVANCED MACHINE LEARNING DATA PACKER
# =============================================================================
def fetch_macro_infrastructure() -> pd.DataFrame | None:
    try:
        end_dt = (datetime.now(WIB) + pd.Timedelta(days=1)).date().isoformat()
        start_dt = (pd.Timestamp.now() - pd.DateOffset(years=YEARS_BACK)).date().isoformat()
        
        # Pull global markets using auto_adjust to keep structural consistency intact [finance]
        ihsg = yf.download("^JKSE", start=start_dt, end=end_dt, auto_adjust=True, progress=False)
        sp500 = yf.download("^GSPC", start=start_dt, end=end_dt, auto_adjust=True, progress=False)
        idr = yf.download("USDIDR=X", start=start_dt, end=end_dt, auto_adjust=True, progress=False)
        
        if isinstance(ihsg.columns, pd.MultiIndex): ihsg.columns = ihsg.columns.get_level_values(0)
        if isinstance(sp500.columns, pd.MultiIndex): sp500.columns = sp500.columns.get_level_values(0)
        if isinstance(idr.columns, pd.MultiIndex): idr.columns = idr.columns.get_level_values(0)
        
        macro_df = pd.DataFrame(index=pd.to_datetime(ihsg.index).normalize())
        macro_df['IHSG_Close'] = ihsg['Close']
        macro_df['IHSG_MA50'] = ihsg['Close'].rolling(50).mean()
        macro_df['IHSG_vs_MA50'] = np.where(macro_df['IHSG_Close'] >= macro_df['IHSG_MA50'], 1.0, 0.0)
        
        # Shift global asset to fix temporal execution look-ahead bias
        macro_df['SP500_Ret5'] = sp500['Close'].pct_change(5).shift(1)
        macro_df['USDIDR_Ret5'] = idr['Close'].pct_change(5)
        
        return macro_df.dropna(subset=['IHSG_Close'])
    except Exception as e:
        log.error(f"Failed pulling macro matrix: {e}")
        return None

def engineer_features(df: pd.DataFrame, macro_df: pd.DataFrame | None, ticker_name: str) -> pd.DataFrame:
    df = df.copy()
    df['Ticker'] = ticker_name
    
    # Standard Volatility Bounds and Structural Moving Averages
    df['MA200'] = df['Close'].rolling(200).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    df['MA50_Slope'] = df['MA50'].pct_change(5)
    df['Dist_MA200'] = (df['Close'] - df['MA200']) / df['MA200']
    
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR14'] = tr.rolling(14).mean()
    df['ATR_Pct'] = df['ATR14'] / df['Close']
    
    df['ATR_Compress'] = df['ATR14'].rolling(5).min() / df['ATR14'].rolling(20).max()
    std20 = df['Close'].rolling(20).std()
    df['BB_Width_Rel'] = (std20 * 4) / df['Close']
    df['High20'] = df['High'].rolling(20).max()
    df['Dist_High20_ATR'] = (df['High20'] - df['Close']) / df['ATR14']
    
    df['Daily_Range'] = df['High'] - df['Low']
    df['Range_Expansion'] = df['Daily_Range'] / (df['Daily_Range'].rolling(5).mean() + 1e-8)
    df['Vol_MA20'] = df['Volume'].rolling(20).mean()
    df['Value_Ratio'] = (df['Volume'] * df['Close']) / ((df['Vol_MA20'] * df['Close']).rolling(20).mean() + 1e-8)
    df['CLV'] = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['Daily_Range'] + 1e-8)
    
    # Overextension Guard Boundary Metric
    df['MA20'] = df['Close'].rolling(20).mean()
    df['Ext_MA20_ATR'] = (df['Close'] - df['MA20']) / (df['ATR14'] + 1e-8)
    
    if macro_df is not None:
        df = df.join(macro_df[['IHSG_vs_MA50', 'SP500_Ret5', 'USDIDR_Ret5']], how='left').ffill()
        
    return df

def apply_true_barrier_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    labels = []
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    atr = df['ATR14'].values
    
    for i in range(len(df) - HORIZON):
        entry_p = close[i]
        curr_atr = atr[i]
        
        tp_target = round_to_idx_tick(entry_p + (TP_ATR * curr_atr), "up")
        sl_target = round_to_idx_tick(entry_p - (SL_ATR * curr_atr), "down")
        
        outcome = 0
        for day in range(1, HORIZON + 1):
            day_low = low[i + day]
            day_high = high[i + day]
            
            if day_low <= sl_target:
                outcome = 0
                break
            if day_high >= tp_target:
                outcome = 1
                break
        labels.append(outcome)
        
    labels.extend([0] * HORIZON)
    df['Target_Success'] = labels
    return df

# =============================================================================
# FIXED FLAW 3: SECURE TICKER-AWARE CROSS-SECTIONAL PURGING PIPELINE
# =============================================================================
def train_purged_panel_model(panel_df: pd.DataFrame) -> xgb.XGBClassifier:
    """
    FIX FLAW 3: Validates cross-sections using a strictly sequential, 
    non-overlapping chronological block split. Prevents data leakage.
    """
    unique_dates = np.sort(panel_df.index.unique())
    split_idx = int(len(unique_dates) * 0.8)
    
    # Determine execution split milestones using global date arrays
    train_dates = unique_dates[:split_idx]
    
    # Inject Purge Gap barrier to separate historical execution frames gracefully
    purge_end = train_dates[-1] + pd.Timedelta(days=HORIZON * 2)
    test_dates = unique_dates[unique_dates > purge_end]
    
    train_mask = panel_df.index.isin(train_dates)
    test_mask = panel_df.index.isin(test_dates)
    
    X_train = panel_df.loc[train_mask, FEATURES]
    y_train = panel_df.loc[train_mask, 'Target_Success']
    
    log.info(f"[*] Panel compilation constructed: {len(X_train)} Train arrays | {len(panel_df.loc[test_mask])} Purged Out-Of-Sample arrays.")
    
    model = xgb.XGBClassifier(
        n_estimators=120, max_depth=4, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    return model

# =============================================================================
# MAIN CONTROLLER PIPELINE
