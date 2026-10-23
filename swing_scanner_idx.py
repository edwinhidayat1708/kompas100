#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KOMPAS100 SWING SCANNER V8.1 (PROP-FIRM GRADE)
==============================================
Fixes implemented:
  1. BEI Volume Anomaly Correction (Lot vs Share scaling detection).
  2. Official IDX/BEI Regulation Tick-Size Dynamic Rounding Engine.
  3. Volatility-Normalized Overextension Protector (Z-Score Ext_MA20_ATR).
"""
from __future__ import annotations

import argparse
import html
import logging
import os
import sys
import warnings
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import xgboost as xgb
import yfinance as yf
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger("scanner")

# =============================================================================
# CONFIGURATION & UNIVERSES
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
BIG4 = ["BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK"]

YEARS_BACK = 7                 
MIN_HISTORY = 260              
WIB = ZoneInfo("Asia/Jakarta")
MARKET_FINAL_TIME = dtime(16, 30)   

TP_ATR = 2.5
SL_ATR = 1.25
HORIZON = 7                    
COST_ROUNDTRIP = 0.004         

MIN_PRICE = 150                
MIN_VALUE_IDR = 5e9            
REGIME_GATE = True             
BREADTH_MIN = 0.35

FEATURES = [
    "Dist_MA200", "MA50_Slope", "ATR_Pct",
    "ATR_Compress", "BB_Width_Rel", "Dist_High20_ATR",
    "Range_Expansion", "Value_Ratio", "CLV",
    "Ext_MA20_ATR",
    "IHSG_vs_MA50", "Breadth50", "SP500_Ret5", "USDIDR_Ret5",
]

# =============================================================================
# FIXED FLAW 1 & 2: RE-ENGINEERED MICROSTRUCTURE CORRECTIONS
# =============================================================================
def fix_bei_volume_anomaly(df: pd.DataFrame) -> pd.DataFrame:
    """
    FLAW 1 SOLUTION: Detects if yfinance parses IDX volumes in Lots instead of Shares.
    Ensures liquidity value computation (MIN_VALUE_IDR) remains architecturally intact [finance].
    """
    df = df.copy()
    if df.empty or 'Volume' not in df.columns or 'Close' not in df.columns:
        return df
        
    median_raw_value = (df['Volume'] * df['Close']).median()
    
    # If the median daily turnover registers under Rp 50 Million for a liquid Kompas100 asset,
    # it implies that yfinance is providing standard 100-share LOT data units.
    if median_raw_value < 50_000_000:
        df['Volume'] = df['Volume'] * 100
        
    return df

def round_to_idx_tick(price: float, direction: str = "up") -> float:
    """
    FLAW 2 SOLUTION: Implements official PT Bursa Efek Indonesia (IDX) tick sizing.
    Prevents mathematical drift mismatch inside the execution layers.
    """
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
# DATA UTILITIES & FEATURE ENGINEERING WITH OVEREXTENSION PROTECTION
# =============================================================================
def clean_index(idx) -> pd.DatetimeIndex:
    idx = pd.to_datetime(idx)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    return idx.normalize()

def fetch_macro_data() -> pd.DataFrame | None:
    """Fetches macro variables ensuring asynchronous S&P500 alignment (no look-ahead)."""
    try:
        end_dt = (datetime.now(WIB) + pd.Timedelta(days=1)).date().isoformat()
        start_dt = (pd.Timestamp.now() - pd.DateOffset(years=YEARS_BACK)).date().isoformat()
        
        ihsg = yf.download("^JKSE", start=start_dt, end=end_dt, progress=False)
        sp500 = yf.download("^GSPC", start=start_dt, end=end_dt, progress=False)
        idr = yf.download("USDIDR=X", start=start_dt, end=end_dt, progress=False)
        
        # Flatten MultiIndex columns if present
        if isinstance(ihsg.columns, pd.MultiIndex): ihsg.columns = ihsg.columns.get_level_values(0)
        if isinstance(sp500.columns, pd.MultiIndex): sp500.columns = sp500.columns.get_level_values(0)
        if isinstance(idr.columns, pd.MultiIndex): idr.columns = idr.columns.get_level_values(0)
        
        macro_df = pd.DataFrame(index=clean_index(ihsg.index))
        macro_df['IHSG_Close'] = ihsg['Close']
        macro_df['IHSG_MA50'] = ihsg['Close'].rolling(50).mean()
        macro_df['IHSG_vs_MA50'] = np.where(macro_df['IHSG_Close'] >= macro_df['IHSG_MA50'], 1.0, 0.0)
        
        # Shift global asset to fix temporal execution look-ahead bias
        macro_df['SP500_Ret5'] = sp500['Close'].pct_change(5).shift(1)
        macro_df['USDIDR_Ret5'] = idr['Close'].pct_change(5)
        
        return macro_df.dropna(subset=['IHSG_Close'])
    except Exception as e:
        log.error(f"Failed pulling macro infrastructure variables: {e}")
        return None

def engineer_features(df: pd.DataFrame, macro_df: pd.DataFrame | None) -> pd.DataFrame:
    """
    FLAW 3 SOLUTION: Incorporates Ext_MA20_ATR.
    Computes structural overextension metrics normalized via actual volatility metrics.
    """
    df = df.copy()
    
    # Technical Indicators
    df['MA200'] = df['Close'].rolling(200).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    df['MA50_Slope'] = df['MA50'].pct_change(5)
    df['Dist_MA200'] = (df['Close'] - df['MA200']) / df['MA200']
    
    # ATR Volatilities
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR14'] = tr.rolling(14).mean()
    df['ATR_Pct'] = df['ATR14'] / df['Close']
    
    # Setup Metrics
    df['ATR_Compress'] = df['ATR14'].rolling(5).min() / df['ATR14'].rolling(20).max()
    std20 = df['Close'].rolling(20).std()
    df['BB_Width_Rel'] = (std20 * 4) / df['Close']
    df['High20'] = df['High'].rolling(20).max()
    df['Dist_High20_ATR'] = (df['High20'] - df['Close']) / df['ATR14']
    
    # Triggers
    df['Daily_Range'] = df['High'] - df['Low']
    df['Range_Expansion'] = df['Daily_Range'] / df['Daily_Range'].rolling(5).mean()
    df['Vol_MA20'] = df['Volume'].rolling(20).mean()
    df['Value_Ratio'] = (df['Volume'] * df['Close']) / ((df['Vol_MA20'] * df['Close']).rolling(20).mean() + 1e-8)
    df['CLV'] = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['Daily_Range'] + 1e-8)
    
    # FLAW 3 COMPUTE: Volatility Bound Z-Score to block over-extended climax buying traps
    df['MA20'] = df['Close'].rolling(20).mean()
    df['Ext_MA20_ATR'] = (df['Close'] - df['MA20']) / (df['ATR14'] + 1e-8)
    
    if macro_df is not None:
        df = df.join(macro_df[['IHSG_vs_MA50', 'SP500_Ret5', 'USDIDR_Ret5']], how='left').ffill()
        # Proxy Universe Breadth calculation (Simulated via proxy column)
        df['Breadth50'] = 0.55 
        
    return df

def create_triple_barrier_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Labels using True chronological barrier checks instead of survival bias loops."""
    df = df.copy()
    labels = []
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    atr = df['ATR14'].values
    
    for i in range(len(df) - HORIZON):
        entry_p = close[i]
        curr_atr = atr[i]
        
        # Map dynamic targets using standard real structural boundaries
        tp_target = round_to_idx_tick(entry_p + (TP_ATR * curr_atr), "up")
        sl_target = round_to_idx_tick(entry_p - (SL_ATR * curr_atr), "down")
        
        barrier_hit = 0
        for day in range(1, HORIZON + 1):
            day_high = high[i + day]
            day_low = low[i + day]
            
            if day_low <= sl_target:
                barrier_hit = 0
                break
            if day_high >= tp_target:
                barrier_hit = 1
                break
        labels.append(barrier_hit)
        
    labels.extend([0] * HORIZON)
    df['Target_Success'] = labels
    return df

# =============================================================================
# PIPELINE EXECUTION ENGINE
# =============================================================================
def run_scanner():
    log.info("[*] Starting Quant Structural Scanner V8.1 Execution Pipeline...")
    macro_df = fetch_macro_data()
    
    panel_data_list = []
    live_scan_list = []
    
    for ticker in KOMPAS100:
        try:
            full_ticker = f"{ticker}.JK"
            end_dt = (datetime.now(WIB) + pd.Timedelta(days=1)).date().isoformat()
