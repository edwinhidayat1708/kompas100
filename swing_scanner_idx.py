#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KOMPAS100 SWING SCANNER V8
==========================
Implementasi langkah 1-6 hasil review:

  1. Bug fix      : ATR benar, baris live tidak ikut dilatih, inf -> NaN, MultiIndex yfinance,
                    bar harian yang belum final dibuang, data macro disejajarkan tanpa look-ahead.
  2. Metodologi   : label triple-barrier berbasis ATR, entry di OPEN hari berikutnya, stop menang
                    jika target & stop kena di hari yang sama, biaya transaksi masuk hitungan,
                    SATU model gabungan (panel) untuk semua saham, walk-forward dengan purge gap,
                    kalibrasi probabilitas, threshold dipilih dari expectancy out-of-sample.
  3. Fitur        : satu tesis (breakout continuation setelah kompresi, di tren naik, saham
                    pemimpin). RSI dibuang -> diganti Ext_MA20_ATR sebagai penjaga overextended.
  4. Filter IDX   : nilai transaksi Rp, harga minimum, tick size (fraksi harga), filter perilaku
                    spekulatif (relatif terhadap universe), regime gate IHSG + breadth.
  5. Risk/Output  : RR, stop/target dibulatkan ke tick, batas entry (jangan kejar gap), time stop,
                    maksimal N sinyal, filter korelasi, pesan Telegram HTML + chunking.
  6. Forward-test : setiap sinyal dicatat ke signals_log.csv dan dievaluasi otomatis di run
                    berikutnya (winrate & expectancy riil out-of-sample).

Peran fitur (tidak ada yang saling bertentangan, semuanya satu tesis):
  Konteks tren   : Dist_MA200, MA50_Slope, ATR_Pct
  Setup (early)  : ATR_Compress, BB_Width_Rel, Dist_High20_ATR
  Trigger        : Range_Expansion, Value_Ratio, CLV
  Kepemimpinan   : RS20, RS60, RS_Rank
  Penjaga telat  : Ext_MA20_ATR
  Regime/makro   : IHSG_vs_MA50, Breadth50, Cross_Asset_Div, SP500_Ret5, USDIDR_Ret5

Dependensi : pip install pandas numpy yfinance xgboost scikit-learn requests
Env        : TELEGRAM_TOKEN, TELEGRAM_CHAT_ID (opsional)
Jalankan   : python scanner_v8.py            (live)
             python scanner_v8.py --dry-run  (cetak pesan, tidak kirim Telegram, tidak tulis log)
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
log = logging.getLogger("scanner")

# =============================================================================
# KONFIGURASI
# =============================================================================
# KOMPAS100 periode efektif 3 Agustus 2026 - 29 Januari 2027 (Peng-00148/BEI.POP/07-2026).
# PERBARUI tiap Februari & Agustus (evaluasi berikutnya diumumkan akhir Januari 2027).
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

# --- Data
YEARS_BACK = 7                 # jendela data training: tepat 7 tahun ke belakang dari hari run
SCAN_TIME_WIB = "18:00"        # jadwal run (diatur di .github/workflows/scanner.yml, cron 11:00 UTC)


def history_window():
    """(start, end) eksplisit. 'period=7y' bukan period resmi yfinance, jadi dipakai tanggal."""
    today = datetime.now(ZoneInfo("Asia/Jakarta")).date()
    start = (pd.Timestamp(today) - pd.DateOffset(years=YEARS_BACK)).date()
    return start.isoformat(), (pd.Timestamp(today) + pd.Timedelta(days=1)).date().isoformat()
MIN_HISTORY = 260              # minimal jumlah bar per saham (MA200 + warm-up)
MIN_UNIVERSE_OK = 0.70         # jika data yang berhasil < 70% universe -> run dianggap GAGAL
WIB = ZoneInfo("Asia/Jakarta")
MARKET_FINAL_TIME = dtime(16, 30)   # bar hari ini dianggap final setelah jam ini (WIB)

# --- Label triple-barrier (semua dalam kelipatan ATR14 pada hari sinyal)
TP_ATR = 2.5
SL_ATR = 1.25
HORIZON = 7                    # time stop (jumlah bar, termasuk hari entry)
COST_ROUNDTRIP = 0.004         # fee beli+jual + slippage kasar (0,4%). Sesuaikan dengan sekuritasmu.

# --- Filter IDX
MIN_PRICE = 150                # hindari fraksi harga besar relatif terhadap stop
MIN_VALUE_IDR = 5e9            # rata-rata nilai transaksi 20 hari (Rp). Cek: volume Yahoo IDX = lembar saham.
SPEC_EXCLUDE_PCT = 0.10        # buang 10% teratas skor perilaku spekulatif (vol & lonjakan harian) di universe
SPEC_JUMP = 0.09               # |return harian| >= 9% dihitung sebagai lonjakan
REGIME_GATE = True             # tidak ada sinyal jika IHSG < MA50 atau breadth < BREADTH_MIN
BREADTH_MIN = 0.35

# --- Validasi walk-forward
WF_FOLDS = 5
WF_TEST_DAYS = 120             # ~6 bulan bursa per fold
MIN_TRAIN_ROWS = 15000
MIN_OOF_TRADES = 60            # minimal jumlah trade out-of-sample (setelah de-duplikasi) agar threshold sah
MIN_EXP_R = 0.10               # expectancy minimal (dalam R, setelah biaya) agar sinyal boleh dikirim

# --- Output
MAX_SIGNALS = 5
CORR_MAX = 0.75                # lewati kandidat yang korelasi 60-harinya > ini dengan sinyal terpilih
ENTRY_CHASE_ATR = 0.5          # batas entry = close + 0,5 ATR (gap lebih tinggi -> jangan dikejar)
LOG_PATH = "signals_log.csv"

FEATURES = [
    "Dist_MA200", "MA50_Slope", "ATR_Pct",
    "ATR_Compress", "BB_Width_Rel", "Dist_High20_ATR",
    "Range_Expansion", "Value_Ratio", "CLV",
    "RS20", "RS60", "RS_Rank",
    "Ext_MA20_ATR",
    "IHSG_vs_MA50", "Breadth50", "Cross_Asset_Div", "SP500_Ret5", "USDIDR_Ret5",
]
OPTIONAL_MACRO = ["SP500_Ret5", "USDIDR_Ret5", "Cross_Asset_Div"]   # boleh NaN jika fetch gagal
REQUIRED = [f for f in FEATURES if f not in OPTIONAL_MACRO]

LOG_COLS = ["signal_date", "ticker", "prob", "close", "atr", "entry_date", "entry_price",
            "tp", "sl", "status", "exit_date", "exit_price", "R_net", "days_held"]


# =============================================================================
# UTILITAS DATA
# =============================================================================
def clean_index(idx) -> pd.DatetimeIndex:
    idx = pd.to_datetime(idx)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    return idx.normalize()


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance versi baru bisa mengembalikan kolom MultiIndex; ambil level yang berisi nama field."""
    if isinstance(df.columns, pd.MultiIndex):
        for lvl in range(df.columns.nlevels):
            if "Close" in df.columns.get_level_values(lvl):
                df = df.copy()
                df.columns = df.columns.get_level_values(lvl)
                break
    return df


def _tidy_ohlcv(df):
    if df is None or len(df) == 0:
        return None
    df = _flatten(df)
    cols = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in df.columns for c in cols):
        return None
    df = df[cols].copy()
    df.index = clean_index(df.index)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df[df["Volume"] > 0]            # buang hari tanpa transaksi / suspensi
    df = df[df["High"] >= df["Low"]]
    return df


def drop_partial_bar(obj):
    """Buang bar hari ini jika run terjadi sebelum bursa final (bar parsial merusak fitur & sinyal)."""
    if obj is None or len(obj) == 0:
        return obj
    now = datetime.now(WIB)
    if now.weekday() < 5 and obj.index[-1].date() == now.date() and now.time() < MARKET_FINAL_TIME:
        return obj.iloc[:-1]
    return obj


def fetch_universe(tickers):
    out, failed = {}, []
    raw = None
    try:
        start, end = history_window()
        raw = yf.download(tickers, start=start, end=end, interval="1d", group_by="ticker",
                          auto_adjust=True, progress=False, threads=True)
    except Exception as e:                      # noqa: BLE001
        log.error("Download batch gagal: %s", e)
    for t in tickers:
        df = None
        try:
            if raw is not None and isinstance(raw.columns, pd.MultiIndex) \
                    and t in raw.columns.get_level_values(0):
                df = _tidy_ohlcv(raw[t])
        except Exception as e:                  # noqa: BLE001
            log.warning("Parse batch %s gagal: %s", t, e)
        if df is None or len(df) < MIN_HISTORY:   # coba ulang per-ticker
            try:
                df = _tidy_ohlcv(yf.Ticker(t).history(start=history_window()[0], end=history_window()[1],
                                                     interval="1d", auto_adjust=True))
            except Exception as e:              # noqa: BLE001
                log.warning("Retry %s gagal: %s", t, e)
        if df is None or len(df) < MIN_HISTORY:
            failed.append(t)
            continue
        out[t] = drop_partial_bar(df)
    return out, failed


def fetch_close(symbol: str):
    start, end = history_window()
    df = yf.download(symbol, start=start, end=end, interval="1d", auto_adjust=True, progress=False)
    df = _flatten(df)
    s = df["Close"].dropna()
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    s.index = clean_index(s.index)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def speculative_filter(frames: dict):
    """Skor perilaku spekulatif RELATIF di dalam universe (robust terhadap rezim pasar)."""
    rows = {}
    for t, df in frames.items():
        ret = df["Close"].pct_change().tail(250)
        rows[t] = {"vol": ret.std() * np.sqrt(245), "jumps": float((ret.abs() >= SPEC_JUMP).sum())}
    s = pd.DataFrame(rows).T
    score = 0.5 * s["vol"].rank(pct=True) + 0.5 * s["jumps"].rank(pct=True)
    drop = score[score > (1.0 - SPEC_EXCLUDE_PCT)].index.tolist()
    return drop, s


# =============================================================================
# MACRO (tanpa look-ahead)
# =============================================================================
def asof_prev(feature: pd.Series, target_index: pd.DatetimeIndex) -> pd.Series:
    """Nilai terakhir feature pada tanggal STRICTLY SEBELUM tiap tanggal target.
    S&P 500 tanggal t-1 (tutup ~04:00 WIB hari t) adalah data terbaru yang diketahui saat IDX tutup hari t.
    """
    left = pd.DataFrame({"date": target_index}).sort_values("date")
    right = feature.dropna().rename("v").reset_index()
    right.columns = ["date", "v"]
    merged = pd.merge_asof(left, right.sort_values("date"), on="date",
                           direction="backward", allow_exact_matches=False)
    return pd.Series(merged["v"].values, index=left["date"].values).reindex(target_index)


def build_macro(ihsg, sp, idr, frames):
    m = pd.DataFrame(index=ihsg.index)
    m["IHSG_Ret5"] = ihsg.pct_change(5)
    m["IHSG_Ret20"] = ihsg.pct_change(20)
    m["IHSG_Ret60"] = ihsg.pct_change(60)
    m["IHSG_vs_MA50"] = ihsg / ihsg.rolling(50).mean() - 1

    big4 = [t for t in BIG4 if t in frames]
    if big4:
        b4 = pd.concat({t: frames[t]["Close"].pct_change(5) for t in big4}, axis=1).mean(axis=1)
        m["Big4_Ret5"] = b4.reindex(m.index)
        m["Cross_Asset_Div"] = m["Big4_Ret5"] - m["IHSG_Ret5"]
    else:
        m["Cross_Asset_Div"] = np.nan

    m["SP500_Ret5"] = asof_prev(sp.pct_change(5), m.index) if sp is not None else np.nan
    m["USDIDR_Ret5"] = asof_prev(idr.pct_change(5), m.index) if idr is not None else np.nan
    return m


# =============================================================================
# LABEL TRIPLE-BARRIER (vektor) & simulasi trade (skalar) - harus konsisten
# =============================================================================
def triple_barrier(o, h, l, c, atr):
    """Entry = OPEN bar t+1. TP = entry + TP_ATR*ATR_t, SL = entry - SL_ATR*ATR_t, horizon HORIZON bar.
    Jika TP & SL tersentuh di bar yang sama -> SL menang. Gap melewati level -> exit di open.
    Return: entry, outcome (1=TP, -1=SL, 0=timeout, NaN=belum bisa dihitung), exit price.
    """
    n = len(c)
    out = np.full(n, np.nan)
    exitp = np.full(n, np.nan)
    entry = np.full(n, np.nan)
    entry[:-1] = o[1:]
    tp = entry + TP_ATR * atr
    sl = entry - SL_ATR * atr

    valid = np.zeros(n, bool)
    if n - HORIZON > 0:
        valid[: n - HORIZON] = True
    valid &= ~np.isnan(entry) & ~np.isnan(atr) & (atr > 0)
    done = np.zeros(n, bool)

    def sh(x, k):
        r = np.full(n, np.nan)
        if n - k > 0:
            r[: n - k] = x[k:]
        return r

    for k in range(1, HORIZON + 1):
        ok_, hk, lk, ck = sh(o, k), sh(h, k), sh(l, k), sh(c, k)
        active = valid & ~done & ~np.isnan(hk)
        if k > 1:
            g_sl = active & (ok_ <= sl)
            out[g_sl] = -1; exitp[g_sl] = ok_[g_sl]; done |= g_sl; active &= ~g_sl
            g_tp = active & (ok_ >= tp)
            out[g_tp] = 1; exitp[g_tp] = ok_[g_tp]; done |= g_tp; active &= ~g_tp
        h_sl = active & (lk <= sl)
        out[h_sl] = -1; exitp[h_sl] = sl[h_sl]; done |= h_sl; active &= ~h_sl
        h_tp = active & (hk >= tp)
        out[h_tp] = 1; exitp[h_tp] = tp[h_tp]; done |= h_tp
        if k == HORIZON:
            to = valid & ~done
            out[to] = 0; exitp[to] = ck[to]; done |= to
    return entry, out, exitp


def simulate_trade(entry, atr, o, h, l, c, complete):
    """Versi skalar dari aturan yang sama, dipakai forward-test. Bar ke-0 = hari entry."""
    tp = entry + TP_ATR * atr
    sl = entry - SL_ATR * atr
    for i in range(len(o)):
        if i > 0:
            if o[i] <= sl:
                return "SL", i, float(o[i])
            if o[i] >= tp:
                return "TP", i, float(o[i])
        if l[i] <= sl:
            return "SL", i, float(sl)
        if h[i] >= tp:
            return "TP", i, float(tp)
    if complete:
        return "TIMEOUT", len(o) - 1, float(c[-1])
    return "OPEN", None, None


def r_multiple(entry, exit_px, atr):
    return ((exit_px / entry - 1.0) - COST_ROUNDTRIP) / (SL_ATR * atr / entry)


# =============================================================================
# FITUR & PANEL
# =============================================================================
def build_stock_frame(df: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d.index.name = "date"
    c, h, l, o, v = d["Close"], d["High"], d["Low"], d["Open"], d["Volume"]

    ma20, ma50, ma200 = c.rolling(20).mean(), c.rolling(50).mean(), c.rolling(200).mean()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)   # [FIX] ATR benar
    atr14, atr5, atr20 = tr.rolling(14).mean(), tr.rolling(5).mean(), tr.rolling(20).mean()
    value = c * v

    d["MA50"], d["MA200"], d["ATR14"] = ma50, ma200, atr14
    d["ValueMA20"] = value.rolling(20).mean()

    # konteks tren
    d["Dist_MA200"] = c / ma200 - 1
    d["MA50_Slope"] = ma50 / ma50.shift(10) - 1
    d["ATR_Pct"] = atr14 / c
    # setup (kompresi & kedekatan ke high)
    d["ATR_Compress"] = atr5 / atr20
    bbw = 4 * c.rolling(20).std() / ma20
    d["BB_Width_Rel"] = bbw / bbw.rolling(60).mean()
    d["Dist_High20_ATR"] = (h.rolling(20).max() - c) / atr14
    # trigger (hari ini dibanding 5/20 hari SEBELUMNYA)
    dr = (h - l) / c
    d["Range_Expansion"] = dr / dr.rolling(5).mean().shift(1)
    d["Value_Ratio"] = value / value.rolling(20).mean().shift(1)
    rng = (h - l).where((h - l) > 0)
    d["CLV"] = (((c - l) - (h - c)) / rng).fillna(0.0)
    # penjaga "sudah terlalu jauh"
    d["Ext_MA20_ATR"] = (c - ma20) / atr14

    # macro (ffill hanya kolom macro)
    d = d.join(macro, how="left")
    mcols = list(macro.columns)
    d[mcols] = d[mcols].ffill()
    # kepemimpinan vs IHSG
    d["RS20"] = c.pct_change(20) - d["IHSG_Ret20"]
    d["RS60"] = c.pct_change(60) - d["IHSG_Ret60"]

    # eligibility tingkat saham
    d["Elig_Stock"] = (c > ma200) & (c >= MIN_PRICE) & (d["ValueMA20"] >= MIN_VALUE_IDR)

    # label
    entry, out, exitp = triple_barrier(o.values, h.values, l.values, c.values, atr14.values)
    d["Entry"], d["Outcome"], d["Exit"] = entry, out, exitp
    d["R_Net"] = r_multiple(d["Entry"], d["Exit"], d["ATR14"])
    d["Y"] = np.where(np.isnan(out), np.nan, (out == 1).astype(float))
    return d.replace([np.inf, -np.inf], np.nan)          # [FIX] inf -> NaN


def build_panel(frames: dict, macro: pd.DataFrame) -> pd.DataFrame:
    per = {t: build_stock_frame(df, macro) for t, df in frames.items()}

    # fitur lintas-saham (tanpa look-ahead: hanya data hari yang sama)
    rs20 = pd.concat({t: d["RS20"] for t, d in per.items()}, axis=1)
    rs_rank = rs20.rank(axis=1, pct=True)
    above = pd.concat({t: (d["Close"] > d["MA50"]).astype(float).where(d["MA50"].notna())
                       for t, d in per.items()}, axis=1)
    breadth = above.mean(axis=1)
    for t, d in per.items():
        d["RS_Rank"] = rs_rank[t].reindex(d.index)
        d["Breadth50"] = breadth.reindex(d.index)
        d["Regime_OK"] = (d["IHSG_vs_MA50"] >= 0) & (d["Breadth50"] >= BREADTH_MIN)

    panel = pd.concat([d.assign(ticker=t) for t, d in per.items()]).reset_index()
    all_dates = np.sort(panel["date"].unique())
    panel["dpos"] = np.searchsorted(all_dates, panel["date"].values)
    return panel


# =============================================================================
# MODEL, WALK-FORWARD, KALIBRASI, THRESHOLD
# =============================================================================
def fit_model(X, y):
    m = xgb.XGBClassifier(
        n_estimators=250, max_depth=3, learning_rate=0.04, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=40, reg_lambda=5.0, tree_method="hist", eval_metric="logloss",
        random_state=42, n_jobs=-1,
    )
    m.fit(X, y)
    return m


def dedupe_signals(df: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """Satu trade per saham per `horizon` hari bursa (sinyal beruntun pada saham yang sama = satu posisi)."""
    if df.empty:
        return df
    df = df.sort_values(["ticker", "dpos"])
    keep, last = [], {}
    for idx, t, dp in zip(df.index, df["ticker"].values, df["dpos"].values):
        if t not in last or dp - last[t] >= horizon:
            keep.append(idx)
            last[t] = dp
    return df.loc[keep]


def trade_stats(df: pd.DataFrame) -> dict:
    n = len(df)
    if n == 0:
        return dict(n=0, tp=np.nan, sl=np.nan, to=np.nan, expR=np.nan, pf=np.nan)
    r = df["R_Net"].values
    gains, losses = r[r > 0].sum(), -r[r < 0].sum()
    return dict(
        n=n, tp=float((df["Outcome"] == 1).mean()), sl=float((df["Outcome"] == -1).mean()),
        to=float((df["Outcome"] == 0).mean()), expR=float(r.mean()),
        pf=float(gains / losses) if losses > 0 else np.inf,
    )


def walk_forward(train_df: pd.DataFrame):
    last = int(train_df["dpos"].max())
    gap = HORIZON + 2                                   # purge: label tumpang-tindih tidak boleh bocor
    oof, rows = [], []
    for f in range(WF_FOLDS):
        te_end = last - (WF_FOLDS - 1 - f) * WF_TEST_DAYS
        te_start = te_end - WF_TEST_DAYS + 1
        tr = train_df[train_df["dpos"] <= te_start - gap]
        te = train_df[(train_df["dpos"] >= te_start) & (train_df["dpos"] <= te_end)]
        if len(tr) < MIN_TRAIN_ROWS or len(te) == 0 or tr["Y"].sum() < 50:
            log.warning("Fold %d dilewati (train=%d, test=%d)", f + 1, len(tr), len(te))
            continue
        m = fit_model(tr[FEATURES], tr["Y"].astype(int))
        p = m.predict_proba(te[FEATURES])[:, 1]
        t2 = te[["date", "dpos", "ticker", "Y", "Outcome", "R_Net", "Regime_OK"]].copy()
        t2["p"], t2["fold"] = p, f + 1
        oof.append(t2)
        y = te["Y"].astype(int).values
        auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan
        top = p >= np.quantile(p, 0.9)
        rows.append(dict(fold=f + 1, n_test=len(te), base=y.mean(), auc=auc,
                         brier=brier_score_loss(y, p), prec_top10=y[top].mean(),
                         lift_top10=y[top].mean() / y.mean() if y.mean() > 0 else np.nan))
    if not oof:
        return None, pd.DataFrame()
    return pd.concat(oof, ignore_index=True), pd.DataFrame(rows)


def select_threshold(oof: pd.DataFrame):
    gate = oof["Regime_OK"].astype(bool) if REGIME_GATE else pd.Series(True, index=oof.index)
    if gate.sum() == 0:
        return None, trade_stats(oof.iloc[0:0])
    base = trade_stats(dedupe_signals(oof[gate]))
    best = None
    for thr in np.unique(np.quantile(oof.loc[gate, "p"], np.linspace(0.50, 0.995, 100))):
        st = trade_stats(dedupe_signals(oof[gate & (oof["p"] >= thr)]))
        if st["n"] < MIN_OOF_TRADES or not st["expR"] >= MIN_EXP_R:
            continue
        score = st["expR"] * np.sqrt(st["n"])
        if best is None or score > best["score"]:
            best = dict(st, thr=float(thr), score=float(score))
    return best, base


# =============================================================================
# TICK SIZE, FORMAT, TELEGRAM
# =============================================================================
def tick_size(p: float) -> int:
    if p < 200: return 1
    if p < 500: return 2
    if p < 2000: return 5
    if p < 5000: return 10
    return 25


def round_tick(p: float, mode: str = "down") -> float:
    t = tick_size(p)
    q = p / t
    if mode == "down":
        return float(np.floor(q + 1e-9) * t)
    if mode == "up":
        return float(np.ceil(q - 1e-9) * t)
    return float(np.round(q) * t)


def idr(x: float) -> str:
    return f"{x:,.0f}".replace(",", ".")


def send_telegram(text: str, dry_run: bool = False) -> bool:
    if dry_run:
        print("\n" + "=" * 70 + "\n[DRY-RUN] pesan Telegram:\n" + "=" * 70)
        print(html.unescape(text.replace("<b>", "").replace("</b>", "").replace("<code>", "").replace("</code>", "")))
        return True
    token, chat_id = os.environ.get("TELEGRAM_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        log.warning("TELEGRAM_TOKEN / TELEGRAM_CHAT_ID tidak diset - pesan tidak dikirim.")
        return False
    chunks, cur = [], ""
    for line in text.split("\n"):
        if len(cur) + len(line) + 1 > 3800:
            chunks.append(cur)
            cur = ""
        cur += line + "\n"
    if cur.strip():
        chunks.append(cur)
    ok = True
    for ch in chunks:
        try:
            r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                              json={"chat_id": chat_id, "text": ch, "parse_mode": "HTML",
                                    "disable_web_page_preview": True}, timeout=20)
            if r.status_code != 200:
                log.error("Telegram HTTP %s: %s", r.status_code, r.text[:200])
                ok = False
        except requests.RequestException as e:
            log.error("Telegram error: %s", e)
            ok = False
    return ok


# =============================================================================
# FORWARD-TEST LOG
# =============================================================================
def load_log(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, parse_dates=["signal_date", "entry_date", "exit_date"])
            for c in LOG_COLS:
                if c not in df.columns:
                    df[c] = np.nan
            return df[LOG_COLS]
        except Exception as e:                      # noqa: BLE001
            log.error("Gagal membaca %s: %s", path, e)
    return pd.DataFrame(columns=LOG_COLS)


def update_log(lg: pd.DataFrame, frames: dict) -> pd.DataFrame:
    """Evaluasi semua sinyal yang belum selesai dengan aturan yang SAMA dengan label training."""
    lg = lg.copy()
    for idx, row in lg[lg["status"].isin(["PENDING", "OPEN"])].iterrows():
        df = frames.get(row["ticker"])
        if df is None:
            continue
        pos = df.index.searchsorted(pd.Timestamp(row["signal_date"]), side="right")   # bar pertama setelah sinyal
        if pos >= len(df):
            continue                                                                # bar entry belum ada
        entry = float(df["Open"].iloc[pos])
        seg = df.iloc[pos: pos + HORIZON]
        status, i, px = simulate_trade(entry, float(row["atr"]), seg["Open"].values, seg["High"].values,
                                       seg["Low"].values, seg["Close"].values, complete=(len(seg) == HORIZON))
        lg.at[idx, "entry_date"] = df.index[pos]
        lg.at[idx, "entry_price"] = entry
        lg.at[idx, "tp"] = entry + TP_ATR * float(row["atr"])
        lg.at[idx, "sl"] = entry - SL_ATR * float(row["atr"])
        lg.at[idx, "status"] = status
        if status != "OPEN":
            lg.at[idx, "exit_date"] = seg.index[i]
            lg.at[idx, "exit_price"] = px
            lg.at[idx, "R_net"] = r_multiple(entry, px, float(row["atr"]))
            lg.at[idx, "days_held"] = i + 1
    return lg


def log_summary(lg: pd.DataFrame) -> str:
    closed = lg[lg["status"].isin(["TP", "SL", "TIMEOUT"])]
    n_open = int(lg["status"].isin(["PENDING", "OPEN"]).sum())
    if closed.empty:
        return f"Forward-test: belum ada trade selesai ({n_open} berjalan)."
    r = closed["R_net"].astype(float)
    g, ls = r[r > 0].sum(), -r[r < 0].sum()
    pf = g / ls if ls > 0 else float("inf")
    return (f"Forward-test: {len(closed)} trade selesai | TP {int((closed.status == 'TP').sum())} / "
            f"SL {int((closed.status == 'SL').sum())} / TO {int((closed.status == 'TIMEOUT').sum())} | "
            f"winrate(TP) {100 * (closed.status == 'TP').mean():.0f}% | expectancy {r.mean():+.2f}R | "
            f"PF {pf:.2f} | {n_open} berjalan")


# =============================================================================
# PEMILIHAN SINYAL LIVE
# =============================================================================
def pick_signals(cands: pd.DataFrame, frames: dict, open_tickers: set) -> pd.DataFrame:
    cands = cands[~cands["ticker"].isin(open_tickers)].sort_values("p", ascending=False)
    chosen = []
    if cands.empty:
        return cands
    rets = pd.concat({t: frames[t]["Close"].pct_change().tail(60) for t in cands["ticker"]}, axis=1)
    corr = rets.corr()
    for _, row in cands.iterrows():
        t = row["ticker"]
        if any(corr.loc[t, s] > CORR_MAX for s in chosen if t in corr.index and s in corr.columns):
            continue
        chosen.append(t)
        if len(chosen) >= MAX_SIGNALS:
            break
    return cands[cands["ticker"].isin(chosen)].set_index("ticker").loc[chosen].reset_index()


def trade_plan(row) -> dict:
    close, atr = float(row["Close"]), float(row["ATR14"])
    stop = round_tick(close - SL_ATR * atr, "down")
    target = round_tick(close + TP_ATR * atr, "down")
    max_entry = round_tick(close + ENTRY_CHASE_ATR * atr, "down")
    risk = close - stop
    return dict(close=close, atr=atr, stop=stop, target=target, max_entry=max_entry,
                rr=(target - close) / risk if risk > 0 else np.nan, risk_pct=100 * risk / close)


# =============================================================================
# MAIN
# =============================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="cetak pesan; jangan kirim Telegram / tulis log")
    ap.add_argument("--log", default=LOG_PATH)
    ap.add_argument("--force", action="store_true", help="abaikan cek 'bursa libur / data belum update'")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # ---------------- 1. DATA ----------------
    tickers = [t + ".JK" for t in KOMPAS100]
    log.info("Mengunduh %d saham...", len(tickers))
    frames, failed = fetch_universe(tickers)
    ok_ratio = len(frames) / len(tickers)
    log.info("Data OK: %d/%d | gagal: %s", len(frames), len(tickers), failed)
    if ok_ratio < MIN_UNIVERSE_OK:
        msg = (f"❌ <b>Scanner V8 GAGAL</b>: data hanya {len(frames)}/{len(tickers)} saham "
               f"(gagal: {html.escape(', '.join(failed[:30]))}). Ini kegagalan DATA, bukan 'tidak ada sinyal'.")
        send_telegram(msg, args.dry_run)
        return 1

    try:
        ihsg = drop_partial_bar(fetch_close("^JKSE"))
    except Exception as e:                                  # noqa: BLE001
        send_telegram(f"❌ <b>Scanner V8 GAGAL</b>: data IHSG tidak bisa diambil ({html.escape(str(e))}).", args.dry_run)
        return 1
    sp = idr_s = None
    for name, sym in (("S&P500", "^GSPC"), ("USDIDR", "USDIDR=X")):
        try:
            s = fetch_close(sym)
            if name == "S&P500":
                sp = s
            else:
                idr_s = s
        except Exception as e:                              # noqa: BLE001
            log.warning("Data %s gagal (%s) - fitur terkait diisi NaN.", name, e)

    global_last = max(df.index[-1] for df in frames.values())
    first_dates = pd.Series({t: df.index[0] for t, df in frames.items()})
    log.info("Fetch %s WIB | rentang data: %s s/d %s (median awal %s, %d saham dengan histori >= 6,5 th)",
             datetime.now(WIB).strftime("%Y-%m-%d %H:%M"), first_dates.min().date(), global_last.date(),
             first_dates.median().date(), int((first_dates <= global_last - pd.DateOffset(years=6, months=6)).sum()))
    today_wib = datetime.now(WIB).date()
    if datetime.now(WIB).weekday() < 5 and global_last.date() < today_wib and not args.force:
        msg = (f"ℹ️ <b>Scanner V8</b>: bursa libur / data Yahoo belum diperbarui "
               f"(bar terakhir {global_last:%d %b %Y}, hari ini {today_wib:%d %b %Y}). Tidak ada sinyal dikirim.")
        send_telegram(msg, args.dry_run)
        return 0
    stale = [t for t, df in frames.items() if df.index[-1] < global_last]
    if stale:
        log.warning("Bar terakhir tertinggal (suspensi/delisting?): %s", stale)

    dropped, _ = speculative_filter(frames)
    for t in dropped:
        frames.pop(t, None)
    log.info("Filter perilaku spekulatif: %d dibuang -> %s", len(dropped), dropped)

    # ---------------- 2-4. FITUR, LABEL, PANEL ----------------
    macro = build_macro(ihsg, sp, idr_s, frames)
    panel = build_panel(frames, macro)
    last_date = panel["date"].max()

    labeled = panel[panel["Elig_Stock"] & panel["Y"].notna()].dropna(subset=REQUIRED)
    log.info("Panel: %d baris | training eligible: %d | base rate TP: %.1f%%",
             len(panel), len(labeled), 100 * labeled["Y"].mean() if len(labeled) else float("nan"))

    # ---------------- 2. VALIDASI WALK-FORWARD ----------------
    oof, wf = walk_forward(labeled)
    best = base = None
    if oof is not None:
        log.info("\n%s", wf.round(3).to_string(index=False))
        best, base = select_threshold(oof)

    lg = load_log(args.log)
    lg = update_log(lg, frames)
    header = [f"📊 <b>KOMPAS100 Swing Scanner V8</b> — data s/d {last_date:%d %b %Y}",
              f"Data: {len(frames)}/{len(tickers)} saham OK"
              + (f" | gagal: {html.escape(', '.join(failed))}" if failed else "")
              + (f" | tertinggal: {html.escape(', '.join(stale))}" if stale else "")]

    live = panel[panel["date"] == last_date].dropna(subset=REQUIRED)
    regime_ok = bool(live["Regime_OK"].iloc[0]) if len(live) else False
    ctx = (f"Regime: IHSG vs MA50 {100 * float(live['IHSG_vs_MA50'].iloc[0]):+.1f}% | "
           f"breadth>MA50 {100 * float(live['Breadth50'].iloc[0]):.0f}% → "
           f"{'OK' if regime_ok else 'RISK-OFF'}") if len(live) else "Regime: n/a"
    header.append(ctx)

    # ---------------- gerbang validasi model ----------------
    if best is None:
        why = "walk-forward tidak bisa dijalankan (data kurang)" if oof is None else (
            f"tidak ada threshold dengan expectancy ≥ {MIN_EXP_R:+.2f}R & ≥ {MIN_OOF_TRADES} trade "
            f"(baseline semua kandidat: {base['expR']:+.2f}R, n={base['n']})")
        msg = "\n".join(header + ["", f"ℹ️ <b>Tidak ada sinyal.</b> Model belum lolos validasi: {html.escape(why)}.",
                                  "Ini hasil validasi, bukan error data.", "", log_summary(lg)])
        send_telegram(msg, args.dry_run)
        if not args.dry_run:
            lg.to_csv(args.log, index=False)
        return 0

    # kalibrasi probabilitas (untuk tampilan) dari out-of-fold
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(oof["p"], oof["Y"].astype(int))
    val = [f"Validasi OOF (thr={best['thr']:.3f}): {best['n']} trade | TP {100 * best['tp']:.0f}% / "
           f"SL {100 * best['sl']:.0f}% / TO {100 * best['to']:.0f}% | expectancy {best['expR']:+.2f}R | PF {best['pf']:.2f}",
           f"Baseline tanpa model: expectancy {base['expR']:+.2f}R (n={base['n']})"]

    if REGIME_GATE and not regime_ok:
        msg = "\n".join(header + val + ["", "⛔ <b>Regime RISK-OFF</b> — tidak ada sinyal beli hari ini.", "", log_summary(lg)])
        send_telegram(msg, args.dry_run)
        if not args.dry_run:
            lg.to_csv(args.log, index=False)
        return 0

    # ---------------- model final & sinyal live ----------------
    model = fit_model(labeled[FEATURES], labeled["Y"].astype(int))      # [FIX] baris live TIDAK ikut dilatih
    imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    log.info("Feature importance:\n%s", imp.round(3).to_string())

    elig = live[live["Elig_Stock"]].copy()
    elig["p"] = model.predict_proba(elig[FEATURES])[:, 1] if len(elig) else []
    cands = elig[elig["p"] >= best["thr"]]
    open_t = set(lg.loc[lg["status"].isin(["PENDING", "OPEN"]), "ticker"])
    sig = pick_signals(cands, frames, open_t)
    log.info("Eligible: %d | lolos threshold: %d | final: %d", len(elig), len(cands), len(sig))

    if sig.empty:
        msg = "\n".join(header + val + ["", f"ℹ️ <b>Tidak ada sinyal hari ini</b> ({len(elig)} saham eligible, "
                                          f"{len(cands)} lolos threshold sebelum filter posisi/korelasi).", "", log_summary(lg)])
        send_telegram(msg, args.dry_run)
        if not args.dry_run:
            lg.to_csv(args.log, index=False)
        return 0

    lines = header + val + ["", f"🚨 <b>{len(sig)} sinyal</b> (entry di open besok; time stop {HORIZON} hari bursa)"]
    new_rows = []
    for _, r in sig.iterrows():
        tp_ = trade_plan(r)
        pc = float(iso.predict([r["p"]])[0])
        lines += [
            "", f"🔹 <b>{html.escape(r['ticker'].replace('.JK', ''))}</b>  | skor {r['p']:.2f} (kalibrasi OOF ≈ {100 * pc:.0f}% TP)",
            f"   Close: <code>Rp {idr(tp_['close'])}</code>  | batas entry: <code>≤ Rp {idr(tp_['max_entry'])}</code>",
            f"   Stop: <code>Rp {idr(tp_['stop'])}</code> (−{tp_['risk_pct']:.1f}%)  | Target: <code>Rp {idr(tp_['target'])}</code>",
            f"   RR {tp_['rr']:.1f}:1  | ATR {100 * tp_['atr'] / tp_['close']:.1f}%  | RS-rank {100 * float(r['RS_Rank']):.0f}%",
        ]
        new_rows.append(dict(signal_date=last_date, ticker=r["ticker"], prob=round(float(r["p"]), 4),
                             close=tp_["close"], atr=tp_["atr"], entry_date=pd.NaT, entry_price=np.nan,
                             tp=np.nan, sl=np.nan, status="PENDING", exit_date=pd.NaT,
                             exit_price=np.nan, R_net=np.nan, days_held=np.nan))
    lines += ["", "Aturan: jangan kejar jika open > batas entry; geser stop ke breakeven setelah +1×ATR; "
                  "keluar di close hari ke-" + str(HORIZON) + " jika belum kena TP/SL.", "", log_summary(lg)]
    send_telegram("\n".join(lines), args.dry_run)

    if not args.dry_run:
        new = pd.DataFrame(new_rows, columns=LOG_COLS)
        if len(lg):
            dup = lg.set_index(["ticker", "signal_date"]).index
            new = new[~new.set_index(["ticker", "signal_date"]).index.isin(dup)]
        lg = pd.concat([lg, new], ignore_index=True) if len(new) else lg
        lg.to_csv(args.log, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
