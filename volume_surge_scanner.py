"""
NSE Volume Surge Scanner
------------------------
Scans the equity market every minute (09:15–15:30 IST) and lists stocks trading
on abnormally high volume versus their 20-day average, adjusted for time of day.

Run on your own machine:
    pip install requests pandas
    python volume_surge_scanner.py

Options:
    --index "NIFTY 500"            universe (default). Others: "NIFTY TOTAL MARKET",
                                   "SECURITIES IN F&O", "NIFTY 200", "NIFTY MIDCAP 150"
    --rvol 3.0                     volume condition: today's cum. volume >= 3x expected-by-now
    --spike 5.0                    or last-15-min volume >= 5x the median of this stock's earlier
                                   15-min windows today, with the day's RVOL >= --spike-rvol (1.5)
                                   and the 15-min candle >= --candle (1%)
    --fast 8.0                     or FAST: last-5-min volume >= 8x the median 5-min bar today,
                                   >= 2% of a normal day's volume, and the 5-min candle >= --candle.
                                   Catches a burst within minutes instead of waiting 15.
    --candle 1.0                   candle size (%) the spike/fast volume must come with. Block
                                   deals print huge volume with no candle; this is what tells a
                                   momentum burst from a block deal. Applies to spike and fast
                                   triggers only; the RVOL trigger stays volume-only.
    --min-move 0                   optional price filter: |change vs prev close| >= N%.
                                   Default 0 = volume only, price not considered.
    --lookback 20                  days of history for "normal" volume
    --once                         run a single scan and exit (for testing)
    --web                          also serve a live dashboard at http://localhost:8765
                                   (auto-refreshes every minute; open it in any browser/phone
                                   on the same Wi-Fi via your laptop's IP)
    --until 12:30                  stop at this IST time instead of 15:30 (for split runs)
    --no-clear                     don't clear the terminal between scans (for log files / CI)
    --telegram                     send new surges to Telegram; needs env vars
                                   TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID
    --source auto|nse|yahoo        live-quote source. auto (default) tries NSE's site and
                                   falls back to Yahoo Finance if NSE refuses (datacenter
                                   IPs such as GitHub runners). Yahoo quotes may lag a
                                   few minutes; NSE's are live.

Outputs:
    terminal table every minute
    alerts_YYYY-MM-DD.csv           every stock that was flagged, each minute
    snapshots_YYYY-MM-DD.csv        full cum-volume snapshot each minute (useful to
                                    calibrate the intraday profile later)
"""

import argparse
import html
import io
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
import time
import zipfile
from http.server import HTTPServer, SimpleHTTPRequestHandler
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

IST_OFFSET = timedelta(hours=5, minutes=30)
MARKET_OPEN = (9, 15)
SCAN_END = (15, 30)
MARKET_CLOSE_MIN = 375
DAY_MINS = MARKET_CLOSE_MIN  # 09:15 -> 15:30

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}

# Fitted from 60 days of 5-min NIFTY 500 data (Sep–Oct 2026): median share of a
# day's volume traded by N minutes after 09:15. Re-fit with backtest_volume_surge.py.
PROFILE = [(0, 0.0), (15, 0.068), (30, 0.12), (45, 0.165), (60, 0.207), (75, 0.244),
           (90, 0.279), (105, 0.313), (120, 0.348), (135, 0.382), (150, 0.415),
           (165, 0.449), (180, 0.481), (195, 0.513), (210, 0.543), (225, 0.575),
           (240, 0.605), (255, 0.636), (270, 0.668), (285, 0.7), (300, 0.735),
           (315, 0.773), (330, 0.818), (345, 0.887), (360, 0.967), (375, 1.0)]


def now_ist():
    return datetime.now(timezone.utc).replace(tzinfo=None) + IST_OFFSET


def minutes_since_open(t):
    return (t.hour - MARKET_OPEN[0]) * 60 + (t.minute - MARKET_OPEN[1])


def expected_fraction(mins):
    """Fraction of daily volume expected to have traded by `mins` after open."""
    if mins <= 0:
        return 0.0
    if mins >= MARKET_CLOSE_MIN:
        return 1.0
    for (m0, f0), (m1, f1) in zip(PROFILE, PROFILE[1:]):
        if m0 <= mins <= m1:
            return f0 + (f1 - f0) * (mins - m0) / (m1 - m0)
    return 1.0


# ----------------------------------------------------------------------------
# NSE session
# ----------------------------------------------------------------------------
class NSE:
    def __init__(self):
        self.s = requests.Session()
        self.s.headers.update(HEADERS)
        self._warm()

    def _warm(self):
        # NSE needs cookies set by the homepage before API calls work.
        self.s.get("https://www.nseindia.com", timeout=10)
        time.sleep(0.5)

    def get_json(self, url, retries=3):
        for i in range(retries):
            try:
                r = self.s.get(url, timeout=15)
                if r.status_code == 200 and r.text.strip():
                    return r.json()
            except (requests.RequestException, json.JSONDecodeError):
                pass
            self._warm()
            time.sleep(1 + i)
        raise RuntimeError(f"NSE request failed: {url}")

    def live_index(self, index):
        url = "https://www.nseindia.com/api/equity-stockIndices?index=" + requests.utils.quote(index)
        data = self.get_json(url)["data"]
        rows = []
        for d in data:
            sym = d.get("symbol")
            if not sym or sym == index or d.get("priority") == 1:
                continue
            rows.append({
                "symbol": sym,
                "ltp": d.get("lastPrice"),
                "pchg": d.get("pChange"),
                "cum_vol": d.get("totalTradedVolume") or 0,
            })
        return pd.DataFrame(rows).set_index("symbol")

    def bhavcopy(self, d):
        """Daily bhavcopy -> Series symbol -> traded quantity. Returns None on holidays."""
        ymd = d.strftime("%Y%m%d")
        urls = [
            f"https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{ymd}_F_0000.csv.zip",
            f"https://archives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{ymd}_F_0000.csv.zip",
        ]
        for url in urls:
            try:
                r = self.s.get(url, timeout=20)
            except requests.RequestException:
                continue
            if r.status_code != 200 or not r.content.startswith(b"PK"):
                continue
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                name = z.namelist()[0]
                df = pd.read_csv(z.open(name))
            df = df[df["SctySrs"].isin(["EQ", "BE"])]
            return df.set_index("TckrSymb")["TtlTradgVol"].astype(float)
        return None



# ----------------------------------------------------------------------------
# Yahoo Finance fallback for live quotes (used when NSE's live API is blocked)
# ----------------------------------------------------------------------------
INDEX_LIST_FILES = {
    "NIFTY 500": "ind_nifty500list.csv", "NIFTY 50": "ind_nifty50list.csv",
    "NIFTY 200": "ind_nifty200list.csv", "NIFTY MIDCAP 150": "ind_niftymidcap150list.csv",
    "NIFTY TOTAL MARKET": "ind_niftytotalmarket_list.csv",
}


def index_symbols(nse, index):
    """Constituents of an NSE index: from NSE's archive, else a local <index>.csv (Symbol column)."""
    fn = INDEX_LIST_FILES.get(index.upper())
    if fn:
        for host in ("https://nsearchives.nseindia.com", "https://archives.nseindia.com"):
            try:
                r = nse.s.get(f"{host}/content/indices/{fn}", timeout=20)
                if r.status_code == 200 and "Symbol" in r.text[:500]:
                    return pd.read_csv(io.StringIO(r.text))["Symbol"].str.strip().tolist()
            except requests.RequestException:
                pass
    local = index.lower().replace(" ", "") + ".csv"
    if os.path.exists(local):
        return pd.read_csv(local)["Symbol"].str.strip().tolist()
    raise RuntimeError(f"No constituent list for {index}: NSE archive unreachable and {local} not found")


class YahooLive:
    """Same live_index() contract as NSE, built on Yahoo's batch quote endpoint."""

    def __init__(self, symbols):
        self.symbols = symbols
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": HEADERS["User-Agent"], "Accept": "application/json"})
        self.crumb = None
        self._auth()

    def _auth(self):
        try:
            self.s.get("https://fc.yahoo.com", timeout=10)  # sets cookies (404 is expected)
            r = self.s.get("https://query1.finance.yahoo.com/v1/test/getcrumb", timeout=10)
            self.crumb = r.text.strip() if r.status_code == 200 and r.text and "<" not in r.text else None
        except requests.RequestException:
            self.crumb = None

    def live_index(self, index):
        rows = self._quote_batch() if self.crumb else None
        if not rows:
            rows = self._chart_all()
        if not rows:
            raise RuntimeError("Yahoo returned no quotes")
        return pd.DataFrame(rows).set_index("symbol")

    def _quote_batch(self):
        """v7 batch quote: 100 symbols per request. Needs the crumb cookie."""
        rows = []
        for i in range(0, len(self.symbols), 100):
            batch = self.symbols[i:i + 100]
            params = {"symbols": ",".join(s + ".NS" for s in batch), "crumb": self.crumb,
                      "fields": "regularMarketPrice,regularMarketChangePercent,regularMarketVolume"}
            try:
                r = self.s.get("https://query1.finance.yahoo.com/v7/finance/quote", params=params, timeout=15)
                if r.status_code != 200:
                    if r.status_code in (401, 403):
                        self.crumb = None
                    return None
                for q in r.json()["quoteResponse"]["result"]:
                    rows.append({"symbol": q.get("symbol", "").removesuffix(".NS"), "ltp": q.get("regularMarketPrice"),
                                 "pchg": q.get("regularMarketChangePercent"), "cum_vol": q.get("regularMarketVolume") or 0})
            except (requests.RequestException, ValueError, KeyError):
                return None
        return rows

    def _chart_one(self, sym):
        url = "https://query1.finance.yahoo.com/v8/finance/chart/" + requests.utils.quote(sym + ".NS", safe="")
        for attempt in range(2):
            try:
                r = self.s.get(url, params={"range": "1d", "interval": "1d"}, timeout=10)
                if r.status_code == 200:
                    m = r.json()["chart"]["result"][0]["meta"]
                    px, prev = m.get("regularMarketPrice"), m.get("chartPreviousClose")
                    pchg = (px / prev - 1) * 100 if px and prev else None
                    return {"symbol": sym, "ltp": px, "pchg": pchg, "cum_vol": m.get("regularMarketVolume") or 0}
                if r.status_code == 429:
                    time.sleep(2 + attempt * 3)
                else:
                    return None
            except (requests.RequestException, ValueError, KeyError, IndexError, TypeError):
                time.sleep(1)
        return None

    def _chart_all(self):
        """Per-symbol chart endpoint (no crumb needed), ~15 s for 500 symbols with a thread pool."""
        with ThreadPoolExecutor(max_workers=12) as ex:
            out = list(ex.map(self._chart_one, self.symbols))
        rows = [o for o in out if o]
        missing = len(out) - len(rows)
        if missing:
            print(f"[yahoo] {missing} symbols returned no quote this minute")
        return rows

# ----------------------------------------------------------------------------
# Baseline volume
# ----------------------------------------------------------------------------
def load_baseline(nse, lookback, cache_dir="cache"):
    os.makedirs(cache_dir, exist_ok=True)
    today = now_ist().date()
    cache = os.path.join(cache_dir, f"baseline_{today}_{lookback}.csv")
    if os.path.exists(cache):
        s = pd.read_csv(cache, index_col=0).iloc[:, 0]
        print(f"[baseline] loaded {len(s)} symbols from cache")
        return s

    frames, d, tries = [], today - timedelta(days=1), 0
    print(f"[baseline] downloading {lookback} daily bhavcopies...", flush=True)
    while len(frames) < lookback and tries < lookback * 3:
        tries += 1
        if d.weekday() < 5:
            s = nse.bhavcopy(d)
            if s is not None:
                frames.append(s.rename(str(d)))
                print(f"  {d} ok ({len(s)} symbols)", flush=True)
        d -= timedelta(days=1)
    if not frames:
        raise RuntimeError("Could not download any bhavcopy — check network / NSE archive URL.")
    hist = pd.concat(frames, axis=1)
    avg = hist.mean(axis=1, skipna=True)
    avg.to_csv(cache)
    print(f"[baseline] {len(frames)} days, {len(avg)} symbols")
    return avg


# ----------------------------------------------------------------------------
# Scanner
# ----------------------------------------------------------------------------
class Scanner:
    def __init__(self, args):
        self.args = args
        self.nse = NSE()
        self.avg = load_baseline(self.nse, args.lookback)
        self.live = self.nse
        self.source = "nse"
        if args.source == "yahoo":
            self.use_yahoo("requested")
        self.history = []  # list of (minute, cum_vol Series, ltp Series), last ~4 hours
        self.windows = []  # (end_minute, 15-min volume Series): spike baseline
        self.bars5 = []    # (end_minute, 5-min volume Series): fast-trigger baseline
        today = now_ist().date()
        self.alerts_file = f"alerts_{today}.csv"
        self.snap_file = f"snapshots_{today}.csv"
        self.seen = set()  # symbols already alerted today (Telegram dedupe)
        if args.telegram:
            self.tg_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
            self.tg_chat = os.environ.get("TELEGRAM_CHAT_ID", "")
            if not (self.tg_token and self.tg_chat):
                print("[telegram] TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set — alerts disabled")
                args.telegram = False
            else:
                self.telegram(f"Volume surge scanner started ({args.index}, {today})")

    def use_yahoo(self, why):
        syms = [s for s in index_symbols(self.nse, self.args.index) if s in self.avg.index]
        self.live = YahooLive(syms)
        self.source = "yahoo"
        print(f"[source] switched to Yahoo Finance quotes ({len(syms)} symbols): {why}")
        if self.args.telegram:
            self.telegram(f"Live source: Yahoo Finance ({why}); quotes may lag a few minutes.")

    def telegram(self, text):
        try:
            requests.post(f"https://api.telegram.org/bot{self.tg_token}/sendMessage",
                          json={"chat_id": self.tg_chat, "text": text, "parse_mode": "HTML"}, timeout=10)
        except requests.RequestException as e:
            print(f"[telegram] send failed: {e}", file=sys.stderr)

    def notify_new(self, t, flagged):
        new = flagged[~flagged.index.isin(self.seen)]
        self.seen.update(new.index)
        if new.empty:
            return
        new = new.sort_values("rvol", ascending=False)
        lines = [f"<b>{t:%H:%M}</b> — {len(new)} new surge{'s' if len(new) > 1 else ''}"]
        for sym, r in new.head(15).iterrows():
            if r["why"] == "FAST":
                why = f" · ⚡5m {r['fast']:.0f}x ({r['move_5m']:+.1f}%)"
            elif r["why"] == "15m":
                why = f" · 15m {r['spike']:.0f}x ({r['move_15m']:+.1f}%)"
            else:
                why = "" if pd.isna(r["spike"]) else f" · 15m {r['spike']:.1f}x"
            lines.append(f"<b>{sym}</b> {r['pchg']:+.1f}% · RVOL {r['rvol']:.1f}x{why} · ₹{r['ltp']}")
        if len(new) > 15:
            lines.append(f"…and {len(new) - 15} more")
        self.telegram("\n".join(lines))

    def scan(self):
        t = now_ist()
        mins = minutes_since_open(t)
        try:
            live = self.live.live_index(self.args.index)
        except RuntimeError as e:
            if self.source == "nse" and self.args.source == "auto":
                self.use_yahoo(f"NSE live API refused: {e}")
                live = self.live.live_index(self.args.index)
            else:
                raise
        live = live.join(self.avg.rename("avg_vol"), how="left").dropna(subset=["avg_vol"])
        live = live[live["avg_vol"] > 0]

        frac_now = expected_fraction(mins)
        live["expected_cum"] = live["avg_vol"] * frac_now
        live["rvol"] = live["cum_vol"] / live["expected_cum"].clip(lower=1)

        # 15-minute spike: last 15 min of volume vs the median of this stock's own earlier
        # 15-min windows today (what the eye sees on a 15m chart: "this bar is 6x its neighbours")
        self.history.append((mins, live["cum_vol"].copy(), live["ltp"].copy()))
        self.history = [h for h in self.history if mins - h[0] <= 240]
        live["vol_15m"] = float("nan")
        live["spike"] = float("nan")
        live["move_15m"] = float("nan")
        live["vol_5m"] = float("nan")
        live["fast"] = float("nan")
        live["move_5m"] = float("nan")
        prior = [h for h in self.history if mins - h[0] >= 15]
        if prior:
            m_prev, cum_prev, ltp_prev = prior[-1]
            last15 = (live["cum_vol"] - cum_prev.reindex(live.index).fillna(0)).clip(lower=0)
            last15 = last15 * (15 / max(mins - m_prev, 1))  # normalise if the gap isn't exactly 15
            live["vol_15m"] = last15
            live["move_15m"] = (live["ltp"] / ltp_prev.reindex(live.index) - 1) * 100
            # record one completed, non-overlapping window every 15 minutes (ignore the opening 15)
            if m_prev >= 15 and (not self.windows or mins - self.windows[-1][0] >= 15):
                self.windows.append((mins, last15.copy()))
                self.windows = self.windows[-24:]
            earlier = [w for w in self.windows if w[0] <= mins - 15]
            if len(earlier) >= 3:
                base = pd.concat([w[1] for w in earlier], axis=1).median(axis=1)
                base = base.reindex(live.index)
                floor = live["avg_vol"] / DAY_MINS * 15 * 0.25  # avoid div-by-tiny on illiquid names
                live["spike"] = last15 / base.clip(lower=floor.clip(lower=1))
        # FAST trigger: last 5 minutes vs this stock's median 5-min bar today
        prior5 = [h for h in self.history if mins - h[0] >= 5]
        if prior5:
            m5, cum5, ltp5 = prior5[-1]
            last5 = (live["cum_vol"] - cum5.reindex(live.index).fillna(0)).clip(lower=0) * (5 / max(mins - m5, 1))
            live["vol_5m"] = last5
            live["move_5m"] = (live["ltp"] / ltp5.reindex(live.index) - 1) * 100
            if m5 >= 15 and (not self.bars5 or mins - self.bars5[-1][0] >= 5):
                self.bars5.append((mins, last5.copy()))
                self.bars5 = self.bars5[-72:]
            earlier5 = [b for b in self.bars5 if b[0] <= mins - 5]
            if len(earlier5) >= 6:
                med5 = pd.concat([b[1] for b in earlier5], axis=1).median(axis=1).reindex(live.index)
                floor5 = live["avg_vol"] / DAY_MINS * 5 * 0.25
                live["fast"] = last5 / med5.clip(lower=floor5.clip(lower=1))

        # snapshot log (for later calibration of PROFILE)
        snap = live[["cum_vol"]].T
        snap.index = [t.strftime("%H:%M")]
        snap.to_csv(self.snap_file, mode="a", header=not os.path.exists(self.snap_file))

        spike_ok = ((live["spike"] >= self.args.spike) & (live["rvol"] >= self.args.spike_rvol)
                    & (live["move_15m"].abs() >= self.args.candle))
        fast_ok = ((live["fast"] >= self.args.fast) & (live["vol_5m"] >= 0.02 * live["avg_vol"])
                   & (live["move_5m"].abs() >= self.args.candle))
        live["why"] = np.select([live["rvol"] >= self.args.rvol, fast_ok, spike_ok], ["RVOL", "FAST", "15m"], "")
        vol_ok = (live["rvol"] >= self.args.rvol) | spike_ok | fast_ok
        move_ok = live["pchg"].abs() >= self.args.min_move
        flagged = live[vol_ok & move_ok]
        flagged = flagged.assign(_k=flagged["fast"].fillna(0).clip(upper=100) + flagged["rvol"]).sort_values("_k", ascending=False).drop(columns="_k")
        self.show(t, mins, flagged, len(live))
        if self.args.web:
            self.write_html(t, mins, flagged, len(live))
        if self.args.telegram:
            self.notify_new(t, flagged)
        if not flagged.empty:
            out = flagged.copy()
            out.insert(0, "time", t.strftime("%H:%M"))
            out.to_csv(self.alerts_file, mode="a", header=not os.path.exists(self.alerts_file))
        return flagged

    def show(self, t, mins, flagged, n):
        if not self.args.no_clear:
            os.system("cls" if os.name == "nt" else "clear")
        print(f"NSE VOLUME SURGE  {t:%Y-%m-%d %H:%M} IST   universe={self.args.index} ({n} stocks, via {self.source})   "
              f"expected {expected_fraction(mins):.0%} of day traded by now")
        mv = f"  and  |move| >= {self.args.min_move}%" if self.args.min_move > 0 else ""
        print(f"flag if RVOL >= {self.args.rvol}x  or  15m spike >= {self.args.spike}x (RVOL >= {self.args.spike_rvol}x)  "
              f"or  FAST 5m >= {self.args.fast}x; spike/fast need a >= {self.args.candle}% candle{mv}\n")
        if flagged.empty:
            print("  no surges this minute")
            return
        cols = ["why", "ltp", "pchg", "cum_vol", "avg_vol", "rvol", "vol_15m", "spike", "fast", "move_5m"]
        fmt = flagged[cols].copy()
        fmt["cum_vol"] = fmt["cum_vol"].map(lambda x: f"{x/1e5:.1f}L")
        fmt["avg_vol"] = fmt["avg_vol"].map(lambda x: f"{x/1e5:.1f}L")
        fmt["vol_15m"] = fmt["vol_15m"].map(lambda x: "-" if pd.isna(x) else f"{x/1e5:.1f}L")
        fmt["rvol"] = fmt["rvol"].map(lambda x: f"{x:.1f}x")
        fmt["spike"] = fmt["spike"].map(lambda x: "-" if pd.isna(x) else f"{x:.1f}x")
        fmt["fast"] = fmt["fast"].map(lambda x: "-" if pd.isna(x) else f"{x:.0f}x")
        fmt["move_5m"] = fmt["move_5m"].map(lambda x: "-" if pd.isna(x) else f"{x:+.1f}%")
        fmt["pchg"] = fmt["pchg"].map(lambda x: f"{x:+.2f}%")
        fmt.columns = ["Why", "LTP", "Chg", "Vol today", "Avg day", "RVOL", "Vol 15m", "Spike", "5m", "5m chg"]
        print(fmt.head(40).to_string())
        if len(flagged) > 40:
            print(f"  ... and {len(flagged) - 40} more (see {self.alerts_file})")

    def write_html(self, t, mins, flagged, n):
        rows = []
        for sym, r in flagged.iterrows():
            up = (r["pchg"] or 0) >= 0
            spike = "-" if pd.isna(r["spike"]) else f'{r["spike"]:.1f}x'
            rows.append(
                f"<tr><td class=sym>{html.escape(sym)}</td><td>{r['ltp']}</td>"
                f"<td class={'up' if up else 'dn'}>{(r['pchg'] or 0):+.2f}%</td>"
                f"<td>{r['cum_vol']/1e5:.1f}L</td><td>{r['avg_vol']/1e5:.1f}L</td>"
                f"<td class=hot>{r['rvol']:.1f}x</td><td>{spike}</td></tr>"
            )
        body = "".join(rows) or "<tr><td colspan=7 class=none>No surges this minute</td></tr>"
        page = f"""<!doctype html><html><head><meta charset=utf-8>
<meta http-equiv=refresh content=60><meta name=viewport content="width=device-width,initial-scale=1">
<title>NSE Volume Surge</title>
<style>
body{{font:14px/1.4 system-ui,sans-serif;background:#0f1115;color:#e6e6e6;margin:0;padding:16px}}
h1{{font-size:18px;margin:0 0 4px}} .meta{{color:#8b93a1;font-size:13px;margin-bottom:12px}}
table{{border-collapse:collapse;width:100%}} th,td{{padding:7px 8px;text-align:right;border-bottom:1px solid #22262e}}
th{{color:#8b93a1;font-weight:500;font-size:12px;text-transform:uppercase}} td.sym,th.sym{{text-align:left;font-weight:600}}
.up{{color:#3ddc84}} .dn{{color:#ff5c5c}} .hot{{color:#ffb347;font-weight:600}} .none{{text-align:center;color:#8b93a1;padding:30px}}
</style></head><body>
<h1>NSE Volume Surge</h1>
<div class=meta>{t:%d %b %Y %H:%M} IST &middot; {html.escape(self.args.index)} ({n} stocks) &middot;
{expected_fraction(mins):.0%} of a normal day's volume expected by now &middot;
flag: RVOL &ge; {self.args.rvol}x or (15-min spike &ge; {self.args.spike}x and RVOL &ge; {self.args.spike_rvol}x){(' and |move| &ge; %g%%' % self.args.min_move) if self.args.min_move > 0 else ''} &middot; {len(flagged)} flagged</div>
<table><tr><th class=sym>Symbol</th><th>LTP</th><th>Chg</th><th>Vol today</th><th>Avg day</th><th>RVOL</th><th>15-min</th></tr>
{body}</table></body></html>"""
        with open("dashboard.html", "w", encoding="utf-8") as f:
            f.write(page)


class _Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *a):  # keep the terminal table clean
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.path = "/dashboard.html"
        return super().do_GET()


def start_web(port=8765):
    with open("dashboard.html", "w", encoding="utf-8") as f:
        f.write("<meta http-equiv=refresh content=10><body style='font:14px system-ui;padding:20px'>"
                "Scanner starting — waiting for first scan…</body>")
    srv = HTTPServer(("0.0.0.0", port), _Quiet)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"[web] dashboard at http://localhost:{port}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index", default="NIFTY 500")
    p.add_argument("--rvol", type=float, default=3.0)
    p.add_argument("--spike", type=float, default=5.0)
    p.add_argument("--spike-rvol", type=float, default=1.5)
    p.add_argument("--fast", type=float, default=8.0)
    p.add_argument("--candle", type=float, default=1.0)
    p.add_argument("--min-move", type=float, default=0.0)
    p.add_argument("--lookback", type=int, default=20)
    p.add_argument("--once", action="store_true")
    p.add_argument("--web", action="store_true")
    p.add_argument("--until", default=f"{SCAN_END[0]:02d}:{SCAN_END[1]:02d}")
    p.add_argument("--no-clear", action="store_true")
    p.add_argument("--telegram", action="store_true")
    p.add_argument("--source", choices=["auto", "nse", "yahoo"], default="auto")
    args = p.parse_args()

    if args.web:
        start_web()
    sc = Scanner(args)
    if args.once:
        sc.scan()
        return

    t = now_ist()
    open_t = t.replace(hour=MARKET_OPEN[0], minute=MARKET_OPEN[1], second=5)
    eh, em = (int(x) for x in args.until.split(":"))
    end_t = t.replace(hour=eh, minute=em, second=0)
    if t.weekday() >= 5:
        print("Weekend — market closed. Use --once to test with stale data.")
        return
    if t < open_t:
        print(f"Waiting for market open at {open_t:%H:%M} IST...")
        time.sleep((open_t - t).total_seconds())

    while now_ist() <= end_t:
        start = time.time()
        try:
            sc.scan()
        except Exception as e:  # keep running through transient NSE errors
            print(f"[{now_ist():%H:%M}] scan error: {e}", file=sys.stderr)
        time.sleep(max(0, 60 - (time.time() - start)))
    print(f"Scan window over ({args.until}). Alerts saved to", sc.alerts_file)
    if args.telegram:
        sc.telegram(f"Scanner stopped at {args.until}. {len(sc.seen)} symbols flagged this run.")


if __name__ == "__main__":
    main()