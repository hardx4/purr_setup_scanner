import os
import requests
import pandas as pd
import time

URL = "https://api.hyperliquid.xyz/info"
COIN = "PURR"
test = 1

MAX_CANDLES_PER_REQUEST = 4999
INTERVALO_SEGUNDOS = 60 * 60  # 1 hora entre varreduras

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

TIMEFRAMES = ["1d", "4h", "1h"]

TF_WEIGHT = {
    "1h": 1,
    "4h": 2,
    "1d": 3
}

INTERVAL_MS = {
    "1h": 60 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
    "1d": 24 * 60 * 60 * 1000,
}


def notify(text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram não configurado (faltam TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID)")
        return

    try:
        response = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": text,
                "parse_mode": "HTML"
            },
            timeout=15
        )
        response.raise_for_status()
    except Exception as e:
        print(f"Falha ao enviar Telegram: {e}")


def fetch_candle_page(interval, start_time, end_time):
    payload = {
        "type": "candleSnapshot",
        "req": {
            "coin": COIN,
            "interval": interval,
            "startTime": int(start_time),
            "endTime": int(end_time)
        }
    }

    response = requests.post(
        URL,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=30
    )
    response.raise_for_status()

    data = response.json()

    if not isinstance(data, list):
        raise Exception(f"Resposta inesperada: {data}")

    return data


def get_candles(interval, start_time, end_time=None):
    if end_time is None:
        end_time = int(time.time() * 1000)

    all_rows = []
    cursor = int(start_time)
    max_pages = 50

    for _ in range(max_pages):
        if cursor >= end_time:
            break

        page_end = min(
            end_time,
            cursor + MAX_CANDLES_PER_REQUEST * INTERVAL_MS[interval]
        )

        page = fetch_candle_page(interval, cursor, page_end)

        if not page:
            if page_end >= end_time:
                break
            cursor = page_end + 1
            continue

        all_rows.extend(page)

        last = page[-1]
        last_close = int(last.get("T", last["t"]))
        next_cursor = last_close + 1

        if next_cursor <= cursor:
            break

        cursor = next_cursor

        if len(page) < MAX_CANDLES_PER_REQUEST and page_end >= end_time:
            break

        time.sleep(0.15)

    if not all_rows:
        raise Exception("Nenhum candle retornado")

    df = pd.DataFrame(all_rows)

    df["open"] = pd.to_numeric(df["o"])
    df["high"] = pd.to_numeric(df["h"])
    df["low"] = pd.to_numeric(df["l"])
    df["close"] = pd.to_numeric(df["c"])
    df["volume"] = pd.to_numeric(df["v"])

    df["datetime"] = pd.to_datetime(
        pd.to_numeric(df["t"]),
        unit="ms",
        utc=True
    )

    df = df.sort_values("t")
    df = df.drop_duplicates(subset=["t"])

    current_price = float(df.iloc[-1]["close"])
    current_time = (
        df.iloc[-1]["datetime"]
        .strftime("%Y-%m-%d %H:%M UTC")
    )

    if len(df) > 1:
        df = df.iloc[:-1].copy()

    df.reset_index(drop=True, inplace=True)

    return df, current_price, current_time


def discover_1d_start():
    end_time = int(time.time() * 1000)
    start_time = end_time - 20 * 365 * 24 * 60 * 60 * 1000

    df, current_price, current_time = get_candles(
        "1d",
        start_time,
        end_time
    )

    first_ts = int(df.iloc[0]["t"])
    first_dt = df.iloc[0]["datetime"].strftime("%Y-%m-%d %H:%M UTC")

    print(
        f"📅 Histórico 1d disponível desde: {first_dt} "
        f"({first_ts})"
    )

    return first_ts, df, current_price, current_time


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss

    return 100 - (100 / (1 + rs))


def macd(series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - signal
    return macd_line, signal, hist


def atr(df, period=14):
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    return tr.ewm(alpha=1 / period, adjust=False).mean()


def adx(df, period=14):
    high = df["high"]
    low = df["low"]
    close = df["close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr_val = tr.ewm(alpha=1 / period, adjust=False).mean()

    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_val
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_val

    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di)) * 100
    adx_value = dx.ewm(alpha=1 / period, adjust=False).mean()

    return adx_value, plus_di, minus_di


def analyze(df):
    close = df["close"]

    df["RSI"] = rsi(close)
    df["MACD"], df["SIGNAL"], df["HIST"] = macd(close)

    df["EMA20"] = close.ewm(span=20, adjust=False).mean()
    df["EMA50"] = close.ewm(span=50, adjust=False).mean()
    df["EMA200"] = close.ewm(span=200, adjust=False).mean()
    df["ATR"] = atr(df)
    df["ADX"], df["PLUS_DI"], df["MINUS_DI"] = adx(df)

    df["ATR_PCT"] = (df["ATR"] / df["close"]) * 100
    df["VOL_MA20"] = df["volume"].rolling(20).mean()
    df["VOL_RATIO"] = df["volume"] / df["VOL_MA20"]

    df["EMA20_SLOPE"] = ((df["EMA20"] / df["EMA20"].shift(5)) - 1) * 100
    df["EMA50_SLOPE"] = ((df["EMA50"] / df["EMA50"].shift(5)) - 1) * 100
    df["PCT_EMA200"] = ((close - df["EMA200"]) / df["EMA200"]) * 100

    df["MACD_CROSS_UP"] = (
        (df["MACD"] > df["SIGNAL"])
        & (df["MACD"].shift(1) <= df["SIGNAL"].shift(1))
    )
    df["MACD_CROSS_DOWN"] = (
        (df["MACD"] < df["SIGNAL"])
        & (df["MACD"].shift(1) >= df["SIGNAL"].shift(1))
    )

    last = df.iloc[-1]

    score_buy = 0
    score_sell = 0

    if last["RSI"] > 55:
        score_buy += 1
    else:
        score_sell += 1

    if last["RSI"] > 65:
        score_buy += 1

    if last["RSI"] < 45:
        score_sell += 1

    if last["RSI"] < 35:
        score_sell += 1

    if last["MACD_CROSS_UP"]:
        score_buy += 2
    elif last["MACD"] > last["SIGNAL"]:
        score_buy += 1

    if last["MACD_CROSS_DOWN"]:
        score_sell += 2
    elif last["MACD"] < last["SIGNAL"]:
        score_sell += 1

    if last["close"] > last["EMA20"]:
        score_buy += 1
    else:
        score_sell += 1

    if last["EMA20"] > last["EMA50"]:
        score_buy += 1
    else:
        score_sell += 1

    if last["EMA50"] > last["EMA200"]:
        score_buy += 1
    else:
        score_sell += 1

    if last["EMA20_SLOPE"] > 0:
        score_buy += 1
    else:
        score_sell += 1

    if last["EMA50_SLOPE"] > 0:
        score_buy += 1
    else:
        score_sell += 1

    if last["VOL_RATIO"] > 1.5:
        score_buy += 1
    elif last["VOL_RATIO"] < 0.7:
        score_sell += 1

    if last["ATR_PCT"] > 15:
        score_sell += 1

    adx_value = last["ADX"]
    plus_di = last["PLUS_DI"]
    minus_di = last["MINUS_DI"]

    if adx_value > 25 and plus_di > minus_di:
        score_buy += 2

    if adx_value > 25 and minus_di > plus_di:
        score_sell += 2

    if adx_value < 20:
        score_buy -= 1
        score_sell -= 1

    pct_ema200 = round(last["PCT_EMA200"], 2)

    if pct_ema200 < 10:
        stretch = "✅ Saudável"
    elif pct_ema200 < 20:
        stretch = "🟢 Forte"
    elif pct_ema200 < 35:
        stretch = "🟡 Esticado"
    else:
        stretch = "🔴 Muito esticado"

    if pct_ema200 > 35:
        score_buy -= 2

    if adx_value < 20:
        score_buy -= 1

    if adx_value < 20:
        adx_status = "⚪ Lateral"
    elif adx_value < 25:
        adx_status = "🟡 Transição"
    elif adx_value < 40:
        adx_status = "🟢 Tendência Forte"
    else:
        adx_status = "🔥 Tendência Muito Forte"

    return {
        "candles": len(df),
        "first_candle": df["datetime"].iloc[0].strftime("%Y-%m-%d %H:%M UTC"),
        "last_candle": df["datetime"].iloc[-1].strftime("%Y-%m-%d %H:%M UTC"),
        "close": round(last["close"], 6),
        "rsi": round(last["RSI"], 2),
        "macd": round(last["MACD"], 6),
        "signal": round(last["SIGNAL"], 6),
        "ema20": round(last["EMA20"], 6),
        "ema50": round(last["EMA50"], 6),
        "ema200": round(last["EMA200"], 6),
        "atr_pct": round(last["ATR_PCT"], 2),
        "vol_ratio": round(last["VOL_RATIO"], 2),
        "adx": round(adx_value, 2),
        "plus_di": round(plus_di, 2),
        "minus_di": round(minus_di, 2),
        "adx_status": adx_status,
        "pct_ema200": pct_ema200,
        "stretch": stretch,
        "score_buy": score_buy,
        "score_sell": score_sell
    }


def format_tf_block(tf, data):
    return (
        f"\n<b>{tf}</b>\n"
        f"Preço: {data['current_price']}\n"
        f"RSI={data['rsi']} | BUY={data['score_buy']} | SELL={data['score_sell']}\n"
        f"ADX={data['adx']} | {data['adx_status']}\n"
        f"EMA200={data['pct_ema200']}% | {data['stretch']}"
    )


def run_scan():
    print("\n===== PURR SETUP SCANNER =====\n")

    results = {}

    try:
        history_start, df_1d, price_1d, time_1d = discover_1d_start()
        analysis_1d = analyze(df_1d)
        analysis_1d["current_price"] = price_1d
        analysis_1d["current_time"] = time_1d
        results["1d"] = analysis_1d
    except Exception as e:
        results["1d"] = {"erro": str(e)}
        history_start = int(time.time() * 1000) - 1500 * 24 * 60 * 60 * 1000

    for tf in ("4h", "1h"):
        try:
            df, current_price, current_time = get_candles(tf, history_start)
            analysis = analyze(df)
            analysis["current_price"] = current_price
            analysis["current_time"] = current_time
            results[tf] = analysis
        except Exception as e:
            results[tf] = {"erro": str(e)}

    buy_count = 0
    sell_count = 0
    detalhes = []

    for tf in TIMEFRAMES:
        data = results[tf]

        if "erro" in data:
            print(f"{tf} | ERRO: {data['erro']}")
            detalhes.append(f"\n{tf}: ERRO {data['erro']}")
            continue

        print(f"\n{tf}")
        print(
            f"🕯️ CANDLES={data['candles']} "
            f"| INÍCIO={data['first_candle']} "
            f"| FIM={data['last_candle']}"
        )
        print(
            f"💰 PREÇO ATUAL: {data['current_price']} "
            f"| {data['current_time']}"
        )
        print(
            f"RSI={data['rsi']} "
            f"| BUY={data['score_buy']} "
            f"| SELL={data['score_sell']}"
        )
        print(
            f"EMA20={data['ema20']} "
            f"| EMA50={data['ema50']} "
            f"| EMA200={data['ema200']}"
        )
        print(
            f"ATR={data['atr_pct']}% "
            f"| VOL={data['vol_ratio']}"
        )
        print(
            f"ADX={data['adx']} "
            f"| +DI={data['plus_di']} "
            f"| -DI={data['minus_di']} "
            f"| {data['adx_status']}"
        )
        print(
            f"EMA200={data['pct_ema200']}% "
            f"| {data['stretch']}"
        )

        detalhes.append(format_tf_block(tf, data))

        if data["score_buy"] >= 8:
            buy_count += TF_WEIGHT[tf]

        if data["score_sell"] >= 8:
            sell_count += TF_WEIGHT[tf]

    print("\n=================================")

    preco = None
    horario = None
    for tf in TIMEFRAMES:
        if "current_price" in results.get(tf, {}):
            preco = results[tf]["current_price"]
            horario = results[tf]["current_time"]
            break

    if test == 1:
        sinal = "🟢 SINAL DE COMPRA [TEST]"
        print(sinal)
        notify(
            f"<b>PURR SETUP SCANNER</b>\n"
            f"{sinal}\n"
            f"Preço: {preco} | {horario}\n"
            f"Score ponderado BUY={buy_count} | SELL={sell_count}"
            + "".join(detalhes)
        )
    elif buy_count >= 4:
        sinal = "🟢 SINAL DE COMPRA"
        print(sinal)
        notify(
            f"<b>PURR SETUP SCANNER</b>\n"
            f"{sinal}\n"
            f"Preço: {preco} | {horario}\n"
            f"Score ponderado BUY={buy_count} | SELL={sell_count}"
            + "".join(detalhes)
        )
    elif sell_count >= 4:
        sinal = "🔴 SINAL DE VENDA"
        print(sinal)
        notify(
            f"<b>PURR SETUP SCANNER</b>\n"
            f"{sinal}\n"
            f"Preço: {preco} | {horario}\n"
            f"Score ponderado BUY={buy_count} | SELL={sell_count}"
            + "".join(detalhes)
        )
    else:
        print("🟡 AGUARDAR")

    print("=================================\n")


if __name__ == "__main__":
    while True:
        inicio = time.time()
        try:
            run_scan()
        except Exception as e:
            print(f"ERRO na varredura: {e}")

        decorrido = time.time() - inicio
        espera = max(0, INTERVALO_SEGUNDOS - decorrido)
        print(f"Próxima varredura em {int(espera / 60)} min\n")
        time.sleep(espera)
