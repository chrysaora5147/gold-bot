import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import google.generativeai as genai
from supabase import create_client, Client
import json
import re
import os

print("🦿 กำลังคำนวณโมเดลคณิตศาสตร์ระบบไฮบริด V12 และเรียกใช้ Gemini API หลังบ้าน...")

# --- ค่า CONFIG ของบอส ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SUPABASE_URL = "https://vsgbfcatnpytrsshdbkr.supabase.co"
SUPABASE_KEY = "sb_publishable_KpdMpkgsChvu0pR_Gh1y8Q_0LxccpPq"

genai.configure(api_key=GOOGLE_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# 1. FETCH HISTORICAL DATA FOR ROLLING & FEATURE ENGINEERING
print("📥 Fetching historical data...")
assets = {
    'gold': 'GC=F', 'sp500': '^GSPC', 'us_dollar': 'DX-Y.NYB', 
    'crude_oil': 'CL=F', 'vix': '^VIX', 'us_10y': '^TNX'
}

df_list = []
for name, ticker in assets.items():
    data = yf.download(ticker, period="4y", interval="1d", group_by='ticker', progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(0)
    close_series = data['Close'].copy()
    close_series.name = name
    df_list.append(close_series)

df = pd.concat(df_list, axis=1).dropna()
df = df.sort_index()

# 2. GENERATE V11 ADVANCED FEATURES (15 อินดิเคเตอร์มหาภาคก้อนวิจัยของบอส)
df['gold_return'] = df['gold'].pct_change()
df['sp500_return'] = df['sp500'].pct_change()
df['us_dollar_return'] = df['us_dollar'].pct_change()
df['crude_oil_return'] = df['crude_oil'].pct_change()

df['sp500_lag1'] = df['sp500_return'].shift(1)
df['us_dollar_lag1'] = df['us_dollar_return'].shift(1)
df['crude_oil_lag1'] = df['crude_oil_return'].shift(1)
df['vix_level'] = df['vix']
df['us_10y_level'] = df['us_10y']

df['gold_volatility'] = df['gold_return'].rolling(window=20).std()
df['gold_mom5'] = df['gold'].pct_change(5)
df['gold_mom20'] = df['gold'].pct_change(20)
df['dollar_mom5'] = df['us_dollar'].pct_change(5)

delta = df['gold'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['gold_rsi14'] = 100 - (100 / (1 + rs))

df['gold_ema10'] = df['gold'].ewm(span=10, adjust=False).mean()
df['gold_ema50'] = df['gold'].ewm(span=50, adjust=False).mean()
df['ema_crossover'] = (df['gold_ema10'] - df['gold_ema50']) / df['gold_ema50']
df['dist_from_ema50'] = (df['gold'] - df['gold_ema50']) / df['gold_ema50']

features = [
    'sp500_return', 'us_dollar_return', 'crude_oil_return', 
    'sp500_lag1', 'us_dollar_lag1', 'crude_oil_lag1',
    'vix_level', 'us_10y_level', 'gold_volatility',
    'gold_mom5', 'gold_mom20', 'dollar_mom5', 'gold_rsi14',
    'ema_crossover', 'dist_from_ema50'
]

# Keep the live feature set separate from the supervised training set.
# The last rows do not have a known 5-day future price yet, so dropping target
# NaNs before selecting X_today would accidentally move "today" into the past.
feature_frame = df.dropna(subset=features).copy()
df['future_gold'] = df['gold'].shift(-5)
df['target'] = (df['future_gold'] > df['gold']).astype(int)
model_frame = df.dropna(subset=features + ['future_gold', 'target']).copy()


def threshold_for_regime(ema_value):
    return 0.550 if ema_value > 0 else 0.600


def signal_from_probability(probability, ema_value):
    return "UP" if probability >= threshold_for_regime(ema_value) else "DOWN"


def calculate_target_position(probability, threshold, ema_value, volatility):
    """Volatility-aware sizing for research signals.

    This keeps gold as a strategic allocation but avoids jumping straight to
    100% exposure from a single classifier output.
    """
    base_exp = 0.25
    max_exp = 0.85

    if probability < threshold:
        return base_exp

    confidence_edge = min(max((probability - threshold) / max(1 - threshold, 1e-9), 0), 1)
    trend_bonus = 0.10 if ema_value > 0 else 0.0

    # GC=F daily volatility is usually around 0.8%-1.6%; scale down when noisy.
    vol_penalty = 1.0
    if pd.notna(volatility) and volatility > 0:
        vol_penalty = min(1.0, 0.012 / float(volatility))

    position = base_exp + (max_exp - base_exp) * confidence_edge * vol_penalty + trend_bonus
    return float(min(max(position, base_exp), max_exp))


def run_walk_forward_backtest(history, feature_cols, lookback=252 * 3, test_days=252):
    """Simple expanding/rolling research audit for the current feature recipe."""
    if len(history) <= lookback + 20:
        return None

    start = max(lookback, len(history) - test_days)
    rows = []

    for idx in range(start, len(history)):
        train = history.iloc[idx - lookback:idx]
        test = history.iloc[[idx]]

        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=3,
            min_samples_leaf=5,
            random_state=42,
        )
        clf.fit(train[feature_cols], train['target'])

        proba = float(clf.predict_proba(test[feature_cols])[0][1])
        ema_value = float(test['ema_crossover'].iloc[0])
        pred = signal_from_probability(proba, ema_value)
        actual = "UP" if int(test['target'].iloc[0]) == 1 else "DOWN"
        fwd_return = float(test['future_gold'].iloc[0] / test['gold'].iloc[0] - 1)

        rows.append({
            "pred": pred,
            "actual": actual,
            "proba": proba,
            "fwd_return": fwd_return,
            "strategy_return": fwd_return if pred == "UP" else 0.0,
        })

    audit = pd.DataFrame(rows)
    wins = int((audit['pred'] == audit['actual']).sum())
    total = int(len(audit))
    avg_strategy_return = float(audit['strategy_return'].mean())
    avg_buy_hold_return = float(audit['fwd_return'].mean())

    return {
        "total": total,
        "win_rate": wins / total if total else 0,
        "avg_strategy_return": avg_strategy_return,
        "avg_buy_hold_return": avg_buy_hold_return,
        "up_share": float((audit['pred'] == "UP").mean()) if total else 0,
    }

# ==========================================
# 🔍 SECTION 2.5: ระบบกรรมการตรวจข้อสอบออโต้ (Rolling Window Grader)
# ==========================================
print("⚖️ กำลังตรวจสอบประวัติคำทำนายย้อนหลัง 5 วันทำการ...")
try:
    # ดึงเฉพาะประวัติที่ยังไม่ได้ตรวจข้อสอบ (PENDING)
    pending_records = supabase.table('gold_predictions').select('*').is_('quant_result', 'null').execute()
    
    if pending_records.data:
        for record in pending_records.data:
            record_id = record['id']
            run_date = pd.to_datetime(record['created_at']).tz_localize(None).normalize()
            
            try:
                # หาตำแหน่ง Index ของวันที่รันบอทในตารางข้อมูลหุ้น
                start_idx = feature_frame.index.get_indexer([run_date], method='nearest')[0]
                current_idx = len(feature_frame) - 1
                
                # เช็กเงื่อนไข: ถ้าเวลาผ่านไปครบ 5 วันทำการ ให้เริ่มตัดเกรด!
                if (current_idx - start_idx) >= 5:
                    start_price = float(feature_frame['gold'].iloc[start_idx])
                    end_price = float(feature_frame['gold'].iloc[start_idx + 5])
                    
                    actual_direction = "UP" if end_price > start_price else "DOWN"
                    
                    quant_pred = record.get('model_direction', '')
                    quant_result = "WIN" if quant_pred == actual_direction else "LOSS"
                    
                    ai_pred = record.get('ai_direction', '')
                    ai_result = "WIN" if ai_pred == actual_direction else "LOSS"
                    
                    # ประทับตราเกรด และอัปเดตกลับลง Supabase
                    supabase.table('gold_predictions').update({
                        'quant_result': quant_result,
                        'ai_result': ai_result,
                        'actual_start_price': start_price,
                        'actual_end_price': end_price
                    }).eq('id', record_id).execute()
                    
                    print(f"✅ ตรวจข้อสอบ ID {record_id} เสร็จสิ้น! [Quant: {quant_result} | AI: {ai_result}]")
            except Exception as inner_e:
                print(f"⚠️ ข้ามการตรวจข้อสอบ ID {record_id} เนื่องจากยังไม่มีข้อมูลอ้างอิง: {inner_e}")
    else:
        print("⏳ ยังไม่มีข้อสอบที่ครบกำหนด 5 วันทำการให้ตรวจในวันนี้")
except Exception as e:
    print(f"⚠️ ระบบตรวจข้อสอบมีปัญหา: {e}")
# ==========================================

# 3. DAILY ROLLING Retrain & Predict (ดึงข้อมูล 3 ปีย้อนหลังเพื่อพยากรณ์วันนี้)
backtest_summary = run_walk_forward_backtest(model_frame, features)
if backtest_summary:
    print(
        "📊 Walk-forward audit "
        f"n={backtest_summary['total']} | "
        f"win_rate={backtest_summary['win_rate']:.1%} | "
        f"avg_signal_5d={backtest_summary['avg_strategy_return']:.3%} | "
        f"avg_buy_hold_5d={backtest_summary['avg_buy_hold_return']:.3%} | "
        f"up_share={backtest_summary['up_share']:.1%}"
    )

train_window = model_frame.tail(252 * 3)
X_train = train_window[features]
y_train = train_window['target']
X_today = feature_frame[features].iloc[[-1]]

model = RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_leaf=5, random_state=42)
model.fit(X_train, y_train)

raw_proba = float(model.predict_proba(X_today)[0][1])
latest_feature_row = feature_frame.iloc[-1]
ema_crossover = float(latest_feature_row['ema_crossover'])
gold_volatility = float(latest_feature_row['gold_volatility'])
signal_date = feature_frame.index[-1]
signal_price = float(latest_feature_row['gold'])

current_thresh = threshold_for_regime(ema_crossover)
model_direction = signal_from_probability(raw_proba, ema_crossover)

# ตรรกะคุมพอร์ต Allocation แบบลดความเสี่ยง ไม่โดด 100% จากสัญญาณเดียว
target_position = calculate_target_position(raw_proba, current_thresh, ema_crossover, gold_volatility)
print(
    f"🎯 Live signal date={signal_date.date()} | "
    f"GC=F={signal_price:.2f} | proba={raw_proba:.3f} | "
    f"threshold={current_thresh:.3f} | direction={model_direction} | "
    f"target_position={target_position:.1%}"
)

# 4. CALL GEMINI AI GENERATION FOR MACRO INSIGHTS
print("🧠 Triggering Gemini Intelligence analysis...")
gemini_model = genai.GenerativeModel('gemini-2.5-flash')

latest_data = latest_feature_row
current_dxy = float(latest_data['us_dollar'])
current_bond = float(latest_data['us_10y_level'])
current_sp500 = float(latest_data['sp500'])
current_vix = float(latest_data['vix_level'])

prompt = f"""
คุณเป็นนักวิเคราะห์ราคาทองคำระดับโลกและผู้เชี่ยวชาญด้านเศรษฐกิจมหภาค (Macro Strategist)
นี่คือข้อมูลตัวเลขดัชนีและปัจจัยตลาดทุนปัจจุบัน:
- ดัชนีเงินดอลลาร์สหรัฐ (DXY): {current_dxy:.2f}
- อัตราผลตอบแทนพันธบัตรรัฐบาลสหรัฐ 10 ปี (Bond Yield): {current_bond:.2f}%
- ดัชนีหุ้น S&P 500: {current_sp500:.2f}
- ดัชนีความกลัวตลาด (VIX Index): {current_vix:.2f}

หน้าที่ของคุณ:
1. จงประมวลผลตัวเลขดัชนีเหล่านี้ ร่วมกับสถานการณ์ข่าวสารเศรษฐกิจ ภูมิรัฐศาสตร์ (Geopolitics) และความรู้สึกเสี่ยง (Market Sentiment) ล่าสุดของโลกในนาทีนี้
2. วิเคราะห์และพยากรณ์แนวโน้มสะสมว่า "ภายในอีก 5 วันทำการข้างหน้า" ปัจจัยสายข่าวและกลไกเงินทุนเหล่านี้จะขับเคลื่อนให้ราคาทองคำปิด "สูงขึ้น (UP)" หรือ "ลดลง (DOWN)" เมื่อเทียบกับวันนี้
3. สรุปอินไซต์สั้นกระชับเป็นภาษาไทย ความยาวไม่เกิน 4 บรรทัด เขียนแยกเป็นข้อๆ 1, 2, 3 เน้นที่กลไกการไหลของเงินทุนและข่าวสารโลก ห้ามพ่นศัพท์เทคนิคของโมเดล Quant ซ้ำซากเด็ดขาด

4. บรรทัดสุดท้ายสุด ให้พิมพ์ผลสรุปทิศทาง (UP หรือ DOWN เพียวๆ เท่านั้น ห้ามใส่คำอธิบายภาษาไทยใน JSON บล็อก) และระดับความมั่นใจของคุณเป็นตัวเลขเปอร์เซ็นต์ (0-100) ให้อยู่ในรูปแบบ JSON บรรทัดเดียวปิดท้ายแบบนี้เท่านั้น ห้ามมีตัวอักษรอื่นปนในบรรทัดนั้นเด็ดขาด:
{{"direction": "UP หรือ DOWN", "confidence": ตัวเลขเปอร์เซ็นต์}}
"""

response = gemini_model.generate_content(prompt)
full_text = response.text

# 🎯 ระบบวิเคราะห์สกัดโครงสร้าง JSON สายข่าวอัจฉริยะ
ai_direction_extracted = "UP" if raw_proba >= 0.50 else "DOWN"
ai_confidence_extracted = 0.50

try:
    lines = [l.strip() for l in full_text.strip().split('\n') if l.strip()]
    
    json_data = None
    for line in reversed(lines):
        match = re.search(r'\{.*\}', line)
        if match:
            json_data = json.loads(match.group(0))
            break
            
    if json_data:
        raw_dir = json_data.get("direction", ai_direction_extracted).upper()
        if "UP" in raw_dir or "ขึ้น" in raw_dir:
            ai_direction_extracted = "UP"
        elif "DOWN" in raw_dir or "ลง" in raw_dir:
            ai_direction_extracted = "DOWN"
            
        raw_conf = float(json_data.get("confidence", 50))
        if raw_conf > 1.0:
            raw_conf = raw_conf / 100.0
        ai_confidence_extracted = raw_conf
    
    cleaned_lines = [l for l in lines if not l.startswith('{') and not l.endswith('}') and not l.startswith('`')]
    ai_reason_text = '\n'.join(cleaned_lines)
except Exception as e:
    print(f"⚠️ Parsing JSON error: {e}")
    ai_reason_text = full_text

if backtest_summary:
    audit_line = (
        "\n\n[Quant audit] "
        f"Live signal date: {signal_date.date()} | "
        f"5D walk-forward win rate: {backtest_summary['win_rate']:.1%} "
        f"over {backtest_summary['total']} samples | "
        f"avg signal 5D return: {backtest_summary['avg_strategy_return']:.3%} | "
        f"target position: {target_position:.1%}"
    )
else:
    audit_line = (
        "\n\n[Quant audit] "
        f"Live signal date: {signal_date.date()} | "
        f"not enough history for walk-forward audit | "
        f"target position: {target_position:.1%}"
    )
ai_reason_text = ai_reason_text + audit_line

# 5. PACK DATA HYBRID PAYLOAD & INJECT TO SUPABASE
payload = {
    "model_direction": model_direction,
    "ai_direction": ai_direction_extracted,
    "model_confidence": float(raw_proba),
    "ai_confidence": float(ai_confidence_extracted),
    "target_position": float(target_position),
    "ema_crossover": float(ema_crossover),
    "ai_reason": ai_reason_text
}

print("🚀 Launching V12 Live Data payload to Supabase right now...")
supabase.table('gold_predictions').insert(payload).execute()

print("*" * 40)
print("🎉 [สำเร็จ!] สมองกลเวอร์ชัน V12 ยิงค่าความมั่นใจแท้จริงของ Gemini เข้าหน้าเว็บเรียบร้อยแล้ว!")
print("*" * 40)
