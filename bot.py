import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import google.generativeai as genai
from supabase import create_client, Client

print("🦿 กำลังคำนวณโมเดลคณิตศาสตร์ระบบไฮบริด V11 และเรียกใช้ Gemini API หลังบ้าน...")

# --- ค่า CONFIG ของบอส ---
import os
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

df['future_gold'] = df['gold'].shift(-5)
df['target'] = (df['future_gold'] > df['gold']).astype(int)
df = df.dropna()

features = [
    'sp500_return', 'us_dollar_return', 'crude_oil_return', 
    'sp500_lag1', 'us_dollar_lag1', 'crude_oil_lag1',
    'vix_level', 'us_10y_level', 'gold_volatility',
    'gold_mom5', 'gold_mom20', 'dollar_mom5', 'gold_rsi14',
    'ema_crossover', 'dist_from_ema50'
]

# 3. DAILY ROLLING Retrain & Predict (ดึงข้อมูล 3 ปีย้อนหลังเพื่อพยากรณ์วันนี้)
train_window = df.tail(252 * 3)
X_train = train_window[features].iloc[:-1]
y_train = train_window['target'].iloc[:-1]
X_today = train_window[features].iloc[[-1]]

model = RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_leaf=5, random_state=42)
model.fit(X_train, y_train)

raw_proba = float(model.predict_proba(X_today)[0][1])
ema_crossover = float(train_window['ema_crossover'].iloc[-1])

# Fixed Robust Parameters จากโมเดล V11
base_exp = 0.25
uptrend_thresh = 0.550
downtrend_thresh = 0.600

current_thresh = uptrend_thresh if ema_crossover > 0 else downtrend_thresh
model_direction = "UP" if raw_proba >= current_thresh else "DOWN"

# ตรรกะคุมพอร์ต Allocation ล็อคน้ำหนัก V11
target_position = 1.0 if raw_proba >= current_thresh else base_exp

# 4. CALL GEMINI AI GENERATION FOR MACRO INSIGHTS
print("🧠 Triggering Gemini Intelligence analysis...")
gemini_model = genai.GenerativeModel('gemini-2.0-flash')

# ดึงค่าตัวเลขจริงหน้างานของวันนี้จาก DataFrame มารอส่งให้ Gemini ชำแหละข่าวดนสดๆ
latest_data = train_window.iloc[-1]
current_dxy = float(latest_data['us_dollar'])
current_bond = float(latest_data['us_10y_level'])
current_sp500 = float(latest_data['sp500'])
current_vix = float(latest_data['vix_level'])

prompt = f"""
คุณเป็นนักวิเคราะห์ราคาทองคำระดับโลกในกองทุน Quant 
นี่คือข้อมูลตัวเลขดัชนีเศรษฐกิจมหภาคและตลาดโลกปัจจุบันของนาทีนี้:
- ดัชนีเงินดอลลาร์สหรัฐ (DXY): {current_dxy:.2f}
- อัตราผลตอบแทนพันธบัตรรัฐบาลสหรัฐ 10 ปี (Bond Yield): {current_bond:.2f}%
- ดัชนีหุ้น S&P 500: {current_sp500:.2f}
- ดัชนีความกลัวตลาด (VIX Index): {current_vix:.2f}

จงสรุปบทวิเคราะห์แนวโน้มราคาทองคำสั้นกระชับเป็นภาษาไทย ความยาวไม่เกิน 4 บรรทัด เขียนแยกเป็นข้อๆ 1, 2, 3 
โดยวิเคราะห์เชื่อมโยงว่าตัวเลขเศรษฐกิจและปัจจัยเชิงมหภาค/ภูมิรัฐศาสตร์เหล่านี้ ส่งผลเชิงบวกหรือเชิงลบต่อทองคำอย่างไรบ้าง 
เน้นเนื้อหาไปที่ปัจจัยสายข่าวและกลไกตลาดทุน ห้ามนำเปอร์เซ็นต์ความมั่นใจหรือตรรกะภายในของโมเดล Quant มาอธิบายซ้ำซากเด็ดขาด
"""

response = gemini_model.generate_content(prompt)
ai_reason_text = response.text

# 5. PACK DATA HYBRID PAYLOAD & INJECT TO SUPABASE
payload = {
    "model_direction": model_direction,
    "ai_direction": "UP" if raw_proba >= 0.50 else "DOWN",
    "model_confidence": float(raw_proba),
    "ai_confidence": 0.80, 
    "target_position": float(target_position),
    "ema_crossover": float(ema_crossover),
    "ai_reason": ai_reason_text
}

print("🚀 Launching V11 Live Data payload to Supabase right now...")
supabase.table('gold_predictions').insert(payload).execute()

print("*" * 40)
print("🎉 [สำเร็จ!] สมองกลเวอร์ชัน V11 คำนวณและยิงเข้าหน้าเว็บของบอสเรียบร้อยแล้ว!")
print("*" * 40)
