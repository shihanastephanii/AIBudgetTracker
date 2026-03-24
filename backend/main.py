from __future__ import annotations

import os
import re
import json
import base64
import secrets
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from passlib.context import CryptContext
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, EmailStr
import jwt
import psycopg2
from psycopg2.extras import RealDictCursor
from openai import OpenAI
import anthropic

import cv2
import numpy as np
from PIL import Image
import io
import base64

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_hex(32))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7

app = FastAPI(title="AI Budget Tracker API", version="2.0")

openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()


def get_db():
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    try:
        yield conn
    finally:
        conn.close()

def db_execute(query: str, params: tuple = None, fetch: str = None):
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    try:
        with conn.cursor() as cur:
            cur.execute(query, params or ())
            if fetch == "one":
                result = cur.fetchone()
            elif fetch == "all":
                result = cur.fetchall()
            else:
                result = None
            conn.commit()
            return result
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


class UserRegister(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6)
    full_name: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    user: Dict[str, Any]

class UserOut(BaseModel):
    id: int
    email: str
    full_name: str
    currency: str
    created_at: str

class TxCreate(BaseModel):
    type: str = Field(..., pattern="^(income|expense)$")
    amount: float = Field(..., gt=0)
    category: str
    note: str = ""
    date: str
    source: str = "manual"

class TxOut(BaseModel):
    id: int
    user_id: int
    type: str
    amount: float
    category: str
    note: str
    date: str
    source: str
    created_at: str

class TxList(BaseModel):
    items: List[TxOut]

class SummaryOut(BaseModel):
    income: float
    expense: float
    net: float
    by_category: Dict[str, float]
    budget_alerts: List[Dict[str, Any]]
    tip: str = ""

class AnalyticsOut(BaseModel):
    monthly_trend: List[Dict[str, Any]]
    category_breakdown: List[Dict[str, Any]]
    top_expenses: List[Dict[str, Any]]
    savings_rate: float
    avg_daily_spend: float

class BudgetGoal(BaseModel):
    category: str
    monthly_limit: float

class ChatIn(BaseModel):
    message: str

class ChatOut(BaseModel):
    reply: str

class ReceiptIngest(BaseModel):
    mode: str = Field(..., pattern="^(single|multiple|combination)$")
    ocr_text: str
    image_base64: Optional[str] = None
    image_media_type: Optional[str] = None

class ReceiptResult(BaseModel):
    saved: int
    extracted: List[Dict[str, Any]]

class VoiceParseIn(BaseModel):
    text: str

class VoiceParseOut(BaseModel):
    type: str
    amount: float
    category: str
    note: str


def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def create_token(user_id: int, email: str) -> str:
    payload = {
        "user_id": user_id,
        "email": email,
        "exp": datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return {"user_id": payload["user_id"], "email": payload["email"]}
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


CATEGORY_RULES = [
    ("Food", ["grocery", "grocer", "restaurant", "cafe", "coffee", "pizza", "burger", "mcd", "kfc", "food", "superstore", "walmart", "costco", "bakery", "deli", "sushi", "noodle", "eat", "drink", "juice", "tea"]),
    ("Transport", ["uber", "lyft", "bus", "taxi", "fuel", "petrol", "gas", "shell", "esso", "chevron", "transit", "parking", "toll", "train", "metro", "grab", "ola"]),
    ("Shopping", ["amazon", "shopping", "mall", "clothes", "fashion", "ikea", "store", "purchase", "retail", "apparel", "shoe", "bag", "watch"]),
    ("Utilities", ["hydro", "electric", "water", "internet", "wifi", "telus", "bell", "rogers", "utility", "phone bill", "electricity", "broadband", "mobile"]),
    ("Entertainment", ["movie", "cinema", "netflix", "spotify", "game", "ticket", "concert", "entertain", "streaming", "theatre", "bowling", "arcade"]),
    ("Health", ["pharmacy", "drug", "clinic", "doctor", "dental", "medicine", "health", "hospital", "medical", "chemist", "lab", "test"]),
    ("Rent", ["rent", "landlord", "lease", "mortgage", "housing"]),
    ("Salary", ["salary", "payroll", "paycheque", "paycheck", "wage"]),
]

def categorize_from_text(text: str) -> str:
    t = (text or "").lower()
    for cat, keys in CATEGORY_RULES:
        if any(k in t for k in keys):
            return cat
    return "Other"

AMOUNT_RE = re.compile(r"(?<!\d)(\d{1,5}(?:[.,]\d{2})?)(?!\d)")

def extract_amounts(text: str) -> List[float]:
    vals: List[float] = []
    for m in AMOUNT_RE.findall(text.replace(",", ".")):
        try:
            v = float(m)
            if v > 0:
                vals.append(v)
        except ValueError:
            continue
    vals = [v for v in vals if 0.5 <= v <= 10000]
    return vals

def is_likely_id_or_card(value: float) -> bool:
    """Heuristic: reject values that look like card numbers, phone numbers, or IDs."""
    s = str(int(value))
    if len(s) >= 8:
        return True
    if value > 9999:
        return True
    return False

RECEIPT_VISION_PROMPT = """You are an expert receipt parser with high accuracy. Analyze this receipt image carefully.

EXTRACTION RULES:
1. Find the GRAND TOTAL or AMOUNT DUE — labeled as "Total", "Grand Total", "Amount Due", "Balance", "Sub-total" + tax combined.
2. Find the STORE/MERCHANT NAME — usually the largest text at the top.
3. Find the DATE — usually near top or bottom, convert to YYYY-MM-DD.
4. Identify INDIVIDUAL LINE ITEMS only if clearly itemized (product name + price on same line).
5. NEVER extract: card numbers (16 digits), terminal IDs, phone numbers, barcodes, loyalty point counts, transaction reference numbers.
6. Amounts must be realistic: between $0.50 and $2000 for a single receipt.
7. If an amount has 4+ digits before the decimal (e.g., 4532.00), it is almost certainly a card number — SKIP IT.
8. Tax line (GST/HST/VAT/Tax) should NOT be listed as a separate transaction — include it in the total only.

CONFIDENCE RULES:
- "high": Total clearly labeled, merchant name clear, amount between $1–$500
- "medium": Total found but merchant unclear, or amount between $500–$1000  
- "low": Amount inferred, no clear total label, or multiple possible totals

RESPOND ONLY with a valid JSON array. No markdown, no explanation. Example:
[
  {
    "type": "expense",
    "amount": 24.75,
    "category": "Food",
    "note": "McDonald's - Meal",
    "date": "2025-01-15",
    "source": "receipt",
    "merchant": "McDonald's",
    "confidence": "high"
  }
]

For "single" mode: return ONE item with the grand total.
For "multiple" mode: return each distinct line item separately.
For "combination" mode: return the grand total PLUS up to 3 major line items."""

RECEIPT_TEXT_PROMPT = """You are an expert receipt parser. Analyze this OCR-extracted receipt text carefully.

OCR text may have errors: letters confused for numbers (O vs 0, l vs 1, S vs 5), missing spaces, jumbled characters. Read it carefully.

EXTRACTION RULES:
1. Find the GRAND TOTAL — look for lines containing "Total", "Amount", "Balance", "Due", "Pay" followed by a number.
2. Find the STORE NAME — usually the first non-numeric line at the top.
3. Find the DATE — look for date patterns like DD/MM/YY, MM-DD-YYYY, etc.
4. NEVER extract: 16-digit card numbers, 10-digit phone numbers, 8+ digit reference IDs.
5. Amounts must be between $0.50 and $2000.
6. If you see something like "4532 1234 5678 9012" or "4532.12345678" — that is a CARD NUMBER, skip it.
7. The total is usually the LARGEST amount on the receipt or the LAST amount listed.

RESPOND ONLY with a valid JSON array, no markdown or explanation."""

def parse_receipt_with_vision(image_base64: str, media_type: str, mode: str) -> List[Dict[str, Any]]:
    """Use Claude Vision or GPT-4V to directly read the receipt image — far more accurate than OCR."""
    today = datetime.now().strftime("%Y-%m-%d")
    # preprocess phone image
    image_base64 = preprocess_receipt_image(image_base64)
    # Try Anthropic Claude Vision first (best accuracy)
    if anthropic_client:
        try:
            message = anthropic_client.messages.create(
                model="claude-opus-4-5",
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_base64,
                                },
                            },
                            {
                                "type": "text",
                                "text": f"Mode: {mode}\n\n{RECEIPT_VISION_PROMPT}"
                            }
                        ],
                    }
                ],
            )
            raw = message.content[0].text.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            entries = json.loads(raw)
            if isinstance(entries, dict):
                entries = [entries]
            return _validate_and_format_entries(entries, today)
        except Exception as e:
            print(f"Claude vision parsing failed: {e}")

    # Fallback: OpenAI GPT-4V
    if openai_client:
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{image_base64}",
                                    "detail": "high"
                                }
                            },
                            {
                                "type": "text",
                                "text": f"Mode: {mode}\n\n{RECEIPT_VISION_PROMPT}"
                            }
                        ],
                    }
                ],
                temperature=0.05,
            )
            raw = response.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            entries = json.loads(raw)
            if isinstance(entries, dict):
                entries = [entries]
            return _validate_and_format_entries(entries, today)
        except Exception as e:
            print(f"GPT-4V vision parsing failed: {e}")

    return []

def parse_receipt_with_ocr_text(ocr_text: str, mode: str) -> List[Dict[str, Any]]:
    """Parse receipt from OCR text using AI with improved prompting."""
    text = (ocr_text or "").strip()
    if not text or len(text) < 10:
        return []

    today = datetime.now().strftime("%Y-%m-%d")

    # Try OpenAI with enhanced prompt
    if openai_client:
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": RECEIPT_TEXT_PROMPT
                    },
                    {
                        "role": "user",
                        "content": f"Mode: {mode}\n\nReceipt OCR Text:\n{text}"
                    }
                ],
                temperature=0.05,
                max_tokens=800
            )
            raw = response.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            entries = json.loads(raw)
            if isinstance(entries, dict):
                entries = [entries]
            return _validate_and_format_entries(entries, today)
        except Exception as e:
            print(f"OpenAI OCR text parsing failed: {e}")

    # Try Anthropic with OCR text
    if anthropic_client:
        try:
            message = anthropic_client.messages.create(
                model="claude-opus-4-5",
                max_tokens=800,
                messages=[
                    {
                        "role": "user",
                        "content": f"{RECEIPT_TEXT_PROMPT}\n\nMode: {mode}\n\nReceipt OCR Text:\n{text}"
                    }
                ],
            )
            raw = message.content[0].text.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            entries = json.loads(raw)
            if isinstance(entries, dict):
                entries = [entries]
            return _validate_and_format_entries(entries, today)
        except Exception as e:
            print(f"Anthropic OCR text parsing failed: {e}")

    # Final fallback: regex-based
    return _regex_fallback_parse(text, mode, today)

def _validate_and_format_entries(entries: List[Any], today: str) -> List[Dict[str, Any]]:
    """Strict validation and formatting of extracted receipt entries."""
    formatted = []
    seen_amounts = set()

    for e in entries:
        try:
            raw_amount = str(e.get("amount", "0"))

            # extract number safely
            match = re.search(r"\d+(?:[.,]\d{1,2})?", raw_amount)
            if not match:
                continue

            amount = float(match.group(0).replace(",", "."))

            if amount < 0.50:
                print(f"Rejected: amount too low ({amount})")
                continue
            if amount > 2000:
                print(f"Rejected: amount too high ({amount})")
                continue
            if is_likely_id_or_card(amount):
                print(f"Rejected: looks like ID/card number ({amount})")
                continue

            # Deduplicate near-identical amounts
            rounded = round(amount, 1)
            if rounded in seen_amounts:
                continue
            seen_amounts.add(rounded)

            # Validate and clean date
            raw_date = e.get("date", today)
            try:
                datetime.strptime(str(raw_date), "%Y-%m-%d")
                clean_date = str(raw_date)
            except ValueError:
                clean_date = today

            # Reject future dates
            if clean_date > today:
                clean_date = today

            category = e.get("category", "Other")
            if category not in ["Food", "Transport", "Shopping", "Utilities", "Entertainment", "Health", "Rent", "Other"]:
                category = "Other"

            merchant = str(e.get("merchant", "Unknown Store"))[:80]
            note = str(e.get("note", merchant))[:120]
            confidence = e.get("confidence", "medium")
            if confidence not in ["high", "medium", "low"]:
                confidence = "medium"

            formatted.append({
                "type": "expense",
                "amount": round(amount, 2),
                "category": category,
                "note": note,
                "date": clean_date,
                "source": "receipt",
                "merchant": merchant,
                "confidence": confidence
            })
        except Exception as ex:
            print(f"Entry validation error: {ex} — entry: {e}")
            continue

    return formatted

def _regex_fallback_parse(text: str, mode: str, today: str) -> List[Dict[str, Any]]:
    """Last-resort regex parsing when all AI methods fail."""
    amounts = [a for a in extract_amounts(text) if not is_likely_id_or_card(a) and a <= 500]
    if not amounts:
        return []

    lines = text.split("\n")
    merchant = "Unknown Store"
    for line in lines[:6]:
        line = line.strip()
        if len(line) > 3 and not re.search(r"\d{3,}", line):
            merchant = line[:60]
            break

    cat = categorize_from_text(text + " " + merchant)
    amounts_sorted = sorted(amounts)

    if mode == "single":
        return [{
            "type": "expense",
            "amount": round(float(amounts_sorted[-1]), 2),
            "category": cat,
            "note": merchant,
            "date": today,
            "source": "receipt",
            "merchant": merchant,
            "confidence": "low"
        }]

    picked = []
    for v in sorted(amounts, reverse=True):
        if len(picked) >= 5:
            break
        if any(abs(v - p) < 0.10 for p in picked):
            continue
        picked.append(v)

    return [
        {
            "type": "expense",
            "amount": round(float(v), 2),
            "category": cat,
            "note": f"{merchant} - Item {i}",
            "date": today,
            "source": "receipt",
            "merchant": merchant,
            "confidence": "low"
        }
        for i, v in enumerate(picked, start=1)
    ]

def guess_receipt_entries(ocr_text: str, mode: str, image_base64: str = None, image_media_type: str = None) -> List[Dict[str, Any]]:
    """Main entry point: prefer vision parsing over OCR text parsing."""
    if image_base64 and image_media_type:
        entries = parse_receipt_with_vision(image_base64, image_media_type, mode)
        if entries:
            return entries
        print("Vision parsing returned nothing, falling back to OCR text")

    if ocr_text and len(ocr_text.strip()) > 20:
        return parse_receipt_with_ocr_text(ocr_text, mode)

    return []


def parse_voice_text(text: str) -> VoiceParseOut:
    t = (text or "").strip()
    if not t:
        raise HTTPException(status_code=400, detail="Empty voice text.")

    if openai_client:
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a voice transaction parser. Extract ONE transaction from spoken text.

Return ONLY valid JSON:
{
  "type": "income" OR "expense",
  "amount": <number>,
  "category": "Food/Transport/Shopping/Utilities/Entertainment/Health/Rent/Salary/Other",
  "note": "<brief description>"
}

Rules:
- "income" keywords: salary, pay, income, wage, paycheck, earned, got paid
- Amount must be a valid positive number
- Note should be short and descriptive"""
                    },
                    {
                        "role": "user",
                        "content": f"Parse this voice input: {t}"
                    }
                ],
                temperature=0.1,
                max_tokens=150
            )
            result = response.choices[0].message.content.strip()
            result = result.replace("```json", "").replace("```", "").strip()
            data = json.loads(result)
            amount = float(data.get("amount", 0))
            if amount <= 0:
                raise ValueError("Invalid amount")
            tx_type = data.get("type", "expense")
            if tx_type not in ["income", "expense"]:
                tx_type = "expense"
            return VoiceParseOut(
                type=tx_type,
                amount=round(amount, 2),
                category=data.get("category", "Other"),
                note=data.get("note", t)[:100]
            )
        except Exception as e:
            print(f"OpenAI voice parsing failed: {e}")

    m = re.search(r"(\d{1,5}(?:[.,]\d{1,2})?)", t.replace(",", "."))
    if not m:
        raise HTTPException(status_code=400, detail="No amount found in voice text. Please say the amount clearly.")
    amount = float(m.group(1))
    lower = t.lower()
    tx_type = "income" if any(k in lower for k in ["salary", "pay", "income", "wage", "paycheck", "earned", "got paid"]) else "expense"
    category = categorize_from_text(lower)
    return VoiceParseOut(type=tx_type, amount=round(amount, 2), category=category, note=t[:100])


async def get_ai_response(message: str, user_context: Dict[str, Any]) -> str:
    if not openai_client:
        return get_simple_response(message, user_context)

    try:
        total_income = user_context.get("income", 0)
        total_expense = user_context.get("expense", 0)
        by_cat = user_context.get("by_category", {})

        context_str = f"""User Financial Summary:
- Income: ${total_income:.2f}
- Expenses: ${total_expense:.2f}
- Net Balance: ${total_income - total_expense:.2f}
- Savings Rate: {((total_income - total_expense) / total_income * 100) if total_income > 0 else 0:.1f}%

Spending by Category:
"""
        for cat, amt in sorted(by_cat.items(), key=lambda x: x[1], reverse=True):
            pct = (amt / total_expense * 100) if total_expense > 0 else 0
            context_str += f"- {cat}: ${amt:.2f} ({pct:.1f}%)\n"

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """You are a helpful personal finance assistant. Guidelines:
- Be friendly, encouraging, and concise
- Use emojis sparingly (1-2 max)
- Give specific, actionable advice referencing actual data
- Keep responses under 150 words
- Suggest concrete next steps"""
                },
                {
                    "role": "user",
                    "content": f"{context_str}\n\nUser question: {message}"
                }
            ],
            temperature=0.7,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI chat failed: {e}")
        return get_simple_response(message, user_context)

def get_simple_response(message: str, user_context: Dict[str, Any]) -> str:
    msg = message.lower()
    total_income = user_context.get("income", 0)
    total_expense = user_context.get("expense", 0)
    by_cat = user_context.get("by_category", {})

    if "how much" in msg and "spent" in msg:
        if "food" in msg:
            return f"You've spent ${by_cat.get('Food', 0):.2f} on Food this month."
        elif "transport" in msg:
            return f"You've spent ${by_cat.get('Transport', 0):.2f} on Transport this month."
        return f"Your total expenses are ${total_expense:.2f} this month."

    if "save" in msg or "saving" in msg:
        net = total_income - total_expense
        if total_income > 0:
            return f"You're saving ${net:.2f} ({(net / total_income * 100):.1f}% of income)."
        return "Add your income to see savings rate."

    if "tip" in msg or "advice" in msg or "help" in msg:
        if not by_cat:
            return "Start tracking expenses to get personalized tips!"
        top_cat = max(by_cat.items(), key=lambda x: x[1])[0]
        return f"Your highest spending is {top_cat} (${by_cat[top_cat]:.2f}). Try setting a monthly budget for it."

    return "I can help with: spending analysis, savings tips, budget advice. Try asking 'How much did I spend?' or 'Give me tips'."

def preprocess_receipt_image(image_base64: str):
    """Improve phone-captured receipt images"""
    
    img_bytes = base64.b64decode(image_base64)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = np.array(img)

    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # increase contrast
    gray = cv2.equalizeHist(gray)

    # sharpen
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    sharp = cv2.filter2D(gray, -1, kernel)

    # adaptive threshold (best for receipts)
    processed = cv2.adaptiveThreshold(
        sharp,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    # resize (VERY IMPORTANT for phone images)
    h, w = processed.shape
    if w > 1600:
        scale = 1600 / w
        processed = cv2.resize(processed, None, fx=scale, fy=scale)

    # convert back to base64
    _, buffer = cv2.imencode(".jpg", processed)
    new_base64 = base64.b64encode(buffer).decode()

    return new_base64


@app.post("/auth/register", response_model=Token)
async def register(user: UserRegister):
    existing = db_execute("SELECT id FROM users WHERE email = %s", (user.email,), fetch="one")
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed = hash_password(user.password)
    user_data = db_execute(
        "INSERT INTO users (email, password_hash, full_name, created_at) VALUES (%s, %s, %s, %s) RETURNING id, email, full_name, currency, created_at",
        (user.email, hashed, user.full_name, datetime.utcnow()),
        fetch="one"
    )
    token = create_token(user_data["id"], user_data["email"])
    return Token(access_token=token, token_type="bearer", user={"id": user_data["id"], "email": user_data["email"], "full_name": user_data["full_name"], "currency": user_data["currency"]})

@app.post("/auth/login", response_model=Token)
async def login(credentials: UserLogin):
    user = db_execute("SELECT * FROM users WHERE email = %s", (credentials.email,), fetch="one")
    if not user or not verify_password(credentials.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_token(user["id"], user["email"])
    return Token(access_token=token, token_type="bearer", user={"id": user["id"], "email": user["email"], "full_name": user["full_name"], "currency": user["currency"]})

@app.get("/auth/me", response_model=UserOut)
async def get_me(current_user: Dict = Depends(get_current_user)):
    user = db_execute("SELECT * FROM users WHERE id = %s", (current_user["user_id"],), fetch="one")
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return UserOut(id=user["id"], email=user["email"], full_name=user["full_name"], currency=user["currency"], created_at=str(user["created_at"]))

@app.post("/transactions", response_model=TxOut)
async def create_tx(tx: TxCreate, current_user: Dict = Depends(get_current_user)):
    try:
        datetime.strptime(tx.date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Date must be YYYY-MM-DD")
    row = db_execute(
        "INSERT INTO transactions (user_id, type, amount, category, note, date, source, created_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING *",
        (current_user["user_id"], tx.type, tx.amount, tx.category, tx.note, tx.date, tx.source, datetime.utcnow()),
        fetch="one"
    )
    return TxOut(id=row["id"], user_id=row["user_id"], type=row["type"], amount=float(row["amount"]), category=row["category"], note=row["note"] or "", date=str(row["date"]), source=row["source"], created_at=str(row["created_at"]))

@app.get("/transactions", response_model=TxList)
async def list_txs(limit: int = 50, current_user: Dict = Depends(get_current_user)):
    limit = max(1, min(limit, 200))
    rows = db_execute(
        "SELECT * FROM transactions WHERE user_id = %s ORDER BY date DESC, id DESC LIMIT %s",
        (current_user["user_id"], limit),
        fetch="all"
    )
    return TxList(items=[TxOut(id=r["id"], user_id=r["user_id"], type=r["type"], amount=float(r["amount"]), category=r["category"], note=r["note"] or "", date=str(r["date"]), source=r["source"], created_at=str(r["created_at"])) for r in rows])

@app.delete("/transactions/{tx_id}")
async def delete_transaction(tx_id: int, current_user: Dict = Depends(get_current_user)):
    existing = db_execute("SELECT id FROM transactions WHERE id = %s AND user_id = %s", (tx_id, current_user["user_id"]), fetch="one")
    if not existing:
        raise HTTPException(status_code=404, detail="Transaction not found")
    db_execute("DELETE FROM transactions WHERE id = %s AND user_id = %s", (tx_id, current_user["user_id"]))
    return {"status": "deleted", "id": tx_id}

class TxUpdate(BaseModel):
    type: str = Field(..., pattern="^(income|expense)$")
    amount: float = Field(..., gt=0)
    category: str
    note: str = ""
    date: str

@app.put("/transactions/{tx_id}", response_model=TxOut)
async def update_transaction(tx_id: int, tx: TxUpdate, current_user: Dict = Depends(get_current_user)):
    try:
        datetime.strptime(tx.date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Date must be YYYY-MM-DD")
    existing = db_execute(
        "SELECT id FROM transactions WHERE id = %s AND user_id = %s",
        (tx_id, current_user["user_id"]), fetch="one"
    )
    if not existing:
        raise HTTPException(status_code=404, detail="Transaction not found")
    row = db_execute(
        "UPDATE transactions SET type=%s, amount=%s, category=%s, note=%s, date=%s WHERE id=%s AND user_id=%s RETURNING *",
        (tx.type, tx.amount, tx.category, tx.note, tx.date, tx_id, current_user["user_id"]),
        fetch="one"
    )
    return TxOut(id=row["id"], user_id=row["user_id"], type=row["type"], amount=float(row["amount"]),
                 category=row["category"], note=row["note"] or "", date=str(row["date"]),
                 source=row["source"], created_at=str(row["created_at"]))

@app.get("/summary", response_model=SummaryOut)
async def summary(current_user: Dict = Depends(get_current_user)):
    user_id = current_user["user_id"]
    income = float(db_execute("SELECT COALESCE(SUM(amount), 0) AS s FROM transactions WHERE user_id = %s AND type = 'income'", (user_id,), fetch="one")["s"])
    expense = float(db_execute("SELECT COALESCE(SUM(amount), 0) AS s FROM transactions WHERE user_id = %s AND type = 'expense'", (user_id,), fetch="one")["s"])
    cat_rows = db_execute("SELECT category, COALESCE(SUM(amount), 0) AS s FROM transactions WHERE user_id = %s AND type = 'expense' GROUP BY category ORDER BY s DESC", (user_id,), fetch="all")
    by_cat = {row["category"]: float(row["s"]) for row in cat_rows}
    goal_rows = db_execute("SELECT * FROM budget_goals WHERE user_id = %s", (user_id,), fetch="all")
    budget_alerts = []
    for goal in goal_rows:
        cat_spend = by_cat.get(goal["category"], 0)
        limit = float(goal["monthly_limit"])
        if cat_spend >= limit:
            budget_alerts.append({"category": goal["category"], "spent": cat_spend, "limit": limit, "status": "exceeded"})
        elif cat_spend >= limit * 0.8:
            budget_alerts.append({"category": goal["category"], "spent": cat_spend, "limit": limit, "status": "warning"})
    tip = "Add transactions to see insights."
    if expense > 0 and by_cat:
        top_cat = max(by_cat.items(), key=lambda x: x[1])[0]
        share = (by_cat[top_cat] / expense) * 100
        tip = f"Your top spending category is {top_cat} (~{share:.0f}%). Consider setting a budget for it."
    return SummaryOut(income=income, expense=expense, net=income - expense, by_category=by_cat, budget_alerts=budget_alerts, tip=tip)

@app.get("/analytics", response_model=AnalyticsOut)
async def analytics(current_user: Dict = Depends(get_current_user)):
    user_id = current_user["user_id"]
    trend_rows = db_execute(
        "SELECT TO_CHAR(date, 'YYYY-MM') as month, SUM(CASE WHEN type='income' THEN amount ELSE 0 END) as income, SUM(CASE WHEN type='expense' THEN amount ELSE 0 END) as expense FROM transactions WHERE user_id = %s GROUP BY month ORDER BY month DESC LIMIT 6",
        (user_id,), fetch="all"
    )
    monthly_trend = [{"month": r["month"], "income": float(r["income"]), "expense": float(r["expense"])} for r in trend_rows]
    monthly_trend.reverse()
    cat_rows = db_execute("SELECT category, SUM(amount) as total FROM transactions WHERE user_id = %s AND type = 'expense' GROUP BY category ORDER BY total DESC", (user_id,), fetch="all")
    category_breakdown = [{"category": r["category"], "amount": float(r["total"])} for r in cat_rows]
    top_rows = db_execute("SELECT category, note, amount, date FROM transactions WHERE user_id = %s AND type = 'expense' ORDER BY amount DESC LIMIT 5", (user_id,), fetch="all")
    top_expenses = [{"category": r["category"], "note": r["note"], "amount": float(r["amount"]), "date": str(r["date"])} for r in top_rows]
    total_income = float(db_execute("SELECT COALESCE(SUM(amount), 0) AS s FROM transactions WHERE user_id = %s AND type = 'income'", (user_id,), fetch="one")["s"])
    total_expense = float(db_execute("SELECT COALESCE(SUM(amount), 0) AS s FROM transactions WHERE user_id = %s AND type = 'expense'", (user_id,), fetch="one")["s"])
    savings_rate = ((total_income - total_expense) / total_income * 100) if total_income > 0 else 0
    days = db_execute("SELECT COUNT(DISTINCT date) as days FROM transactions WHERE user_id = %s AND type = 'expense'", (user_id,), fetch="one")["days"] or 1
    return AnalyticsOut(monthly_trend=monthly_trend, category_breakdown=category_breakdown, top_expenses=top_expenses, savings_rate=savings_rate, avg_daily_spend=total_expense / days)

@app.post("/budget-goals")
async def set_budget_goal(goal: BudgetGoal, current_user: Dict = Depends(get_current_user)):
    db_execute("DELETE FROM budget_goals WHERE user_id = %s AND category = %s", (current_user["user_id"], goal.category))
    db_execute("INSERT INTO budget_goals (user_id, category, monthly_limit, created_at) VALUES (%s, %s, %s, %s)", (current_user["user_id"], goal.category, goal.monthly_limit, datetime.utcnow()))
    return {"status": "success", "category": goal.category, "limit": goal.monthly_limit}

@app.get("/budget-goals")
async def get_budget_goals(current_user: Dict = Depends(get_current_user)):
    rows = db_execute("SELECT * FROM budget_goals WHERE user_id = %s", (current_user["user_id"],), fetch="all")
    return {"goals": [dict(r) for r in rows]}

@app.delete("/budget-goals/{category}")
async def delete_budget_goal(category: str, current_user: Dict = Depends(get_current_user)):
    existing = db_execute("SELECT id FROM budget_goals WHERE user_id = %s AND category = %s", (current_user["user_id"], category), fetch="one")
    if not existing:
        raise HTTPException(status_code=404, detail="Budget goal not found")
    db_execute("DELETE FROM budget_goals WHERE user_id = %s AND category = %s", (current_user["user_id"], category))
    return {"status": "deleted", "category": category}

@app.post("/receipt/parse", response_model=ReceiptResult)
async def receipt_parse(payload: ReceiptIngest, current_user: Dict = Depends(get_current_user)):
    """Parse receipt — uses vision AI if image provided, otherwise falls back to OCR text."""
    entries = guess_receipt_entries(
        ocr_text=payload.ocr_text,
        mode=payload.mode,
        image_base64=payload.image_base64,
        image_media_type=payload.image_media_type
    )
    return ReceiptResult(saved=0, extracted=entries)

@app.post("/receipt/confirm")
async def receipt_confirm(payload: Dict[str, Any], current_user: Dict = Depends(get_current_user)):
    entries = payload.get("entries", [])
    if not entries:
        raise HTTPException(status_code=400, detail="No entries to save")
    saved_transactions = []
    for e in entries:
        tx = TxCreate(
            type=e.get("type", "expense"),
            amount=float(e.get("amount", 0)),
            category=e.get("category", "Other"),
            note=e.get("note", "Receipt"),
            date=e.get("date", datetime.now().strftime("%Y-%m-%d")),
            source="receipt",
        )
        saved_transactions.append(await create_tx(tx, current_user))
    return {"status": "success", "saved": len(saved_transactions), "transactions": saved_transactions}

@app.post("/voice/parse", response_model=VoiceParseOut)
async def voice_parse(payload: VoiceParseIn):
    return parse_voice_text(payload.text)

@app.post("/ai/chat", response_model=ChatOut)
async def ai_chat(payload: ChatIn, current_user: Dict = Depends(get_current_user)):
    summary_data = await summary(current_user)
    reply = await get_ai_response(payload.message, {"income": summary_data.income, "expense": summary_data.expense, "by_category": summary_data.by_category})
    return ChatOut(reply=reply)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "2.0",
        "database": "postgresql",
        "openai": "enabled" if openai_client else "disabled",
        "anthropic_vision": "enabled" if anthropic_client else "disabled"
    }
