from __future__ import annotations

import os
import re
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

# Load environment variables
load_dotenv()

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL")
SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_hex(32))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

app = FastAPI(title="AI Budget Tracker API", version="2.0")

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:5500",
        "http://127.0.0.1:5500",  
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*"  # Allow all for development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# -------------------------
# Database helpers
# -------------------------
def get_db():
    """Get database connection"""
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    try:
        yield conn
    finally:
        conn.close()

def db_execute(query: str, params: tuple = None, fetch: str = None):
    """Execute database query"""
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

# -------------------------
# Auth Models
# -------------------------
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

# -------------------------
# Transaction Models
# -------------------------
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

# -------------------------
# Auth Helpers
# -------------------------
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

# -------------------------
# Categorizer
# -------------------------
CATEGORY_RULES = [
    ("Food", ["grocery", "grocer", "restaurant", "cafe", "coffee", "pizza", "burger", "mcd", "kfc", "food", "superstore", "walmart", "costco"]),
    ("Transport", ["uber", "lyft", "bus", "taxi", "fuel", "petrol", "gas", "shell", "esso", "chevron", "transit", "parking"]),
    ("Shopping", ["amazon", "shopping", "mall", "clothes", "fashion", "ikea", "store", "purchase"]),
    ("Utilities", ["hydro", "electric", "water", "internet", "wifi", "telus", "bell", "rogers", "utility", "phone bill"]),
    ("Entertainment", ["movie", "cinema", "netflix", "spotify", "game", "ticket", "concert", "entertain"]),
    ("Health", ["pharmacy", "drug", "clinic", "doctor", "dental", "medicine", "health"]),
    ("Rent", ["rent", "landlord", "lease", "mortgage"]),
    ("Salary", ["salary", "payroll", "paycheque", "paycheck", "wage"]),
]

def categorize_from_text(text: str) -> str:
    t = (text or "").lower()
    for cat, keys in CATEGORY_RULES:
        if any(k in t for k in keys):
            return cat
    return "Other"

# -------------------------
# Receipt Parsing with OpenAI (Enhanced)
# -------------------------
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

def guess_receipt_entries(ocr_text: str, mode: str) -> List[Dict[str, Any]]:
    """Enhanced receipt parsing with OpenAI - returns entries for verification, doesn't save yet"""
    text = (ocr_text or "").strip()
    if not text:
        return []

    # Try OpenAI-powered parsing first
    if openai_client and len(text) > 20:
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert receipt parser. Analyze receipt text and extract ACCURATE transaction data.

CRITICAL RULES:
1. IGNORE card numbers (16 digits like 4532-1234-5678-9012)
2. IGNORE transaction IDs, reference numbers, terminal IDs
3. IGNORE phone numbers
4. Find the TOTAL or AMOUNT DUE - this is usually labeled "TOTAL", "AMOUNT", "BALANCE DUE"
5. Look for merchant/store name at the TOP of receipt
6. Date is usually near top or bottom
7. Amounts are usually formatted like: $12.99, 12.99, or 1299 (in cents)
8. Filter out unrealistic amounts (> $500 for single receipts, < $0.50)

Return JSON array:
- For "single" mode: ONE object with the total amount
- For "multiple" mode: Separate objects for each line item (if itemized receipt)
- For "combination" mode: Both total AND major items if available

Each object needs:
{
  "type": "expense",
  "amount": <number between 0.50 and 500>,
  "category": "Food/Transport/Shopping/Utilities/Entertainment/Health/Rent/Other",
  "note": "<merchant name or item description>",
  "date": "YYYY-MM-DD",
  "source": "receipt",
  "merchant": "<store name>",
  "confidence": "high/medium/low"
}"""
                    },
                    {
                        "role": "user",
                        "content": f"Mode: {mode}\n\nReceipt OCR Text:\n{text}\n\nParse this receipt carefully. Focus on finding the correct total amount and merchant name."
                    }
                ],
                temperature=0.1,  # Lower temperature for more consistent parsing
                max_tokens=800
            )
            
            result = response.choices[0].message.content
            
            # Remove markdown code blocks if present
            result = result.replace('```json', '').replace('```', '').strip()
            
            # Parse JSON from response
            import json
            entries = json.loads(result)
            
            # Ensure it's a list
            if isinstance(entries, dict):
                entries = [entries]
            
            # Validate and format with strict filtering
            today = datetime.now().strftime("%Y-%m-%d")
            formatted = []
            
            for e in entries:
                amount = float(e.get("amount", 0))
                
                # Strict validation
                if amount < 0.50:
                    print(f"Rejected amount too low: {amount}")
                    continue
                if amount > 1000:
                    print(f"Rejected amount too high: {amount}")
                    continue
                
                # Check if amount looks like a card number (too many digits)
                amount_str = str(amount)
                if len(amount_str.replace('.', '')) > 6:
                    print(f"Rejected: looks like card/ID number: {amount}")
                    continue
                
                formatted.append({
                    "type": "expense",
                    "amount": round(amount, 2),
                    "category": e.get("category", "Other"),
                    "note": e.get("note", e.get("merchant", "Receipt scan")),
                    "date": e.get("date", today),
                    "source": "receipt",
                    "merchant": e.get("merchant", "Unknown"),
                    "confidence": e.get("confidence", "medium")
                })
            
            if formatted:
                return formatted
                
        except Exception as e:
            print(f"OpenAI parsing failed, falling back to regex: {e}")
            # Fall back to regex-based parsing

    # Fallback: Improved Regex-based parsing
    amounts = extract_amounts(text)
    
    # Filter out obvious card numbers and IDs
    filtered_amounts = []
    for amt in amounts:
        # Skip if looks like card number (4 groups of 4 digits)
        if amt > 1000 and len(str(int(amt))) >= 10:
            continue
        # Skip unrealistic amounts
        if amt > 500 or amt < 0.50:
            continue
        filtered_amounts.append(amt)
    
    if not filtered_amounts:
        return []

    amounts_sorted = sorted(filtered_amounts)
    
    # Try to find merchant name from first few lines
    lines = text.split('\n')
    merchant = "Unknown Store"
    for line in lines[:5]:  # Check first 5 lines
        line = line.strip()
        if len(line) > 3 and not any(char.isdigit() for char in line):
            merchant = line[:50]  # First non-numeric line is usually merchant
            break
    
    cat = categorize_from_text(text + " " + merchant)
    today = datetime.now().strftime("%Y-%m-%d")

    if mode == "single":
        # Use the largest amount as total
        total = amounts_sorted[-1]
        return [{
            "type": "expense",
            "amount": round(float(total), 2),
            "category": cat,
            "note": f"{merchant}",
            "date": today,
            "source": "receipt",
            "merchant": merchant,
            "confidence": "medium"
        }]

    # For multiple mode, pick top 3-5 distinct amounts
    picked: List[float] = []
    for v in sorted(filtered_amounts, reverse=True):
        if len(picked) >= 5:
            break
        # Skip near-duplicates
        if any(abs(v - p) < 0.10 for p in picked):
            continue
        picked.append(v)

    entries: List[Dict[str, Any]] = []
    for i, v in enumerate(picked, start=1):
        entries.append({
            "type": "expense",
            "amount": round(float(v), 2),
            "category": cat,
            "note": f"{merchant} - Item {i}",
            "date": today,
            "source": "receipt",
            "merchant": merchant,
            "confidence": "low"
        })
    
    return entries

# -------------------------
# Voice Parsing with OpenAI (Enhanced)
# -------------------------
def parse_voice_text(text: str) -> VoiceParseOut:
    """Enhanced voice parsing with OpenAI - handles single transaction for now"""
    t = (text or "").strip()
    if not t:
        raise HTTPException(status_code=400, detail="Empty voice text.")

    # Try OpenAI-powered parsing first
    if openai_client:
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a voice transaction parser. Extract ONE transaction from spoken text (use the first/main one if multiple mentioned).

Return ONLY valid JSON with these fields:
{
  "type": "income" OR "expense",
  "amount": <number>,
  "category": "Food/Transport/Shopping/Utilities/Entertainment/Health/Rent/Salary/Other",
  "note": "<brief description>"
}

RULES:
- "income" keywords: salary, pay, income, wage, paycheck, earned, got paid
- Everything else is "expense"
- Amount must be a valid number (e.g., 18.99, 12.5, 3500)
- Category must match one from the list
- Note should be short and descriptive

Examples:
"grocery 18.99" → {"type": "expense", "amount": 18.99, "category": "Food", "note": "Grocery shopping"}
"uber ride 12.50" → {"type": "expense", "amount": 12.50, "category": "Transport", "note": "Uber ride"}
"got salary 3500" → {"type": "income", "amount": 3500, "category": "Salary", "note": "Salary payment"}
"coffee 5 and lunch 12" → {"type": "expense", "amount": 5, "category": "Food", "note": "Coffee"}
"""
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
            
            # Remove markdown code blocks if present
            result = result.replace('```json', '').replace('```', '').strip()
            
            import json
            data = json.loads(result)
            
            # Validate
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
                note=data.get("note", t)[:100]  # Limit note length
            )
        except Exception as e:
            print(f"OpenAI voice parsing failed, falling back to regex: {e}")
            # Fall back to regex

    # Fallback: Regex-based parsing
    m = re.search(r"(\d{1,5}(?:[.,]\d{1,2})?)", t.replace(",", "."))
    if not m:
        raise HTTPException(status_code=400, detail="No amount found in voice text. Please say the amount clearly.")
    amount = float(m.group(1))

    lower = t.lower()
    tx_type = "income" if any(k in lower for k in ["salary", "pay", "income", "wage", "paycheck", "earned", "got paid"]) else "expense"
    category = categorize_from_text(lower)
    note = t[:100]

    return VoiceParseOut(type=tx_type, amount=round(amount, 2), category=category, note=note)

# -------------------------
# AI Chat with OpenAI
# -------------------------
async def get_ai_response(message: str, user_context: Dict[str, Any]) -> str:
    """AI-powered financial assistant using OpenAI"""
    
    if not openai_client:
        # Fallback to simple rule-based responses
        return get_simple_response(message, user_context)
    
    try:
        # Build context for AI
        total_income = user_context.get("income", 0)
        total_expense = user_context.get("expense", 0)
        by_cat = user_context.get("by_category", {})
        
        context_str = f"""
User's Financial Summary:
- Total Income: ${total_income:.2f}
- Total Expenses: ${total_expense:.2f}
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
                    "content": """You are a helpful personal finance assistant. You help users understand their spending habits and provide actionable financial advice.

Guidelines:
- Be friendly, encouraging, and concise
- Use emojis sparingly (1-2 max)
- Give specific, actionable advice
- Reference their actual spending data
- Keep responses under 150 words
- Focus on insights, not just repeating data
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
        # Fallback to simple responses
        return get_simple_response(message, user_context)

def get_simple_response(message: str, user_context: Dict[str, Any]) -> str:
    """Simple rule-based responses (fallback)"""
    msg = message.lower()
    
    total_income = user_context.get("income", 0)
    total_expense = user_context.get("expense", 0)
    by_cat = user_context.get("by_category", {})
    
    if "how much" in msg and "spent" in msg:
        if "food" in msg:
            food_amt = by_cat.get("Food", 0)
            return f"You've spent ${food_amt:.2f} on Food this month."
        elif "transport" in msg:
            transport_amt = by_cat.get("Transport", 0)
            return f"You've spent ${transport_amt:.2f} on Transport this month."
        else:
            return f"Your total expenses are ${total_expense:.2f} this month."
    
    if "save" in msg or "saving" in msg:
        net = total_income - total_expense
        if total_income > 0:
            savings_rate = (net / total_income) * 100
            return f"You're saving ${net:.2f} ({savings_rate:.1f}% of income). Great job!"
        return "Add your income to see savings rate."
    
    if "tip" in msg or "advice" in msg or "help" in msg:
        if not by_cat:
            return "Start tracking expenses to get personalized tips!"
        
        top_cat = max(by_cat.items(), key=lambda x: x[1])[0] if by_cat else "Unknown"
        top_amt = by_cat.get(top_cat, 0)
        
        tips = [
            f"💡 Your highest spending is on {top_cat} (${top_amt:.2f}). Consider setting a monthly budget.",
            f"📊 Track your {top_cat} expenses weekly to identify patterns.",
            f"💰 Try the 50/30/20 rule: 50% needs, 30% wants, 20% savings.",
            f"🎯 Set a goal to reduce {top_cat} spending by 10% next month."
        ]
        return tips[len(by_cat) % len(tips)]
    
    if "budget" in msg:
        return "Set category budgets in the app to get alerts when you're close to limits!"
    
    return "I can help with: spending analysis, savings tips, budget advice. Try asking 'How much did I spend?' or 'Give me tips'."

# -------------------------
# Auth Endpoints
# -------------------------
@app.post("/auth/register", response_model=Token)
async def register(user: UserRegister):
    # Check if user exists
    existing = db_execute(
        "SELECT id FROM users WHERE email = %s",
        (user.email,),
        fetch="one"
    )
    
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create user
    hashed = hash_password(user.password)
    
    user_data = db_execute(
        """
        INSERT INTO users (email, password_hash, full_name, created_at)
        VALUES (%s, %s, %s, %s)
        RETURNING id, email, full_name, currency, created_at
        """,
        (user.email, hashed, user.full_name, datetime.utcnow()),
        fetch="one"
    )
    
    # Create token
    token = create_token(user_data["id"], user_data["email"])
    
    return Token(
        access_token=token,
        token_type="bearer",
        user={
            "id": user_data["id"],
            "email": user_data["email"],
            "full_name": user_data["full_name"],
            "currency": user_data["currency"]
        }
    )

@app.post("/auth/login", response_model=Token)
async def login(credentials: UserLogin):
    user = db_execute(
        "SELECT * FROM users WHERE email = %s",
        (credentials.email,),
        fetch="one"
    )
    
    if not user or not verify_password(credentials.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = create_token(user["id"], user["email"])
    
    return Token(
        access_token=token,
        token_type="bearer",
        user={
            "id": user["id"],
            "email": user["email"],
            "full_name": user["full_name"],
            "currency": user["currency"]
        }
    )

@app.get("/auth/me", response_model=UserOut)
async def get_me(current_user: Dict = Depends(get_current_user)):
    user = db_execute(
        "SELECT * FROM users WHERE id = %s",
        (current_user["user_id"],),
        fetch="one"
    )
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return UserOut(
        id=user["id"],
        email=user["email"],
        full_name=user["full_name"],
        currency=user["currency"],
        created_at=str(user["created_at"])
    )

# -------------------------
# Transaction Endpoints
# -------------------------
@app.post("/transactions", response_model=TxOut)
async def create_tx(tx: TxCreate, current_user: Dict = Depends(get_current_user)):
    try:
        datetime.strptime(tx.date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Date must be YYYY-MM-DD")

    row = db_execute(
        """
        INSERT INTO transactions (user_id, type, amount, category, note, date, source, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING *
        """,
        (current_user["user_id"], tx.type, tx.amount, tx.category, tx.note, tx.date, tx.source, datetime.utcnow()),
        fetch="one"
    )

    return TxOut(
        id=row["id"],
        user_id=row["user_id"],
        type=row["type"],
        amount=float(row["amount"]),
        category=row["category"],
        note=row["note"] or "",
        date=str(row["date"]),
        source=row["source"],
        created_at=str(row["created_at"])
    )

@app.get("/transactions", response_model=TxList)
async def list_txs(limit: int = 50, current_user: Dict = Depends(get_current_user)):
    limit = max(1, min(limit, 200))
    
    rows = db_execute(
        """
        SELECT * FROM transactions
        WHERE user_id = %s
        ORDER BY date DESC, id DESC
        LIMIT %s
        """,
        (current_user["user_id"], limit),
        fetch="all"
    )
    
    items = []
    for r in rows:
        items.append(TxOut(
            id=r["id"],
            user_id=r["user_id"],
            type=r["type"],
            amount=float(r["amount"]),
            category=r["category"],
            note=r["note"] or "",
            date=str(r["date"]),
            source=r["source"],
            created_at=str(r["created_at"])
        ))
    
    return TxList(items=items)

@app.get("/summary", response_model=SummaryOut)
async def summary(current_user: Dict = Depends(get_current_user)):
    user_id = current_user["user_id"]

    income_row = db_execute(
        "SELECT COALESCE(SUM(amount), 0) AS s FROM transactions WHERE user_id = %s AND type = 'income'",
        (user_id,),
        fetch="one"
    )
    income = float(income_row["s"])

    expense_row = db_execute(
        "SELECT COALESCE(SUM(amount), 0) AS s FROM transactions WHERE user_id = %s AND type = 'expense'",
        (user_id,),
        fetch="one"
    )
    expense = float(expense_row["s"])

    cat_rows = db_execute(
        """
        SELECT category, COALESCE(SUM(amount), 0) AS s
        FROM transactions
        WHERE user_id = %s AND type = 'expense'
        GROUP BY category
        ORDER BY s DESC
        """,
        (user_id,),
        fetch="all"
    )
    by_cat = {row["category"]: float(row["s"]) for row in cat_rows}

    # Budget alerts
    goal_rows = db_execute(
        "SELECT * FROM budget_goals WHERE user_id = %s",
        (user_id,),
        fetch="all"
    )
    
    budget_alerts = []
    for goal in goal_rows:
        cat_spend = by_cat.get(goal["category"], 0)
        limit = float(goal["monthly_limit"])
        if cat_spend >= limit:
            budget_alerts.append({
                "category": goal["category"],
                "spent": cat_spend,
                "limit": limit,
                "status": "exceeded"
            })
        elif cat_spend >= limit * 0.8:
            budget_alerts.append({
                "category": goal["category"],
                "spent": cat_spend,
                "limit": limit,
                "status": "warning"
            })

    net = income - expense
    tip = "Add transactions to see insights."
    if expense > 0 and by_cat:
        top_cat = max(by_cat.items(), key=lambda x: x[1])[0]
        top_amt = by_cat[top_cat]
        share = (top_amt / expense) * 100 if expense else 0
        tip = f"Your top spending category is {top_cat} (~{share:.0f}%). Consider setting a budget for it."

    return SummaryOut(
        income=income,
        expense=expense,
        net=net,
        by_category=by_cat,
        budget_alerts=budget_alerts,
        tip=tip,
    )

@app.get("/analytics", response_model=AnalyticsOut)
async def analytics(current_user: Dict = Depends(get_current_user)):
    user_id = current_user["user_id"]
    
    # Monthly trend
    trend_rows = db_execute(
        """
        SELECT TO_CHAR(date, 'YYYY-MM') as month,
               SUM(CASE WHEN type='income' THEN amount ELSE 0 END) as income,
               SUM(CASE WHEN type='expense' THEN amount ELSE 0 END) as expense
        FROM transactions
        WHERE user_id = %s
        GROUP BY month
        ORDER BY month DESC
        LIMIT 6
        """,
        (user_id,),
        fetch="all"
    )
    monthly_trend = [{"month": r["month"], "income": float(r["income"]), "expense": float(r["expense"])} for r in trend_rows]
    monthly_trend.reverse()
    
    # Category breakdown
    cat_rows = db_execute(
        """
        SELECT category, SUM(amount) as total
        FROM transactions
        WHERE user_id = %s AND type = 'expense'
        GROUP BY category
        ORDER BY total DESC
        """,
        (user_id,),
        fetch="all"
    )
    category_breakdown = [{"category": r["category"], "amount": float(r["total"])} for r in cat_rows]
    
    # Top expenses
    top_rows = db_execute(
        """
        SELECT category, note, amount, date
        FROM transactions
        WHERE user_id = %s AND type = 'expense'
        ORDER BY amount DESC
        LIMIT 5
        """,
        (user_id,),
        fetch="all"
    )
    top_expenses = [{"category": r["category"], "note": r["note"], "amount": float(r["amount"]), "date": str(r["date"])} for r in top_rows]
    
    # Savings rate
    income_row = db_execute(
        "SELECT COALESCE(SUM(amount), 0) AS s FROM transactions WHERE user_id = %s AND type = 'income'",
        (user_id,),
        fetch="one"
    )
    total_income = float(income_row["s"])
    
    expense_row = db_execute(
        "SELECT COALESCE(SUM(amount), 0) AS s FROM transactions WHERE user_id = %s AND type = 'expense'",
        (user_id,),
        fetch="one"
    )
    total_expense = float(expense_row["s"])
    
    savings_rate = ((total_income - total_expense) / total_income * 100) if total_income > 0 else 0
    
    # Avg daily spend
    days_row = db_execute(
        "SELECT COUNT(DISTINCT date) as days FROM transactions WHERE user_id = %s AND type = 'expense'",
        (user_id,),
        fetch="one"
    )
    days = days_row["days"] or 1
    avg_daily_spend = total_expense / days
    
    return AnalyticsOut(
        monthly_trend=monthly_trend,
        category_breakdown=category_breakdown,
        top_expenses=top_expenses,
        savings_rate=savings_rate,
        avg_daily_spend=avg_daily_spend
    )

# -------------------------
# Budget Goals
# -------------------------
@app.post("/budget-goals")
async def set_budget_goal(goal: BudgetGoal, current_user: Dict = Depends(get_current_user)):
    # Delete existing
    db_execute(
        "DELETE FROM budget_goals WHERE user_id = %s AND category = %s",
        (current_user["user_id"], goal.category)
    )
    
    # Insert new
    db_execute(
        "INSERT INTO budget_goals (user_id, category, monthly_limit, created_at) VALUES (%s, %s, %s, %s)",
        (current_user["user_id"], goal.category, goal.monthly_limit, datetime.utcnow())
    )
    
    return {"status": "success", "category": goal.category, "limit": goal.monthly_limit}

@app.get("/budget-goals")
async def get_budget_goals(current_user: Dict = Depends(get_current_user)):
    rows = db_execute(
        "SELECT * FROM budget_goals WHERE user_id = %s",
        (current_user["user_id"],),
        fetch="all"
    )
    goals = [dict(r) for r in rows]
    return {"goals": goals}

# -------------------------
# Receipt & Voice
# -------------------------
@app.post("/receipt/parse", response_model=ReceiptResult)
async def receipt_parse(payload: ReceiptIngest, current_user: Dict = Depends(get_current_user)):
    """Parse receipt and return extracted data for user verification - DOES NOT SAVE YET"""
    entries = guess_receipt_entries(payload.ocr_text, payload.mode)
    
    # Return extracted data without saving
    return ReceiptResult(
        saved=0,  # Not saved yet
        extracted=entries
    )

@app.post("/receipt/confirm")
async def receipt_confirm(payload: Dict[str, Any], current_user: Dict = Depends(get_current_user)):
    """Save verified receipt entries after user confirmation"""
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
        
        saved_tx = await create_tx(tx, current_user)
        saved_transactions.append(saved_tx)
    
    return {
        "status": "success",
        "saved": len(saved_transactions),
        "transactions": saved_transactions
    }

@app.post("/voice/parse", response_model=VoiceParseOut)
async def voice_parse(payload: VoiceParseIn):
    return parse_voice_text(payload.text)

# -------------------------
# AI Chat
# -------------------------
@app.post("/ai/chat", response_model=ChatOut)
async def ai_chat(payload: ChatIn, current_user: Dict = Depends(get_current_user)):
    summary_data = await summary(current_user)
    context = {
        "income": summary_data.income,
        "expense": summary_data.expense,
        "by_category": summary_data.by_category
    }
    
    reply = await get_ai_response(payload.message, context)
    return ChatOut(reply=reply)

# -------------------------
# Health
# -------------------------
@app.get("/health")
def health():
    openai_status = "enabled" if openai_client else "disabled (using fallback)"
    return {
        "status": "ok",
        "version": "2.0",
        "database": "postgresql",
        "ai": openai_status

    }
