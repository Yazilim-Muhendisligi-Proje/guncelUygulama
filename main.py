"""
GNN-AIOps Backend — FastAPI + GAT Model + SQLite Streaming + RBAC Auth
Port: 5003
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
import time
from pathlib import Path

import jwt  # PyJWT
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from torch import nn
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "gigafactory_gat_model.pth"
DB_PATH = BASE_DIR / "gigafactory_live.db"
USERS_PATH = BASE_DIR / "users.json"

# ─── Auth Config ──────────────────────────────────────────────────────────────
JWT_SECRET = "nexus-aiops-super-secret-2025"
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = 8

# ─── Load & Save Users ────────────────────────────────────────────────────────
# ─── SQLite User Helpers ──────────────────────────────────────────────────────
def get_db_connection():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def get_user_by_username(username: str):
    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    if user:
        # Convert Row to dict and handle comma-separated pages
        u = dict(user)
        u["pages"] = u["pages"].split(",") if u["pages"] else []
        return u
    return None

def save_new_user(user_data: dict):
    conn = get_db_connection()
    pages_str = ",".join(user_data.get("pages", ["dashboard", "profile"]))
    conn.execute('''
        INSERT INTO users (username, password, name, initials, email, department, role, pages)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        user_data["username"],
        user_data["password"],
        user_data["name"],
        user_data["initials"],
        user_data["email"],
        user_data["department"],
        user_data["role"],
        pages_str
    ))
    conn.commit()
    conn.close()

def update_user_db(username: str, update_data: dict):
    conn = get_db_connection()
    # Build query dynamically based on provided fields
    fields = []
    values = []
    for k, v in update_data.items():
        if k == "pages" and isinstance(v, list):
            v = ",".join(v)
        fields.append(f"{k} = ?")
        values.append(v)
    
    if not fields:
        return
        
    query = f"UPDATE users SET {', '.join(fields)} WHERE username = ?"
    values.append(username)
    conn.execute(query, values)
    conn.commit()
    conn.close()

# ─── GAT Model ────────────────────────────────────────────────────────────────
class GATAnomalyModel(nn.Module):
    """
    2-layer GAT (Gigafactory):
      conv1: GATConv(in=4, out=32, heads=4)  → [N, 128]
      conv2: GATConv(in=128, out=32, heads=1) → [N, 32]
      lin  : Linear(32, 2)
    """
    def __init__(self):
        super().__init__()
        self.conv1 = GATConv(4, 32, heads=4, concat=True)
        self.conv2 = GATConv(128, 32, heads=1, concat=True)
        self.lin = nn.Linear(32, 2)

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index
        # conv1
        x, (edge_idx1, attn1) = self.conv1(x, edge_index, return_attention_weights=True)
        x = F.elu(x)
        # conv2 — captures per-edge attention scalars
        x, (edge_idx2, attn2) = self.conv2(x, edge_index, return_attention_weights=True)
        x = F.elu(x)
        # Global mean pooling
        out = x.mean(dim=0, keepdim=True)
        logits = self.lin(out)
        return logits, attn2  # attn2: [num_edges, 1]


# ─── 4-node fully-connected edge index ────────────────────────────────────────
def build_edge_index(num_nodes: int = 4) -> torch.Tensor:
    """All directed edges among 4 nodes (excluding self-loops)."""
    rows, cols = [], []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)
    return torch.tensor([rows, cols], dtype=torch.long)


EDGE_INDEX = build_edge_index(4)

# ─── Shared State ─────────────────────────────────────────────────────────────
state: dict = {}


# ─── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model
    model = GATAnomalyModel()
    state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    state["model"] = model
    state["total_rows"] = 500000    # gigafactory_live.db has 500K rows

    print(f"[OK] GAT model loaded from {MODEL_PATH}")
    print(f"[OK] DB: {DB_PATH}  ({state['total_rows']:,} rows)")
    yield
    # Cleanup (nothing heavy needed)


# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="GNN-AIOps", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Column mapping ───────────────────────────────────────────────────────────
COLS = [
    "net_lat", "net_thr", "net_p_loss", "net_sec",        # Node 0: IT/Network
    "iot_vib", "iot_temp", "iot_torq", "iot_cycle",       # Node 1: IoT
    "fin_cost", "fin_fraud", "fin_risk", "fin_inv",        # Node 2: Finans
    "log_path", "log_coll", "log_soc", "log_task",         # Node 3: Lojistik
]

NODE_NAMES = ["IT", "IoT", "Finans", "Lojistik"]


class LoginRequest(BaseModel):
    username: str
    password: str

class UserCreateRequest(BaseModel):
    username: str
    password: str
    role: str
    name: str
    initials: str
    email: str
    department: str
    pages: list[str] = ["dashboard", "profile"]

class ProfileUpdateRequest(BaseModel):
    name: str | None = None
    email: str | None = None
    initials: str | None = None

class PasswordChangeRequest(BaseModel):
    old_password: str
    new_password: str

class EmergencyAuthRequest(BaseModel):
    department: str
    reason: str
    level: str


# ─── /api/login ───────────────────────────────────────────────────────────────
@app.post("/api/login")
async def login(req: LoginRequest):
    user = get_user_by_username(req.username)
    if not user or user["password"] != req.password:
        raise HTTPException(status_code=401, detail="Kullanıcı adı veya şifre hatalı")

    payload = {
        "sub": user["username"],
        "name": user["name"],
        "role": user["role"],
        "initials": user["initials"],
        "email": user["email"],
        "department": user["department"],
        "pages": user["pages"],
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRE_HOURS)
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return {"token": token, "user": {k: v for k, v in user.items() if k != "password"}}


@app.get("/api/me")
async def me(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Token gerekli")
    token = authorization.split(" ", 1)[1]
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token süresi doldu")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Geçersiz token")

# ─── /api/profile/update ──────────────────────────────────────────────────────
@app.post("/api/profile/update")
async def update_profile(req: ProfileUpdateRequest, authorization: str = Header(None)):
    curr_user = await me(authorization)
    update_data = {k: v for k, v in req.model_dump().items() if v is not None}
    if not update_data:
        return {"message": "Güncellenecek veri yok"}
    
    update_user_db(curr_user["sub"], update_data)
    return {"message": "Profil başarıyla güncellendi", "updated": update_data}

# ─── /api/change-password ─────────────────────────────────────────────────────
@app.post("/api/change-password")
async def change_password(req: PasswordChangeRequest, authorization: str = Header(None)):
    curr_user = await me(authorization)
    user = get_user_by_username(curr_user["sub"])
    
    if not user:
        raise HTTPException(status_code=404, detail="Kullanıcı bulunamadı")
    
    if user["password"] != req.old_password:
        raise HTTPException(status_code=400, detail="Mevcut şifre hatalı")
    
    update_user_db(curr_user["sub"], {"password": req.new_password})
    return {"message": "Şifre başarıyla güncellendi"}

# ─── /api/users (Create User - Admin Only) ───────────────────────────────────
@app.post("/api/users")
async def create_user(req: UserCreateRequest, authorization: str = Header(None)):
    curr_user = await me(authorization)
    if curr_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Bu işlem için yönetici yetkisi gereklidir")
    
    if get_user_by_username(req.username):
        raise HTTPException(status_code=400, detail="Bu kullanıcı adı zaten alınmış")
    
    save_new_user(req.model_dump())
    return {"message": f"Kullanıcı {req.username} başarıyla oluşturuldu"}

# ─── /api/admin/users (List Users - Admin Only) ──────────────────────────────
@app.get("/api/admin/users")
async def list_users(authorization: str = Header(None)):
    curr_user = await me(authorization)
    if curr_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Bu işlem için yönetici yetkisi gereklidir")
    
    conn = get_db_connection()
    users = conn.execute("SELECT username, name, email, department, role FROM users").fetchall()
    conn.close()
    return [dict(u) for u in users]

# ─── /api/admin/users/{username} (Delete User - Admin Only) ─────────────────
@app.delete("/api/admin/users/{username}")
async def delete_user(username: str, authorization: str = Header(None)):
    curr_user = await me(authorization)
    if curr_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Bu işlem için yönetici yetkisi gereklidir")
    
    if username == curr_user["sub"]:
        raise HTTPException(status_code=400, detail="Kendi hesabınızı silemezsiniz")
        
    conn = get_db_connection()
    conn.execute("DELETE FROM users WHERE username = ?", (username,))
    conn.commit()
    conn.close()
    return {"message": f"Kullanıcı {username} başarıyla silindi"}

# ─── /api/emergency-auth ──────────────────────────────────────────────────────
@app.post("/api/emergency-auth")
async def emergency_auth(req: EmergencyAuthRequest, authorization: str = Header(None)):
    curr_user = await me(authorization)
    # Production logic would send alerts to CSO/Admin
    print(f"[EMERGENCY PROTOCOL] User {curr_user['sub']} requested {req.level} for {req.department}. Reason: {req.reason}")
    return {"status": "success", "protocol": "ALPHA-SECURE", "timestamp": datetime.now().isoformat()}


# ─── /api/health ──────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {"status": "ok", "model": "GATAnomalyModel", "rows": state.get("total_rows", 0)}


# ─── /api/stream ──────────────────────────────────────────────────────────────
@app.get("/api/stream")
async def stream():
    model: GATAnomalyModel = state["model"]

    # Deterministic row_id based on current time to ensure synchronization across clients
    # Any client requesting in the same second gets the same row.
    current_time_seconds = int(time.time())
    row_id = (current_time_seconds % state["total_rows"]) + 1

    # Fetch one row from SQLite using ROWID
    conn = sqlite3.connect(str(DB_PATH))
    try:
        cur = conn.cursor()
        cur.execute(
            f"SELECT {', '.join(COLS)} FROM live_telemetry WHERE ROWID = ?",
            (row_id,),
        )
        row = cur.fetchone()
        if row is None:
            # Fallback: deterministic random based on row_id
            np.random.seed(row_id)
            row = tuple(np.random.rand(16).tolist())
    finally:
        conn.close()

    # Build 4×4 feature tensor
    features = torch.tensor(row, dtype=torch.float32).view(4, 4)
    data = Data(x=features, edge_index=EDGE_INDEX)

    with torch.no_grad():
        logits, attn_weights = model(data)
        prob = F.softmax(logits, dim=1)
        anomaly = int(prob[0, 1] > 0.5)
        anomaly_score = float(prob[0, 1])

    # Attention weights: one scalar per edge → aggregate to per-node
    attn = attn_weights.squeeze(-1).numpy()  # shape: [12]
    # Map edges back to destination nodes → mean attention received
    node_attn = np.zeros(4)
    num_nodes = 4
    idx = 0
    for src in range(num_nodes):
        for dst in range(num_nodes):
            if src != dst:
                node_attn[dst] += attn[idx]
                idx += 1
    node_attn /= (num_nodes - 1)  # normalize

    row_dict = dict(zip(COLS, row))

    return JSONResponse({
        "row_id": row_id,
        "anomaly": anomaly,
        "anomaly_score": round(anomaly_score, 4),
        # Raw features — IT/Network node
        "net_lat":    round(row_dict["net_lat"], 4),
        "net_thr":    round(row_dict["net_thr"], 4),
        "net_p_loss": round(row_dict["net_p_loss"], 4),
        "net_sec":    round(row_dict["net_sec"], 4),
        # Raw features — IoT node
        "iot_vib":   round(row_dict["iot_vib"],   4),
        "iot_temp":  round(row_dict["iot_temp"],  4),
        "iot_torq":  round(row_dict["iot_torq"],  4),
        "iot_cycle": round(row_dict["iot_cycle"], 4),
        # Raw features — Finans node
        "fin_cost":  round(row_dict["fin_cost"],  4),
        "fin_fraud": round(row_dict["fin_fraud"], 4),
        "fin_risk":  round(row_dict["fin_risk"],  4),
        "fin_inv":   round(row_dict["fin_inv"],   4),
        # Raw features — Lojistik node
        "log_path":  round(row_dict["log_path"],  4),
        "log_coll":  round(row_dict["log_coll"],  4),
        "log_soc":   round(row_dict["log_soc"],   4),
        "log_task":  round(row_dict["log_task"],  4),
        # Attention weights per node
        "attention_weights": [round(float(v), 4) for v in node_attn],
        # Department risk scores (0-100)
        "risk_scores": {
            "IT":       round(float(node_attn[0]) * 100, 1),
            "IoT":      round(float(node_attn[1]) * 100, 1),
            "Finans":   round(float(node_attn[2]) * 100, 1),
            "HR":       round(float(node_attn[3]) * 100, 1),
        },
    })


# ─── Static Files (HTML Pages) ────────────────────────────────────────────────
@app.get("/")
async def root():
    return RedirectResponse(url="/login.html")

app.mount("/", StaticFiles(directory=str(BASE_DIR), html=True), name="static")


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=5003, reload=False)
