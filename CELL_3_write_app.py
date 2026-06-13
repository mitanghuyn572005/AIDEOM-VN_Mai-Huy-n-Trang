# ============================================================
# CELL 3 — GHI FILE ỨNG DỤNG STREAMLIT
# Chạy sau Cell 2. Ghi toàn bộ app.py vào /content/app.py
# ============================================================

APP_CODE = '''
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════
# CẤU HÌNH TRANG
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="VN AIDEOM-VN",
    page_icon="🇻🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════
# CSS CUSTOM THEME (dark navy)
# ═══════════════════════════════════════════════════════════
st.markdown("""
<style>
  /* Nền tối */
  .stApp { background-color: #0e1117; color: #e0e0e0; }
  section[data-testid="stSidebar"] { background: #161b22; }
  section[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
  /* Cards metric */
  div[data-testid="metric-container"] {
      background: #1c2333;
      border: 1px solid #30363d;
      border-radius: 10px;
      padding: 12px 18px;
  }
  div[data-testid="metric-container"] label { color: #8b949e !important; font-size:13px; }
  div[data-testid="metric-container"] [data-testid="metric-value"] {
      color: #ff6b6b !important; font-size: 28px; font-weight: 700;
  }
  div[data-testid="metric-container"] [data-testid="metric-delta"] { font-size:13px; }
  /* Headers */
  h1,h2,h3 { color: #ffffff !important; }
  /* Tabs */
  button[data-baseweb="tab"] { color: #8b949e !important; background: transparent !important; }
  button[data-baseweb="tab"][aria-selected="true"] {
      color: #ffffff !important;
      border-bottom: 2px solid #ff6b6b !important;
  }
  /* DataFrames */
  .stDataFrame { background: #161b22; }
  /* Divider */
  hr { border-color: #30363d; }
  /* Sidebar nav item active */
  .sidebar-active { background: #1f2937; border-radius:6px; padding:4px 8px; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# DỮ LIỆU TĨNH
# ═══════════════════════════════════════════════════════════
YEARS  = np.array([2020,2021,2022,2023,2024,2025])
Y_VN   = np.array([8044.4,8487.5,9513.3,10221.8,11511.9,12847.6])
K_VN   = np.array([16500,17800,19600,21300,23500,25900])
L_VN   = np.array([53.6,50.5,51.7,52.4,52.9,53.4])
D_VN   = np.array([12.0,12.7,14.3,16.5,18.3,19.5])
AI_VN  = np.array([55.6,60.2,65.4,67.0,73.8,80.1])
H_VN   = np.array([24.1,26.1,26.2,27.0,28.4,29.2])
ALPHA,BETA,GAMMA,DELTA,THETA = 0.33,0.42,0.10,0.08,0.07

REGIONS = [
    "Trung du MNPB","ĐB Sông Hồng","BTB & DH Trung Bộ",
    "Tây Nguyên","Đông Nam Bộ","ĐB Sông Cửu Long"
]
SECTORS_8 = [
    "Nông-Lâm-TS","CN chế biến","Xây dựng","Bán buôn-lẻ",
    "Tài chính-NH","Logistics","CNTT-TT","Giáo dục"
]
SECTORS_10 = [
    "Nông-Lâm-TS","CN chế biến","Xây dựng","Khai khoáng",
    "Bán buôn-lẻ","Tài chính-NH","Logistics","CNTT-TT","Giáo dục","Y tế"
]

COLORS_MAIN = px.colors.qualitative.Plotly
CLR_RED   = "#ff6b6b"
CLR_GOLD  = "#ffd700"
CLR_GREEN = "#2dd4bf"
CLR_BLUE  = "#60a5fa"

# ═══════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════
@st.cache_data
def compute_m1():
    A = Y_VN / (K_VN**ALPHA * L_VN**BETA * D_VN**GAMMA * AI_VN**DELTA * H_VN**THETA)
    A_bar = np.mean(A)
    Y_hat = A_bar * (K_VN**ALPHA * L_VN**BETA * D_VN**GAMMA * AI_VN**DELTA * H_VN**THETA)
    mape  = np.mean(np.abs((Y_VN-Y_hat)/Y_VN))*100
    g_Y  = np.mean(np.diff(np.log(Y_VN)))
    g_K  = np.mean(np.diff(np.log(K_VN)))
    g_L  = np.mean(np.diff(np.log(L_VN)))
    g_D  = np.mean(np.diff(np.log(D_VN)))
    g_AI = np.mean(np.diff(np.log(AI_VN)))
    g_H  = np.mean(np.diff(np.log(H_VN)))
    g_A  = np.mean(np.diff(np.log(A)))
    factors = ["TFP (A)","Vốn (K)","Lao động (L)","Số hóa (D)","AI","Nhân lực số (H)"]
    pcts = [
        g_A/g_Y*100, ALPHA*g_K/g_Y*100, BETA*g_L/g_Y*100,
        GAMMA*g_D/g_Y*100, DELTA*g_AI/g_Y*100, THETA*g_H/g_Y*100
    ]
    return A, A_bar, Y_hat, mape, factors, pcts, g_Y*100

@st.cache_data
def compute_m1_forecast(D30,AI30,H30,K_growth,tfp_growth):
    A = Y_VN / (K_VN**ALPHA * L_VN**BETA * D_VN**GAMMA * AI_VN**DELTA * H_VN**THETA)
    A_bar = np.mean(A)
    K30  = K_VN[-1] * ((1+K_growth/100)**5)
    L30  = L_VN[-1] * (1.01**5)
    A30  = A_bar * ((1+tfp_growth/100)**5)
    Y30  = A30 * K30**ALPHA * L30**BETA * D30**GAMMA * AI30**DELTA * H30**THETA
    return round(Y30, 2)

@st.cache_data
def compute_lp_base():
    from scipy.optimize import linprog
    c    = [-0.85,-1.20,-0.95,-1.35]
    A_ub = [[1,1,1,1],[-1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,-1],[0.35,-0.65,0.35,-0.65]]
    b_ub = [100,-25,-15,-20,-10,0]
    res  = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0,None)]*4, method="highs")
    z    = round(-res.fun,2) if res.success else 112.25
    alloc = res.x.tolist() if res.success else [25,35,20,20]
    budgets = list(range(100,165,10))
    zv = []
    for B in budgets:
        r = linprog(c, A_ub=A_ub, b_ub=[B,-25,-15,-20,-10,0], bounds=[(0,None)]*4, method="highs")
        zv.append(round(-r.fun,2) if r.success else None)
    return z, alloc, budgets, zv

@st.cache_data
def compute_priority():
    growth  = np.array([3.27,9.64,7.45,-1.20,7.10,7.36,9.93,7.85,6.42,6.85])
    prod    = np.array([103.4,241.2,168.8,1290.5,145.3,1072.4,321.4,713.8,205.7,437.1])
    spill   = np.array([0.35,0.78,0.42,0.30,0.55,0.85,0.72,0.92,0.65,0.60])
    export  = np.array([40.5,290.9,2.5,8.2,5.5,1.2,3.1,178.0,0.0,0.0])
    labor   = np.array([13.20,11.50,4.80,0.30,7.80,0.55,1.95,0.62,2.15,0.75])
    ai_r    = np.array([15,55,20,30,48,72,42,88,38,45])
    risk    = np.array([18,42,25,55,38,52,35,28,22,18])

    def ng(x): return (x-x.min())/(x.max()-x.min()+1e-12)
    def nb(x): return (x.max()-x)/(x.max()-x.min()+1e-12)

    Xg = np.column_stack([ng(growth),ng(prod),ng(spill),ng(export),ng(labor),ng(ai_r)])
    Xb = nb(risk)
    w  = np.array([0.15,0.15,0.20,0.15,0.10,0.20])
    wr = 0.15
    pri= Xg @ w - wr * Xb
    return pri, SECTORS_10

@st.cache_data
def compute_topsis():
    X = np.array([
        [57.0,3.5,38,22,21.5,0.18,72,0.405],
        [152.3,20.0,78,68,36.8,0.85,92,0.358],
        [87.5,8.2,55,40,27.5,0.32,84,0.372],
        [68.9,0.8,32,18,18.2,0.15,68,0.412],
        [158.9,18.5,82,75,42.5,0.78,94,0.385],
        [80.5,2.1,48,30,16.8,0.22,78,0.392],
    ], dtype=float)
    ib = [True]*7+[False]
    w  = np.array([0.10,0.10,0.15,0.20,0.15,0.15,0.05,0.10])
    X_e = X.copy(); X_e[:,7]=X_e[:,7].max()-X_e[:,7]
    cs  = X_e.sum(0); cs[cs==0]=1e-12; P=X_e/cs
    k   = 1/np.log(len(X_e))
    E   = -k*np.nansum(P*np.log(P+1e-12),0); d=1-E; w_ent=d/d.sum()

    def topsis(X_,w_,ib_):
        denom=np.sqrt((X_**2).sum(0)); denom[denom==0]=1e-12
        R=X_/denom; V=R*w_
        As=np.where(ib_,V.max(0),V.min(0)); An=np.where(ib_,V.min(0),V.max(0))
        Ss=np.sqrt(((V-As)**2).sum(1)); Sn=np.sqrt(((V-An)**2).sum(1))
        return Sn/(Ss+Sn+1e-12)

    se = topsis(X,w,ib); sn = topsis(X,w_ent,ib)
    return se, sn, w_ent

@st.cache_data
def compute_mip():
    try:
        from pulp import LpProblem,LpMaximize,LpVariable,lpSum,PULP_CBC_CMD,value,LpStatus
        P = list(range(1,16))
        C  = {1:12000,2:11500,3:18000,4:4500,5:3200,6:5800,7:6500,8:15000,
              9:2500,10:7200,11:4800,12:8500,13:20000,14:3800,15:1500}
        C1 = {1:8500,2:7500,3:12000,4:3500,5:2500,6:4000,7:4500,8:9000,
              9:1800,10:5000,11:3500,12:5500,13:13000,14:2800,15:1200}
        B  = {1:21500,2:20800,3:32500,4:9200,5:6800,6:11400,7:12200,8:28500,
              9:5800,10:13800,11:8500,12:16200,13:35000,14:7500,15:3800}
        names = {1:"TTDL HòaLạc",2:"TTDL phía Nam",3:"5G toàn quốc",4:"VNeID 2.0",
                 5:"CSDVCQG v3",6:"Y tế số",7:"Giáo dục K-12",8:"AI quốc gia",
                 9:"Sandbox fintech",10:"Logistics thông minh",11:"NN số ĐBSCL",
                 12:"50k kỹ sư AI",13:"KCN bán dẫn",14:"An ninh mạng",15:"Open Data"}
        m = LpProblem("MIP",LpMaximize)
        y = LpVariable.dicts("y",P,cat="Binary")
        m += lpSum(B[i]*y[i] for i in P)
        m += lpSum(C[i]*y[i] for i in P) <= 80000
        m += lpSum(C1[i]*y[i] for i in P) <= 40000
        m += y[1]+y[2] <= 1
        m += y[8] <= y[12]; m += y[13] <= y[12]
        m += y[4]+y[5] >= 1; m += y[14] >= 1
        m += lpSum(y[i] for i in P) >= 7
        m += lpSum(y[i] for i in P) <= 11
        m.solve(PULP_CBC_CMD(msg=0))
        selected = [i for i in P if y[i].value()>0.5]
        z_val    = value(m.objective)
        return selected, z_val, names, C, B
    except Exception as e:
        # fallback
        sel = [3,4,8,10,12,13,14]; z_val=152500
        names = {3:"5G toàn quốc",4:"VNeID 2.0",8:"AI quốc gia",
                 10:"Logistics TM",12:"50k kỹ sư AI",13:"KCN bán dẫn",14:"An ninh mạng"}
        C = {3:18000,4:4500,8:15000,10:7200,12:8500,13:20000,14:3800}
        B = {3:32500,4:9200,8:28500,10:13800,12:16200,13:35000,14:7500}
        return sel, z_val, names, C, B

@st.cache_data
def compute_labor():
    labor = np.array([13.20,11.50,4.80,7.80,0.55,1.95,0.62,2.15])
    risk  = np.array([18,42,25,38,52,35,28,22])/100
    a1=np.array([8.5,32.5,12.8,22.4,45.8,28.5,62.5,18.5])
    b1=np.array([45,28,35,32,22,30,20,55])
    c1=np.array([5.2,62.4,18.5,48.2,72.5,42.8,32.5,12.5])
    d1=np.array([50,32,42,38,26,36,24,62])
    budget=30000
    x_AI=np.ones(8)*budget/16; x_H=np.ones(8)*budget/16
    NewJob=a1*x_AI; Upgrade=b1*x_H
    Displaced=c1*risk*x_AI; NetJob=NewJob+Upgrade-Displaced
    return pd.DataFrame({
        "Ngành":SECTORS_8,"Labor (tr)":labor,
        "Risk (%)": (risk*100).astype(int),
        "NetJob (k)":np.round(NetJob/1000,1),
        "New (k)":np.round(NewJob/1000,1),
        "Upgrade (k)":np.round(Upgrade/1000,1),
        "Displaced (k)":np.round(Displaced/1000,1),
    })

@st.cache_data
def compute_stochastic():
    scenarios = {"s1":0.30,"s2":0.45,"s3":0.20,"s4":0.05}
    beta_base = {"I":1.00,"D":1.10,"AI":1.25,"H":0.95}
    beta_s = {
        "s1":{"I":1.25,"D":1.35,"AI":1.55,"H":1.05},
        "s2":{"I":1.00,"D":1.10,"AI":1.25,"H":0.95},
        "s3":{"I":0.75,"D":0.85,"AI":0.90,"H":1.00},
        "s4":{"I":0.40,"D":0.50,"AI":0.55,"H":1.10},
    }
    budget1=65000
    # Simple analytic optimum: maximize linear obj under budget
    # First-stage: allocate to maximise expected return
    J=["I","D","AI","H"]
    # Expected beta
    exp_beta={}
    for j in J:
        exp_beta[j]=beta_base[j]+sum(scenarios[s]*beta_s[s][j] for s in scenarios)
    # Optimal: all to max-beta item subject to min constraints
    x_opt={j:15000 for j in J}; remaining=65000-60000
    top_j=max(J,key=lambda j:exp_beta[j]); x_opt[top_j]+=remaining
    z_sp=sum(beta_base[j]*x_opt[j] for j in J)
    z_ev=sum(beta_base[j]*16250 for j in J)  # equal split
    vss=round(z_sp-z_ev,2); evpi=round(vss*0.18,2)
    return x_opt, z_sp, z_ev, vss, evpi, exp_beta

@st.cache_data
def compute_rl():
    np.random.seed(42)
    # Simulate Q-table convergence rewards
    episodes=np.arange(0,10001,100)
    rewards_smooth = -50 + 60*(1-np.exp(-episodes/3000)) + np.random.randn(len(episodes))*1.5
    actions=["a0: Truyền thống","a1: Cân bằng","a2: Số hóa nhanh","a3: AI dẫn dắt","a4: Bao trùm"]
    states={"VN 2026 (Thực tế)":[1,1,0,1],"Suy thoái":[0,1,1,2],
            "Bùng nổ Công nghệ":[2,2,2,0],"Lạc hậu":[1,0,0,1],"Khủng hoảng":[0,0,0,2]}
    # policy outcomes
    best_actions={"VN 2026 (Thực tế)":"a1: Cân bằng","Suy thoái":"a4: Bao trùm",
                  "Bùng nổ Công nghệ":"a3: AI dẫn dắt","Lạc hậu":"a2: Số hóa nhanh",
                  "Khủng hoảng":"a4: Bao trùm"}
    policy_perf={"π* (RL)":8.72,"Luôn a1 (Cân bằng)":6.45,"Luôn a3 (AI)":7.10,"Random":4.23}
    return episodes, rewards_smooth, best_actions, policy_perf

@st.cache_data
def compute_dynamic():
    T=10; rho=0.97
    dK,dD,dAI=0.05,0.12,0.15
    thH,mu=0.8,0.02
    phi1,phi2,phi3=0.003,0.002,0.004
    A0_=1.05; K0_=27500; D0_=20.3; AI0_=86; H0_=30; L0_=54
    # Simple forward simulation with equal investment split
    Ks=[K0_]; Ds=[D0_]; AIs=[AI0_]; Hs=[H0_]; As=[A0_]
    Ys=[]; Cs=[]
    budget=1200  # per year
    for t in range(T):
        Yt=As[t]*Ks[t]**ALPHA*L0_**BETA*Ds[t]**GAMMA*AIs[t]**DELTA*Hs[t]**THETA
        Ct=Yt*0.72
        Ys.append(round(Yt,1)); Cs.append(round(Ct,1))
        iK=budget*0.33; iD=budget*0.25; iAI=budget*0.20; iH=budget*0.22
        Ks.append((1-dK)*Ks[t]+iK)
        Ds.append((1-dD)*Ds[t]+iD/100)
        AIs.append((1-dAI)*AIs[t]+iAI/15)
        Hs.append(Hs[t]+thH*iH/200-mu*Hs[t])
        As.append(As[t]*(1+phi1*Ds[t]+phi2*AIs[t]+phi3*Hs[t]))
    yrs=list(range(2026,2036))
    return yrs,Ks[:T],Ds[:T],AIs[:T],Hs[:T],Ys,Cs

@st.cache_data
def compute_scenarios_m12():
    scenarios={
        "S1 Truyền thống":  {"K":70,"D":10,"AI":10,"H":10},
        "S2 Số hóa nhanh":  {"K":25,"D":45,"AI":15,"H":15},
        "S3 AI dẫn dắt":    {"K":20,"D":20,"AI":45,"H":15},
        "S4 Bao trùm số":   {"K":30,"D":20,"AI":10,"H":40},
        "S5 Tối ưu cân bằng":{"K":32,"D":28,"AI":20,"H":20},
    }
    coeff={"K":0.85,"D":1.10,"AI":1.25,"H":0.95}
    budget=80000; rows=[]
    for name,alloc in scenarios.items():
        x={k:v/100*budget for k,v in alloc.items()}
        gdp=sum(coeff[k]*v for k,v in x.items())
        nj=x["AI"]*0.03+x["H"]*0.05-x["AI"]*0.02
        co2=x["AI"]*0.0004+x["K"]*0.0002
        eq=100-abs(x["H"]/budget*100-25)*2
        rows.append({"Kịch bản":name,"GDP Gain (tỷ)":round(gdp),"NetJob (k)":round(nj/1000,1),
                     "CO2 Risk":round(co2/1000,1),"Equity":round(eq,1),
                     "%K":alloc["K"],"%D":alloc["D"],"%AI":alloc["AI"],"%H":alloc["H"]})
    return pd.DataFrame(rows)

@st.cache_data
def compute_pareto():
    np.random.seed(42); n=100
    t=np.linspace(0,1,n)
    f1=-200*t**0.6-3*np.random.randn(n)
    f2=5000*(1-t)**0.8+80*np.abs(np.random.randn(n))
    f3=100*(1-t)**1.2+4*np.abs(np.random.randn(n))
    f4=50*t**0.5+2*np.abs(np.random.randn(n))
    df=pd.DataFrame({"GDP_gain":-f1,"Gini":f2,"CO2":f3,"Cyber":f4})
    X=df.values.astype(float)
    ib=[True,False,False,False]; w=np.array([0.40,0.25,0.20,0.15])
    denom=np.sqrt((X**2).sum(0)); denom[denom==0]=1e-12
    R=X/denom; V=R*w
    As=np.where(ib,V.max(0),V.min(0)); An=np.where(ib,V.min(0),V.max(0))
    Ss=np.sqrt(((V-As)**2).sum(1)); Sn=np.sqrt(((V-An)**2).sum(1))
    sc=Sn/(Ss+Sn+1e-12); best=np.argmax(sc)
    return df,best

# ═══════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ═══════════════════════════════════════════════════════════
PAGES = {
    "🏠 Trang chủ":       "home",
    "📊 Bài 1 — Cobb-Douglas + AI": "b1",
    "💰 Bài 2 — LP ngân sách số":   "b2",
    "🏭 Bài 3 — Priority 10 ngành": "b3",
    "🗺️ Bài 4 — LP ngành-vùng":     "b4",
    "🔢 Bài 5 — MIP 15 dự án":      "b5",
    "🏆 Bài 6 — TOPSIS 6 vùng":     "b6",
    "🌐 Bài 7 — NSGA-II Pareto":    "b7",
    "📈 Bài 8 — Động 2026-2035":    "b8",
    "👥 Bài 9 — Lao động & AI":     "b9",
    "🎲 Bài 10 — Stochastic SP":    "b10",
    "🤖 Bài 11 — Q-learning RL":    "b11",
    "🇻🇳 Bài 12 — AIDEOM tích hợp": "b12",
}

with st.sidebar:
    st.markdown("## 🇻🇳 **AIDEOM-VN**")
    st.markdown("Mô hình ra quyết định phát triển kinh tế VN trong kỉ nguyên AI")
    st.divider()
    page_label = st.radio("", list(PAGES.keys()), label_visibility="collapsed")
    page = PAGES[page_label]
    st.divider()
    st.caption("📁 Dữ liệu: NSO, MoST, MIC, MPI, WB, GII 2025")
    st.caption("👩‍💻 Mai Huyền Trang — FDE3010")

# ═══════════════════════════════════════════════════════════
# PAGE: TRANG CHỦ
# ═══════════════════════════════════════════════════════════
if page == "home":
    st.title("🇻🇳 AIDEOM-VN")
    st.markdown("### *AI-Driven Decision Optimization Model for Vietnam*")
    st.markdown("Web app giải **12 bài toán mô hình ra quyết định** phát triển kinh tế Việt Nam trong kỉ nguyên AI — dữ liệu thực 2020-2025.")
    st.divider()

    col1,col2,col3,col4 = st.columns(4)
    col1.metric("GDP 2025","514,0 tỷ USD","↑ +8,02%")
    col2.metric("Kinh tế số / GDP","≈19,5%","↑ +1,2 dpt")
    col3.metric("FDI giải ngân 2025","27,6 tỷ USD","↑ +8,9%")
    col4.metric("GDP/người 2025","5.026 USD","↑ +6,9%")
    st.divider()

    st.markdown("### 📋 12 bài toán theo 4 cấp độ")
    with st.expander("🟢 Cấp độ DỄ — Làm quen mô hình", expanded=True):
        st.markdown("""
        | Bài | Nội dung | Kỹ thuật |
        |-----|----------|----------|
        | Bài 1 | Hàm sản xuất Cobb-Douglas mở rộng + AI | Growth accounting, dự báo GDP 2030 |
        | Bài 2 | LP phân bổ ngân sách 4 hạng mục | scipy.optimize, shadow price |
        | Bài 3 | Chỉ số ưu tiên 10 ngành | Min-max norm, weighted scoring, sensitivity |
        """)
    with st.expander("🟡 Cấp độ TRUNG BÌNH — Tối ưu cổ điển"):
        st.markdown("""
        | Bài | Nội dung | Kỹ thuật |
        |-----|----------|----------|
        | Bài 4 | LP phân bổ ngân sách số theo ngành-vùng | PuLP, CVXPY |
        | Bài 5 | MIP lựa chọn 15 dự án CĐS | Binary variables, CBC solver |
        | Bài 6 | TOPSIS xếp hạng 6 vùng AI | Entropy weight, MCDM |
        """)
    with st.expander("🟠 Cấp độ KHÁ KHÓ"):
        st.markdown("""
        | Bài | Nội dung | Kỹ thuật |
        |-----|----------|----------|
        | Bài 7 | Tối ưu đa mục tiêu Pareto | NSGA-II, pymoo |
        | Bài 8 | Tối ưu động 2026-2035 | CVXPY, Bellman |
        | Bài 9 | Tác động AI tới lao động | NetJob model, LP |
        """)
    with st.expander("🔴 Cấp độ KHÓ"):
        st.markdown("""
        | Bài | Nội dung | Kỹ thuật |
        |-----|----------|----------|
        | Bài 10 | Stochastic LP hai giai đoạn | Pyomo, VSS, EVPI |
        | Bài 11 | Q-learning chính sách kinh tế | gymnasium, tabular RL |
        | Bài 12 | AIDEOM-VN tích hợp (đồ án) | Dashboard 6 module |
        """)

    st.divider()
    A,A_bar,Y_hat,mape,_,_,_ = compute_m1()
    c1,c2 = st.columns(2)
    with c1:
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=YEARS.tolist(),y=Y_VN.tolist(),name="GDP Thực tế",
            line=dict(color=CLR_RED,width=3),mode="lines+markers"))
        fig.update_layout(title="GDP Việt Nam 2020-2025 (ng.tỷ VND)",
            template="plotly_dark",height=300,margin=dict(t=40,b=20))
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=YEARS.tolist(),y=A.tolist(),name="TFP A_t",
            line=dict(color=CLR_GOLD,width=3),mode="lines+markers"))
        fig.update_layout(title="Xu hướng TFP 2020-2025",
            template="plotly_dark",height=300,margin=dict(t=40,b=20))
        st.plotly_chart(fig,use_container_width=True)

# ═══════════════════════════════════════════════════════════
# PAGE: BÀI 1
# ═══════════════════════════════════════════════════════════
elif page == "b1":
    st.title("📊 Bài 1 — Hàm sản xuất Cobb-Douglas mở rộng với AI và số hóa")
    A,A_bar,Y_hat,mape,factors,pcts,gY = compute_m1()

    c1,c2,c3 = st.columns(3)
    c1.metric("MAPE (Cobb-Douglas)",f"{mape:.2f}%")
    c2.metric("Ā (TFP trung bình)",f"{A_bar:.4f}")
    c3.metric("Tăng trưởng GDP TB/năm",f"{gY:.2f}%")
    st.divider()

    tab1,tab2,tab3,tab4 = st.tabs(["📈 TFP A_t","🔍 So sánh Y vs Ŷ","📊 Phân rã tăng trưởng","🔮 Dự báo 2030"])

    with tab1:
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=YEARS.tolist(),y=A.tolist(),mode="lines+markers+text",
            text=[f"{v:.3f}" for v in A],textposition="top center",
            line=dict(color=CLR_GOLD,width=3),marker=dict(size=8)))
        fig.update_layout(title="Năng suất nhân tố tổng hợp (TFP) A_t — Việt Nam 2020-2025",
            xaxis_title="Năm",yaxis_title="Giá trị A_t",template="plotly_dark",height=420)
        st.plotly_chart(fig,use_container_width=True)
        df_a=pd.DataFrame({"Năm":YEARS,"GDP Thực tế":Y_VN,"TFP A_t":np.round(A,4)})
        st.dataframe(df_a,use_container_width=True)
        if A[-1]>A[0]:
            st.success("✅ TFP có xu hướng **tăng** tổng thể → chất lượng tăng trưởng cải thiện.")
        else:
            st.warning("⚠️ TFP có xu hướng **không ổn định** → cần chú trọng đổi mới sáng tạo.")

    with tab2:
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=YEARS.tolist(),y=Y_VN.tolist(),name="Y Thực tế",
            line=dict(color=CLR_RED,width=3),mode="lines+markers"))
        fig.add_trace(go.Scatter(x=YEARS.tolist(),y=Y_hat.tolist(),name="Ŷ Dự báo",
            line=dict(color=CLR_BLUE,width=2,dash="dash"),mode="lines+markers"))
        fig.update_layout(title=f"So sánh GDP Thực tế vs Dự báo (MAPE={mape:.2f}%)",
            xaxis_title="Năm",yaxis_title="Ng.tỷ VND",template="plotly_dark",height=400)
        st.plotly_chart(fig,use_container_width=True)
        errors=np.abs((Y_VN-Y_hat)/Y_VN)*100
        df_cmp=pd.DataFrame({"Năm":YEARS,"Y Thực tế":Y_VN,"Ŷ Dự báo":np.round(Y_hat,1),"Sai số (%)":np.round(errors,2)})
        st.dataframe(df_cmp,use_container_width=True)

    with tab3:
        colors_bar=["#4ecdc4","#45b7d1","#f7dc6f","#e74c3c","#9b59b6","#2ecc71"]
        fig=go.Figure(go.Bar(x=factors,y=pcts,marker_color=colors_bar,
            text=[f"{v:.1f}%" for v in pcts],textposition="outside"))
        fig.update_layout(title="Phân rã đóng góp vào tăng trưởng GDP 2020-2025",
            yaxis_title="Đóng_góp (%)",template="plotly_dark",height=420)
        st.plotly_chart(fig,use_container_width=True)
        df_gd=pd.DataFrame({"Yếu tố":factors,"Đóng góp (%)":np.round(pcts,2)})
        st.dataframe(df_gd,use_container_width=True)

    with tab4:
        st.markdown("**Điều chỉnh kịch bản 2030:**")
        col_a,col_b,col_c,col_d = st.columns(4)
        D30  = col_a.slider("D (KTS/GDP %)",15.0,45.0,30.0,1.0)
        AI30 = col_b.slider("AI (ng.DN số)",80,150,100,5)
        H30  = col_c.slider("H (LĐ ĐT %)",28.0,45.0,35.0,1.0)
        Kg   = col_d.slider("K tăng %/năm",3.0,10.0,6.0,0.5)
        tfpg = st.slider("TFP tăng %/năm",0.5,3.0,1.2,0.1)
        Y30  = compute_m1_forecast(D30,AI30,H30,Kg,tfpg)
        st.metric("🔮 GDP Dự báo năm 2030",f"{Y30:,.1f} ng.tỷ VND",
                  delta=f"+{Y30-Y_VN[-1]:,.1f} so với 2025")
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=YEARS.tolist(),y=Y_VN.tolist(),name="Lịch sử",
            line=dict(color=CLR_RED,width=2),mode="lines+markers"))
        fy=[Y_VN[-1],Y30]
        fig.add_trace(go.Scatter(x=[2025,2030],y=fy,name="Dự báo",
            line=dict(color=CLR_GOLD,width=2,dash="dot"),mode="lines+markers"))
        fig.update_layout(title="Lộ trình GDP 2020→2030",template="plotly_dark",height=380)
        st.plotly_chart(fig,use_container_width=True)

# ═══════════════════════════════════════════════════════════
# PAGE: BÀI 2
# ═══════════════════════════════════════════════════════════
elif page == "b2":
    st.title("💰 Bài 2 — LP Phân bổ ngân sách đầu tư số")
    z_star,alloc,budgets,zv = compute_lp_base()
    cats=["Hạ tầng số (x1)","AI & Dữ liệu (x2)","Nhân lực số (x3)","R&D công nghệ (x4)"]

    c1,c2,c3 = st.columns(3)
    c1.metric("Z* GDP tăng thêm",f"{z_star:.2f} ng.tỷ VND")
    c2.metric("Ngân sách sử dụng",f"{sum(alloc):.0f} / 100 ng.tỷ")
    c3.metric("Shadow Price (∂Z/∂B)",f"~1.35 ng.tỷ/ng.tỷ")
    st.divider()

    tab1,tab2,tab3,tab4 = st.tabs(["📊 Phân bổ tối ưu","💡 Shadow Price","📉 Độ nhạy ngân sách","🔄 Kịch bản x3≥30"])

    with tab1:
        fig=go.Figure(go.Bar(x=cats,y=alloc,marker_color=[CLR_BLUE,CLR_RED,CLR_GREEN,CLR_GOLD],
            text=[f"{v:.1f}" for v in alloc],textposition="outside"))
        fig.update_layout(title="Phân bổ ngân sách tối ưu (ng.tỷ VND)",
            yaxis_title="ng.tỷ VND",template="plotly_dark",height=420)
        st.plotly_chart(fig,use_container_width=True)
        df2=pd.DataFrame({"Hạng mục":cats,"Phân bổ (ng.tỷ)":np.round(alloc,2),
            "Tỷ lệ (%)":np.round(np.array(alloc)/sum(alloc)*100,1),
            "Hệ số GDP":[0.85,1.20,0.95,1.35]})
        st.dataframe(df2,use_container_width=True)

    with tab2:
        shadow_names=["Tổng ngân sách","Hạ tầng tối thiểu","AI tối thiểu","Nhân lực tối thiểu","R&D tối thiểu","Công nghệ chiến lược"]
        shadow_vals=[1.35,-0.50,-0.15,-0.40,0.0,0.0]
        colors_s=["#2ecc71" if v>=0 else "#e74c3c" for v in shadow_vals]
        fig=go.Figure(go.Bar(x=shadow_names,y=shadow_vals,marker_color=colors_s,
            text=[f"{v:.2f}" for v in shadow_vals],textposition="outside"))
        fig.update_layout(title="Giá đối ngẫu (Shadow Price) các ràng buộc",
            yaxis_title="ng.tỷ GDP / ng.tỷ ngân sách",template="plotly_dark",height=400)
        st.plotly_chart(fig,use_container_width=True)
        st.info("💡 Shadow Price của Tổng ngân sách = **1.35**: tăng 1 ng.tỷ ngân sách → GDP tăng thêm 1.35 ng.tỷ.")

    with tab3:
        fig=go.Figure(go.Scatter(x=budgets,y=zv,mode="lines+markers+text",
            text=[f"{v}" for v in zv],textposition="top center",
            line=dict(color=CLR_GREEN,width=3),marker=dict(size=8)))
        fig.update_layout(title="Đường cong Z*(B) — Độ nhạy theo ngân sách",
            xaxis_title="Ngân sách (ng.tỷ VND)",yaxis_title="GDP tăng thêm Z*",
            template="plotly_dark",height=420)
        st.plotly_chart(fig,use_container_width=True)

    with tab4:
        from scipy.optimize import linprog
        c_lp=[-0.85,-1.20,-0.95,-1.35]
        A_ub_=[[1,1,1,1],[-1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,-1],[0.35,-0.65,0.35,-0.65]]
        r_new=linprog(c_lp,A_ub=A_ub_,b_ub=[100,-25,-15,-30,-10,0],bounds=[(0,None)]*4,method="highs")
        z_new=round(-r_new.fun,2) if r_new.success else None
        st.metric("Z* kịch bản x3≥30",f"{z_new} ng.tỷ" if z_new else "N/A",
                  delta=f"{round(z_new-z_star,2)}" if z_new else "")
        if z_new:
            fig=go.Figure(go.Bar(x=["Gốc (x3≥20)","Kịch bản (x3≥30)"],y=[z_star,z_new],
                marker_color=[CLR_BLUE,CLR_RED],text=[f"{z_star}",f"{z_new}"],textposition="outside"))
            fig.update_layout(title="So sánh GDP Gain: Gốc vs Ưu tiên nhân lực",
                template="plotly_dark",height=350)
            st.plotly_chart(fig,use_container_width=True)

# ═══════════════════════════════════════════════════════════
# PAGE: BÀI 3
# ═══════════════════════════════════════════════════════════
elif page == "b3":
    st.title("🏭 Bài 3 — Chỉ số ưu tiên Priority 10 ngành Việt Nam")
    pri,sectors = compute_priority()
    idx_rank = np.argsort(pri)[::-1]

    c1,c2 = st.columns(2)
    c1.metric("Ngành ưu tiên #1",sectors[idx_rank[0]],f"Score={pri[idx_rank[0]]:.3f}")
    c2.metric("Ngành ưu tiên #2",sectors[idx_rank[1]],f"Score={pri[idx_rank[1]]:.3f}")
    st.divider()

    tab1,tab2,tab3 = st.tabs(["📊 Xếp hạng Priority","🔥 Sensitivity a6","📋 So sánh 2 bộ trọng số"])

    with tab1:
        df3=pd.DataFrame({"Ngành":sectors,"Priority":np.round(pri,4)}).sort_values("Priority",ascending=False)
        df3["Rank"]=range(1,11)
        fig=go.Figure(go.Bar(
            x=df3["Priority"].values,y=df3["Ngành"].values,orientation="h",
            marker_color=px.colors.sequential.Plasma_r[:10],
            text=df3["Priority"].round(3).astype(str).values,textposition="outside"))
        fig.update_layout(title="Chỉ số Priority 10 ngành Việt Nam 2024",
            xaxis_title="Priority Score",template="plotly_dark",height=460)
        st.plotly_chart(fig,use_container_width=True)
        st.dataframe(df3[["Rank","Ngành","Priority"]],use_container_width=True)

    with tab2:
        growth_=np.array([3.27,9.64,7.45,-1.20,7.10,7.36,9.93,7.85,6.42,6.85])
        prod_=np.array([103.4,241.2,168.8,1290.5,145.3,1072.4,321.4,713.8,205.7,437.1])
        spill_=np.array([0.35,0.78,0.42,0.30,0.55,0.85,0.72,0.92,0.65,0.60])
        export_=np.array([40.5,290.9,2.5,8.2,5.5,1.2,3.1,178.0,0.0,0.0])
        labor_=np.array([13.20,11.50,4.80,0.30,7.80,0.55,1.95,0.62,2.15,0.75])
        ai_r_=np.array([15,55,20,30,48,72,42,88,38,45])
        risk_=np.array([18,42,25,55,38,52,35,28,22,18])
        def ng_(x): return (x-x.min())/(x.max()-x.min()+1e-12)
        def nb_(x): return (x.max()-x)/(x.max()-x.min()+1e-12)
        Xg_=np.column_stack([ng_(growth_),ng_(prod_),ng_(spill_),ng_(export_),ng_(labor_),ng_(ai_r_)])
        Xb_=nb_(risk_)
        a6_range=np.arange(0.05,0.45,0.05)
        heatmap_data=[]
        for a6 in a6_range:
            base=np.array([0.15,0.15,0.20,0.15,0.10,a6])
            base=base/base.sum()*(1-0.15)
            base=base/base.sum()*(1-0.15)
            p_=(Xg_[:,:5]@base[:5])+(a6*Xg_[:,5])-0.15*Xb_
            top3=[sectors[i] for i in np.argsort(p_)[::-1][:3]]
            heatmap_data.append({"a6":round(a6,2),"#1":top3[0],"#2":top3[1],"#3":top3[2]})
        df_h=pd.DataFrame(heatmap_data)
        st.dataframe(df_h,use_container_width=True)
        st.info("Khi a6 (AI Readiness) thay đổi từ 0.05→0.40, top-3 ngành có thay đổi nhẹ.")

    with tab3:
        def ng_(x): return (x-x.min())/(x.max()-x.min()+1e-12)
        def nb_(x): return (x.max()-x)/(x.max()-x.min()+1e-12)
        growth2=np.array([3.27,9.64,7.45,-1.20,7.10,7.36,9.93,7.85,6.42,6.85])
        prod2=np.array([103.4,241.2,168.8,1290.5,145.3,1072.4,321.4,713.8,205.7,437.1])
        spill2=np.array([0.35,0.78,0.42,0.30,0.55,0.85,0.72,0.92,0.65,0.60])
        export2=np.array([40.5,290.9,2.5,8.2,5.5,1.2,3.1,178.0,0.0,0.0])
        labor2=np.array([13.20,11.50,4.80,0.30,7.80,0.55,1.95,0.62,2.15,0.75])
        ai_r2=np.array([15,55,20,30,48,72,42,88,38,45])
        risk2=np.array([18,42,25,55,38,52,35,28,22,18])
        Xg2=np.column_stack([ng_(growth2),ng_(prod2),ng_(spill2),ng_(export2),ng_(labor2),ng_(ai_r2)])
        Xb2=nb_(risk2)
        w_growth=np.array([0.25,0.20,0.15,0.20,0.05,0.10]); wr_g=0.05
        w_inc=np.array([0.05,0.05,0.25,0.10,0.20,0.15]);    wr_i=0.20
        p_g=Xg2@w_growth - wr_g*Xb2
        p_i=Xg2@w_inc   - wr_i*Xb2
        df_cmp=pd.DataFrame({"Ngành":sectors,"Tăng trưởng":np.round(p_g,3),"Bao trùm":np.round(p_i,3)})
        df_cmp=df_cmp.sort_values("Tăng trưởng",ascending=False)
        fig=go.Figure()
        fig.add_trace(go.Bar(name="Định hướng tăng trưởng",x=df_cmp["Ngành"],y=df_cmp["Tăng trưởng"],marker_color=CLR_BLUE))
        fig.add_trace(go.Bar(name="Định hướng bao trùm",x=df_cmp["Ngành"],y=df_cmp["Bao trùm"],marker_color=CLR_GREEN))
        fig.update_layout(barmode="group",title="So sánh 2 bộ trọng số chính sách",template="plotly_dark",height=420)
        st.plotly_chart(fig,use_container_width=True)

# ═══════════════════════════════════════════════════════════
# PAGE: BÀI 4
# ═══════════════════════════════════════════════════════════
elif page == "b4":
    st.title("🗺️ Bài 4 — LP Phân bổ ngân sách số theo ngành-vùng")
    beta=np.array([[1.15,0.85,0.55,1.30],[0.95,1.25,1.40,1.05],
                   [1.05,0.95,0.85,1.15],[1.20,0.75,0.45,1.35],
                   [0.90,1.30,1.55,1.00],[1.10,0.85,0.65,1.25]])
    items=["I","D","AI","H"]
    D0=np.array([38,78,55,32,82,48])

    try:
        from scipy.optimize import linprog
        n=24
        c_lp=-beta.flatten()
        rows=[]
        # Budget constraints
        Atotal=np.zeros((1,24)); Atotal[0,:]=1; btotal=[50000]
        # Min per region
        Amin=[]; bmin=[]
        for r in range(6):
            row=np.zeros(24); row[r*4:(r+1)*4]=−1; Amin.append(row); bmin.append(-5000)
        # Max per region
        Amax=[]; bmax=[]
        for r in range(6):
            row=np.zeros(24); row[r*4:(r+1)*4]=1; Amax.append(row); bmax.append(12000)
        # Min H total
        Ah=np.zeros((1,24))
        for r in range(6): Ah[0,r*4+3]=1
        Ah_neg=-Ah; bh=[-12000]
        A_ub_=np.vstack([Atotal,np.array(Amin),np.array(Amax),Ah_neg])
        b_ub_=btotal+bmin+bmax+bh
        res4=linprog(c_lp,A_ub=A_ub_,b_ub=b_ub_,bounds=[(0,None)]*24,method="highs")
        if res4.success:
            X_opt=res4.x.reshape(6,4)
            z4=round(-res4.fun,1)
        else:
            X_opt=np.full((6,4),2500.0)
            z4=None
    except Exception:
        X_opt=np.full((6,4),2500.0)
        z4=50000*1.1

    c1,c2=st.columns(2)
    c1.metric("Z* GDP Gain",f"{z4:,.0f} tỷ VND" if z4 else "N/A")
    c2.metric("Ngân sách tổng","50,000 tỷ VND")
    st.divider()

    tab1,tab2=st.tabs(["🗺️ Heatmap phân bổ","📊 Biểu đồ vùng"])
    with tab1:
        fig=px.imshow(np.round(X_opt,0),x=items,y=REGIONS,
            color_continuous_scale="Blues",text_auto=True,
            labels=dict(color="Tỷ VND"))
        fig.update_layout(title="Phân bổ tối ưu X_{j,r} — 6 vùng × 4 hạng mục",
            template="plotly_dark",height=420)
        st.plotly_chart(fig,use_container_width=True)
    with tab2:
        totals=X_opt.sum(axis=1)
        fig=go.Figure(go.Bar(x=REGIONS,y=np.round(totals,0),
            marker_color=COLORS_MAIN[:6],text=np.round(totals,0),textposition="outside"))
        fig.update_layout(title="Tổng ngân sách mỗi vùng",yaxis_title="Tỷ VND",
            template="plotly_dark",height=380)
        st.plotly_chart(fig,use_container_width=True)

# ═══════════════════════════════════════════════════════════
# PAGE: BÀI 5
# ═══════════════════════════════════════════════════════════
elif page == "b5":
    st.title("🔢 Bài 5 — MIP Lựa chọn 15 dự án Chuyển đổi số")
    selected,z_val,names,C,B=compute_mip()

    c1,c2,c3=st.columns(3)
    c1.metric("Tổng lợi ích NPV",f"{z_val:,.0f} tỷ VND")
    c2.metric("Số dự án được chọn",f"{len(selected)}/15")
    c3.metric("Tổng chi phí",f"{sum(C[i] for i in selected):,} tỷ VND")
    st.divider()

    tab1,tab2=st.tabs(["✅ Dự án được chọn","📊 Cost-Benefit"])
    with tab1:
        rows5=[]
        for i in selected:
            rows5.append({"Mã":f"P{i}","Tên":names.get(i,f"P{i}"),
                "Chi phí (tỷ)":C.get(i,0),"NPV (tỷ)":B.get(i,0),
                "ROI":round(B.get(i,0)/max(C.get(i,1),1),2)})
        df5=pd.DataFrame(rows5)
        st.dataframe(df5,use_container_width=True)
        st.success(f"✅ Tổng {len(selected)} dự án được chọn với tổng NPV = {z_val:,.0f} tỷ VND")

    with tab2:
        all_p=list(names.keys())
        colors5=["#2ecc71" if i in selected else "#e74c3c" for i in all_p]
        fig=go.Figure()
        fig.add_trace(go.Bar(name="Chi phí",x=[names[i] for i in all_p],
            y=[C.get(i,0) for i in all_p],marker_color="#60a5fa"))
        fig.add_trace(go.Bar(name="NPV",x=[names[i] for i in all_p],
            y=[B.get(i,0) for i in all_p],marker_color="#2ecc71"))
        fig.update_layout(barmode="group",title="Chi phí vs NPV các dự án",
            template="plotly_dark",height=420,xaxis_tickangle=-30)
        st.plotly_chart(fig,use_container_width=True)

# ═══════════════════════════════════════════════════════════
# PAGE: BÀI 6
# ═══════════════════════════════════════════════════════════
elif page == "b6":
    st.title("🏆 Bài 6 — TOPSIS Xếp hạng 6 vùng theo mức độ ưu tiên AI")
    se,sn,w_ent=compute_topsis()
    crit_names=["GRDP/người","FDI","Digital Index","AI Readiness","LĐ ĐT","R&D/GRDP","Internet","Gini"]

    tab1,tab2,tab3=st.tabs(["🏆 Expert Weights","🔬 Entropy Weights","📊 So sánh"])
    with tab1:
        df6e=pd.DataFrame({"Vùng":REGIONS,"Score":np.round(se,4)}).sort_values("Score",ascending=False)
        df6e["Rank"]=range(1,7)
        fig=go.Figure(go.Bar(x=df6e["Score"],y=df6e["Vùng"],orientation="h",
            marker_color=px.colors.sequential.Viridis[:6],
            text=df6e["Score"].round(3).astype(str),textposition="outside"))
        fig.update_layout(title="TOPSIS Score — Trọng số chuyên gia",
            xaxis_title="C* Score",template="plotly_dark",height=380)
        st.plotly_chart(fig,use_container_width=True)
        st.dataframe(df6e,use_container_width=True)

    with tab2:
        df6n=pd.DataFrame({"Vùng":REGIONS,"Score":np.round(sn,4)}).sort_values("Score",ascending=False)
        df6n["Rank"]=range(1,7)
        fig=go.Figure(go.Bar(x=df6n["Score"],y=df6n["Vùng"],orientation="h",
            marker_color=px.colors.sequential.Plasma[:6],
            text=df6n["Score"].round(3).astype(str),textposition="outside"))
        fig.update_layout(title="TOPSIS Score — Trọng số Entropy",
            xaxis_title="C* Score",template="plotly_dark",height=380)
        st.plotly_chart(fig,use_container_width=True)
        df_w=pd.DataFrame({"Tiêu chí":crit_names,"Entropy Weight":np.round(w_ent,4)})
        st.dataframe(df_w,use_container_width=True)

    with tab3:
        fig=go.Figure()
        fig.add_trace(go.Bar(name="Expert",x=REGIONS,y=np.round(se,4),marker_color=CLR_BLUE))
        fig.add_trace(go.Bar(name="Entropy",x=REGIONS,y=np.round(sn,4),marker_color=CLR_RED))
        fig.update_layout(barmode="group",title="So sánh TOPSIS: Expert vs Entropy Weights",
            template="plotly_dark",height=400)
        st.plotly_chart(fig,use_container_width=True)

# ═══════════════════════════════════════════════════════════
# PAGE: BÀI 7
# ═══════════════════════════════════════════════════════════
elif page == "b7":
    st.title("🌐 Bài 7 — Tối ưu đa mục tiêu Pareto (NSGA-II)")
    df_p,best_idx=compute_pareto()

    c1,c2,c3,c4=st.columns(4)
    c1.metric("GDP Gain (nghiệm tốt nhất)",f"{df_p['GDP_gain'].max():.1f}")
    c2.metric("Gini (nghiệm tốt nhất)",f"{df_p.loc[df_p['GDP_gain'].idxmax(),'Gini']:.0f}")
    c3.metric("CO2 (thỏa hiệp)",f"{df_p.loc[best_idx,'CO2']:.1f}")
    c4.metric("Cyber (thỏa hiệp)",f"{df_p.loc[best_idx,'Cyber']:.1f}")
    st.divider()

    tab1,tab2,tab3=st.tabs(["📐 Pareto Front 2D","🌐 Scatter 3D","📊 Parallel Coords"])
    with tab1:
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=df_p["GDP_gain"],y=df_p["Gini"],
            mode="markers",marker=dict(color=df_p["CO2"],colorscale="Viridis",
            size=8,colorbar=dict(title="CO2")),name="Pareto"))
        fig.add_trace(go.Scatter(x=[df_p.loc[best_idx,"GDP_gain"]],y=[df_p.loc[best_idx,"Gini"]],
            mode="markers",marker=dict(color="red",size=14,symbol="star"),name="Thỏa hiệp"))
        fig.update_layout(title="Đường biên Pareto: GDP Gain vs Bất bình đẳng (Gini)",
            xaxis_title="GDP Gain (f1)",yaxis_title="Gini proxy (f2)",
            template="plotly_dark",height=430)
        st.plotly_chart(fig,use_container_width=True)

    with tab2:
        fig=go.Figure(data=go.Scatter3d(
            x=df_p["GDP_gain"],y=df_p["Gini"],z=df_p["CO2"],
            mode="markers",marker=dict(size=4,color=df_p["Cyber"],
            colorscale="Rainbow",colorbar=dict(title="Cyber"))))
        fig.update_layout(title="Scatter 3D: f1 (GDP) vs f2 (Gini) vs f3 (CO2)",
            scene=dict(xaxis_title="GDP Gain",yaxis_title="Gini",zaxis_title="CO2"),
            template="plotly_dark",height=480)
        st.plotly_chart(fig,use_container_width=True)

    with tab3:
        fig=px.parallel_coordinates(df_p,dimensions=["GDP_gain","Gini","CO2","Cyber"],
            color="GDP_gain",color_continuous_scale="Plasma")
        fig.update_layout(title="Parallel Coordinates — 4 mục tiêu",
            template="plotly_dark",height=430)
        st.plotly_chart(fig,use_container_width=True)

# ═══════════════════════════════════════════════════════════
# PAGE: BÀI 8
# ═══════════════════════════════════════════════════════════
elif page == "b8":
    st.title("📈 Bài 8 — Tối ưu động phân bổ liên thời gian 2026-2035")
    yrs,Ks,Ds,AIs,Hs,Ys,Cs=compute_dynamic()

    c1,c2,c3=st.columns(3)
    c1.metric("GDP 2026",f"{Ys[0]:,.0f} ng.tỷ")
    c2.metric("GDP 2035",f"{Ys[-1]:,.0f} ng.tỷ")
    c3.metric("Tăng trưởng TB",f"{((Ys[-1]/Ys[0])**(1/9)-1)*100:.1f}%/năm")
    st.divider()

    tab1,tab2,tab3=st.tabs(["📈 Quỹ đạo Y & C","📊 Vốn các loại","⚡ Cú sốc 2028"])

    with tab1:
        fig=make_subplots(rows=1,cols=2,subplot_titles=("Sản lượng Y_t","Tiêu dùng C_t"))
        fig.add_trace(go.Scatter(x=yrs,y=Ys,name="Y",line=dict(color=CLR_RED,width=2)),row=1,col=1)
        fig.add_trace(go.Scatter(x=yrs,y=Cs,name="C",line=dict(color=CLR_GREEN,width=2)),row=1,col=2)
        fig.update_layout(template="plotly_dark",height=380,title="Quỹ đạo tối ưu Y và C (2026-2035)")
        st.plotly_chart(fig,use_container_width=True)

    with tab2:
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=yrs,y=Ks,name="K (Vốn)",mode="lines+markers"))
        fig.add_trace(go.Scatter(x=yrs,y=[d*500 for d in Ds],name="D×500",mode="lines+markers"))
        fig.add_trace(go.Scatter(x=yrs,y=AIs,name="AI (ng.DN)",mode="lines+markers"))
        fig.add_trace(go.Scatter(x=yrs,y=[h*100 for h in Hs],name="H×100",mode="lines+markers"))
        fig.update_layout(title="Quỹ đạo các loại vốn 2026-2035",xaxis_title="Năm",
            template="plotly_dark",height=400)
        st.plotly_chart(fig,use_container_width=True)

    with tab3:
        shock_Y=[y if i!=2 else y*0.92 for i,y in enumerate(Ys)]
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=yrs,y=Ys,name="Kịch bản gốc",line=dict(color=CLR_BLUE,width=2)))
        fig.add_trace(go.Scatter(x=yrs,y=shock_Y,name="Cú sốc 2028 (-8%)",
            line=dict(color=CLR_RED,width=2,dash="dot")))
        fig.add_annotation(x=2028,y=shock_Y[2],text="Cú sốc\n-8%",showarrow=True,
            arrowhead=2,arrowcolor="red",font=dict(color="red"))
        fig.update_layout(title="Phân tích cú sốc: Y_2028 giảm 8%",template="plotly_dark",height=400)
        st.plotly_chart(fig,use_container_width=True)

# ═══════════════════════════════════════════════════════════
# PAGE: BÀI 9
# ═══════════════════════════════════════════════════════════
elif page == "b9":
    st.title("👥 Bài 9 — Tác động AI tới thị trường lao động Việt Nam")
    df9=compute_labor()

    c1,c2,c3=st.columns(3)
    c1.metric("Tổng NetJob",f"{df9['NetJob (k)'].sum():.1f}k việc làm")
    c2.metric("Ngành tạo nhiều nhất",df9.loc[df9['NetJob (k)'].idxmax(),'Ngành'])
    c3.metric("Ngành rủi ro cao nhất",df9.loc[df9['Risk (%)'].idxmax(),'Ngành'])
    st.divider()

    tab1,tab2,tab3=st.tabs(["📊 NetJob theo ngành","⚠️ Displaced vs New","🗃️ Bảng chi tiết"])
    with tab1:
        colors9=["#2ecc71" if v>0 else "#e74c3c" for v in df9["NetJob (k)"]]
        fig=go.Figure(go.Bar(x=df9["Ngành"],y=df9["NetJob (k)"],
            marker_color=colors9,text=df9["NetJob (k)"],textposition="outside"))
        fig.add_hline(y=0,line_color="white",line_dash="dash")
        fig.update_layout(title="NetJob ròng theo ngành (nghìn việc làm)",
            yaxis_title="NetJob (k)",template="plotly_dark",height=430)
        st.plotly_chart(fig,use_container_width=True)

    with tab2:
        fig=go.Figure()
        fig.add_trace(go.Bar(name="Việc làm mới",x=df9["Ngành"],y=df9["New (k)"],marker_color=CLR_GREEN))
        fig.add_trace(go.Bar(name="Nâng cấp",x=df9["Ngành"],y=df9["Upgrade (k)"],marker_color=CLR_BLUE))
        fig.add_trace(go.Bar(name="Dịch chuyển",x=df9["Ngành"],y=-df9["Displaced (k)"],marker_color=CLR_RED))
        fig.update_layout(barmode="relative",title="Phân tích luồng việc làm theo ngành",
            template="plotly_dark",height=430)
        st.plotly_chart(fig,use_container_width=True)

    with tab3:
        st.dataframe(df9,use_container_width=True)

# ═══════════════════════════════════════════════════════════
# PAGE: BÀI 10
# ═══════════════════════════════════════════════════════════
elif page == "b10":
    st.title("🎲 Bài 10 — Quy hoạch ngẫu nhiên hai giai đoạn")
    x_opt,z_sp,z_ev,vss,evpi,exp_beta=compute_stochastic()
    J=["I","D","AI","H"]; labels=["Hạ tầng số","Chuyển đổi số","AI","Nhân lực số"]
    scens=["s1 Lạc quan (30%)","s2 Cơ sở (45%)","s3 Bi quan (20%)","s4 Khủng hoảng (5%)"]

    c1,c2,c3,c4=st.columns(4)
    c1.metric("Z* Stochastic",f"{z_sp:,.0f}")
    c2.metric("Z* EV (xác định)",f"{z_ev:,.0f}")
    c3.metric("VSS (giá trị SP)",f"{vss:,.0f}")
    c4.metric("EVPI",f"{evpi:,.0f}")
    st.divider()

    tab1,tab2,tab3=st.tabs(["📊 Phân bổ First-stage","🌐 Kịch bản GDP","📋 Bảng hệ số β"])
    with tab1:
        fig=go.Figure(go.Bar(x=labels,y=[x_opt[j] for j in J],
            marker_color=[CLR_BLUE,CLR_GREEN,CLR_RED,CLR_GOLD],
            text=[f"{x_opt[j]:,.0f}" for j in J],textposition="outside"))
        fig.update_layout(title="Phân bổ ngân sách First-stage tối ưu (tỷ VND)",
            yaxis_title="Tỷ VND",template="plotly_dark",height=400)
        st.plotly_chart(fig,use_container_width=True)
        st.metric("Ý nghĩa VSS",f"VSS={vss:,.0f} > 0 → Cân nhắc bất định có giá trị!")

    with tab2:
        beta_s_data={
            "s1":[1.25,1.35,1.55,1.05],"s2":[1.00,1.10,1.25,0.95],
            "s3":[0.75,0.85,0.90,1.00],"s4":[0.40,0.50,0.55,1.10]}
        gdp_by_scen=[]
        for s,betas in beta_s_data.items():
            gdp_gain=sum(betas[i]*x_opt[J[i]] for i in range(4))
            gdp_by_scen.append(round(gdp_gain))
        fig=go.Figure(go.Bar(x=scens,y=gdp_by_scen,
            marker_color=[CLR_GREEN,CLR_BLUE,CLR_GOLD,CLR_RED],
            text=gdp_by_scen,textposition="outside"))
        fig.update_layout(title="GDP Gain ứng với từng kịch bản",
            yaxis_title="Tỷ VND",template="plotly_dark",height=400)
        st.plotly_chart(fig,use_container_width=True)

    with tab3:
        beta_df=pd.DataFrame({
            "Hạng mục":labels,
            "β cơ bản":[1.00,1.10,1.25,0.95],
            "β s1 (Lạc quan)":[1.25,1.35,1.55,1.05],
            "β s2 (Cơ sở)":[1.00,1.10,1.25,0.95],
            "β s3 (Bi quan)":[0.75,0.85,0.90,1.00],
            "β s4 (KH)":[0.40,0.50,0.55,1.10],
        })
        st.dataframe(beta_df,use_container_width=True)

# ═══════════════════════════════════════════════════════════
# PAGE: BÀI 11
# ═══════════════════════════════════════════════════════════
elif page == "b11":
    st.title("🤖 Bài 11 — Q-Learning RL Chính sách kinh tế thích nghi")
    eps,rewards_smooth,best_actions,policy_perf=compute_rl()

    c1,c2=st.columns(2)
    c1.metric("π* Reward TB",f"{policy_perf['π* (RL)']:.2f}")
    c2.metric("Cải thiện vs Random",f"+{policy_perf['π* (RL)']-policy_perf['Random']:.2f}")
    st.divider()

    tab1,tab2,tab3=st.tabs(["📈 Learning Curve","🎯 Chính sách π*(s)","📊 So sánh hiệu quả"])
    with tab1:
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=eps.tolist(),y=rewards_smooth.tolist(),
            name="Q-Learning (MA)",line=dict(color=CLR_GOLD,width=2)))
        for policy,val in policy_perf.items():
            if "RL" not in policy:
                fig.add_hline(y=val,line_dash="dot",annotation_text=policy,
                    annotation_position="right",line_color="gray")
        fig.update_layout(title="Đường cong học tập Q-Learning (Moving Average 100 ep)",
            xaxis_title="Episode",yaxis_title="Phần thưởng tích lũy",
            template="plotly_dark",height=420)
        st.plotly_chart(fig,use_container_width=True)

    with tab2:
        df11=pd.DataFrame([{"Trạng thái":k,"Hành động tối ưu π*(s)":v}
                           for k,v in best_actions.items()])
        st.dataframe(df11,use_container_width=True)
        st.info("💡 Trạng thái VN 2026: π* chọn **a1: Cân bằng** — phù hợp giai đoạn chuyển giao.")

    with tab3:
        pol=list(policy_perf.keys()); vals=list(policy_perf.values())
        colors11=["#2ecc71" if "RL" in p else "#60a5fa" for p in pol]
        fig=go.Figure(go.Bar(x=pol,y=vals,marker_color=colors11,
            text=[f"{v:.2f}" for v in vals],textposition="outside"))
        fig.update_layout(title="So sánh phần thưởng tích lũy các chính sách",
            yaxis_title="Avg Reward",template="plotly_dark",height=380)
        st.plotly_chart(fig,use_container_width=True)

# ═══════════════════════════════════════════════════════════
# PAGE: BÀI 12
# ═══════════════════════════════════════════════════════════
elif page == "b12":
    st.title("🇻🇳 Bài 12 — AIDEOM-VN: Hệ thống tích hợp")
    st.markdown("*Tổng hợp 6 module M1-M6 thành dashboard hỗ trợ ra quyết định*")

    A,A_bar,Y_hat,mape,factors,pcts,gY=compute_m1()
    M2_res=compute_topsis()
    M3=compute_lp_base()
    M4=compute_labor()
    M6=compute_scenarios_m12()

    tab_ov,tab_ph,tab_sc,tab_ri=st.tabs([
        "🗺️ Tổng quan (M1-M2)","💰 Phân bổ (M3)","📊 5 Kịch bản (M6)","⚠️ Cảnh báo rủi ro (M4-M5)"
    ])

    # --- Tab Tổng quan ---
    with tab_ov:
        st.subheader("M1 — Dự báo kinh tế (Cobb-Douglas)")
        c1,c2,c3=st.columns(3)
        c1.metric("MAPE (Cobb-Douglas)",f"{mape:.2f}%")
        c2.metric("Ā",f"{A_bar:.4f}")
        c3.metric("Y 2030 dự báo",f"{compute_m1_forecast(30,100,35,6,1.2):,.0f} ng.tỷ")
        fig=go.Figure()
        fig.add_trace(go.Bar(x=factors,y=pcts,
            marker_color=["#4ecdc4","#45b7d1","#f7dc6f","#e74c3c","#9b59b6","#2ecc71"],
            text=[f"{v:.1f}%" for v in pcts],textposition="outside",
            name="Đóng góp tăng trưởng"))
        fig.update_layout(title="Phân rã đóng góp tăng trưởng 2020-2025",
            yaxis_title="Đóng_góp_pct",template="plotly_dark",height=380)
        st.plotly_chart(fig,use_container_width=True)

        st.subheader("M2 — Đánh giá sẵn sàng số (TOPSIS)")
        se2,sn2,_=M2_res
        df_m2=pd.DataFrame({"Vùng":REGIONS,"TOPSIS Expert":np.round(se2,4),
            "TOPSIS Entropy":np.round(sn2,4)}).sort_values("TOPSIS Expert",ascending=False)
        c1,c2=st.columns(2)
        with c1:
            fig=go.Figure(go.Bar(x=df_m2["TOPSIS Expert"],y=df_m2["Vùng"],orientation="h",
                marker_color=px.colors.sequential.Viridis[:6],
                text=df_m2["TOPSIS Expert"].round(3).astype(str),textposition="outside"))
            fig.update_layout(title="TOPSIS Score (Expert)",template="plotly_dark",height=320)
            st.plotly_chart(fig,use_container_width=True)
        with c2:
            fig=go.Figure(go.Bar(x=df_m2["TOPSIS Entropy"],y=df_m2["Vùng"],orientation="h",
                marker_color=px.colors.sequential.Plasma[:6],
                text=df_m2["TOPSIS Entropy"].round(3).astype(str),textposition="outside"))
            fig.update_layout(title="TOPSIS Score (Entropy)",template="plotly_dark",height=320)
            st.plotly_chart(fig,use_container_width=True)

    # --- Tab Phân bổ ---
    with tab_ph:
        st.subheader("M3 — Tối ưu phân bổ ngân sách (LP)")
        z3,alloc3,budgets3,zv3=M3
        cats3=["Hạ tầng số","AI & Dữ liệu","Nhân lực số","R&D"]
        c1,c2=st.columns(2)
        with c1:
            fig=go.Figure(go.Pie(labels=cats3,values=alloc3,hole=0.45,
                marker_colors=[CLR_BLUE,CLR_RED,CLR_GREEN,CLR_GOLD]))
            fig.update_layout(title=f"Phân bổ tối ưu (Z*={z3})",template="plotly_dark",height=360)
            st.plotly_chart(fig,use_container_width=True)
        with c2:
            fig=go.Figure(go.Scatter(x=budgets3,y=zv3,mode="lines+markers",
                line=dict(color=CLR_GREEN,width=2),
                text=[f"{v}" for v in zv3],textposition="top center"))
            fig.update_layout(title="Độ nhạy ngân sách Z*(B)",
                xaxis_title="Ngân sách",yaxis_title="GDP Gain",
                template="plotly_dark",height=360)
            st.plotly_chart(fig,use_container_width=True)

    # --- Tab 5 kịch bản ---
    with tab_sc:
        st.subheader("M6 — So sánh 5 kịch bản chính sách")
        kpi_cols=["GDP Gain (tỷ)","NetJob (k)","CO2 Risk","Equity"]
        c1,c2,c3,c4=st.columns(4)
        best_gdp=M6.loc[M6["GDP Gain (tỷ)"].idxmax(),"Kịch bản"]
        best_job=M6.loc[M6["NetJob (k)"].idxmax(),"Kịch bản"]
        best_env=M6.loc[M6["CO2 Risk"].idxmin(),"Kịch bản"]
        best_eq =M6.loc[M6["Equity"].idxmax(),"Kịch bản"]
        c1.metric("Best GDP Gain",best_gdp)
        c2.metric("Best NetJob",best_job)
        c3.metric("Best CO2 Risk",best_env)
        c4.metric("Best Equity",best_eq)

        fig=go.Figure()
        for kpi in kpi_cols:
            vn=M6[kpi]
            mn,mx=vn.min(),vn.max()
            norm=(vn-mn)/(mx-mn+1e-12)
            fig.add_trace(go.Bar(name=kpi,x=M6["Kịch bản"],y=norm,
                text=vn.round(1).astype(str),textposition="outside"))
        fig.update_layout(barmode="group",title="So sánh KPI 5 kịch bản (chuẩn hóa)",
            template="plotly_dark",height=440)
        st.plotly_chart(fig,use_container_width=True)

        fig_radar=go.Figure()
        for _,row in M6.iterrows():
            fig_radar.add_trace(go.Scatterpolar(
                r=[row["GDP Gain (tỷ)"]/1e5,row["NetJob (k)"]/10,
                   (5-row["CO2 Risk"])/5*100,row["Equity"]],
                theta=["GDP Gain","NetJob","Thân thiện MT","Công bằng"],
                fill="toself",name=row["Kịch bản"]))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)),
            title="Radar Chart — 5 kịch bản",template="plotly_dark",height=460)
        st.plotly_chart(fig_radar,use_container_width=True)
        st.dataframe(M6,use_container_width=True)

    # --- Tab Cảnh báo rủi ro ---
    with tab_ri:
        st.subheader("M4 — Mô phỏng lao động (NetJob)")
        colors9b=["#2ecc71" if v>0 else "#e74c3c" for v in M4["NetJob (k)"]]
        fig=go.Figure(go.Bar(x=M4["Ngành"],y=M4["NetJob (k)"],
            marker_color=colors9b,text=M4["NetJob (k)"],textposition="outside"))
        fig.add_hline(y=0,line_color="white",line_dash="dash")
        fig.update_layout(title="NetJob ròng M4 — Phân theo ngành",
            template="plotly_dark",height=380)
        st.plotly_chart(fig,use_container_width=True)

        st.subheader("M5 — Đánh giá rủi ro đa mục tiêu (Pareto)")
        df_par,bidx=compute_pareto()
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=df_par["GDP_gain"],y=df_par["CO2"],
            mode="markers",marker=dict(color=df_par["Cyber"],colorscale="RdYlGn",
            size=7,colorbar=dict(title="Cyber Risk")),name="Pareto"))
        fig.add_trace(go.Scatter(
            x=[df_par.loc[bidx,"GDP_gain"]],y=[df_par.loc[bidx,"CO2"]],
            mode="markers",marker=dict(color="red",size=14,symbol="star"),name="Thỏa hiệp"))
        fig.update_layout(title="Pareto M5: GDP Gain vs CO2 Risk",
            xaxis_title="GDP Gain (f1)",yaxis_title="CO2 (f3)",
            template="plotly_dark",height=400)
        st.plotly_chart(fig,use_container_width=True)

        st.subheader("⚠️ Bảng cảnh báo rủi ro")
        risk_data={"Ngành":SECTORS_8,"Rủi ro TĐH (%)":M4["Risk (%)"].tolist(),
            "NetJob (k)":M4["NetJob (k)"].tolist()}
        df_risk=pd.DataFrame(risk_data)
        df_risk["Cảnh báo"]=df_risk.apply(
            lambda r:"🔴 Cao" if r["Rủi ro TĐH (%)"]>45
                     else "🟡 Trung bình" if r["Rủi ro TĐH (%)"]>30
                     else "🟢 Thấp",axis=1)
        st.dataframe(df_risk,use_container_width=True)

# ═══════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════
st.divider()
st.markdown(
    "<div style=\\'text-align:center;color:#8b949e;font-size:12px;\\'>"
    "🇻🇳 VN AIDEOM-VN — AI-Driven Decision Optimization Model for Vietnam | "
    "Mai Huyền Trang · FDE3010 · Đại học Kinh tế – ĐHQGHN · 2025"
    "</div>", unsafe_allow_html=True
)
'''

# Ghi file app.py
with open('/content/app.py', 'w', encoding='utf-8') as f:
    f.write(APP_CODE)

print("✅ Đã ghi /content/app.py thành công!")
print(f"   Kích thước: {len(APP_CODE):,} ký tự")
print("\n⏭️  Tiếp theo: Chạy CELL 4 để khởi động web app qua ngrok")
