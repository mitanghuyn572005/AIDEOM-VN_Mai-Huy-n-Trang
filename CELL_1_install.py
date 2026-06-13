# ============================================================
# CELL 1 — CÀI ĐẶT THƯ VIỆN
# Chạy cell này đầu tiên để cài tất cả thư viện cần thiết
# ============================================================

import subprocess, sys

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

packages = [
    "streamlit",
    "plotly",
    "pandas",
    "numpy",
    "scipy",
    "matplotlib",
    "pulp",
    "cvxpy",
    "pymoo",
    "pyngrok",
    "gymnasium",
    "openpyxl",
    "seaborn",
    "kaleido",        # plotly static export
]

for p in packages:
    try:
        install(p)
        print(f"✅ {p}")
    except Exception as e:
        print(f"⚠️  {p}: {e}")

# Cài pyomo riêng
try:
    install("pyomo")
    subprocess.check_call(["apt-get", "install", "-y", "-q", "glpk-utils"], stdout=subprocess.DEVNULL)
    print("✅ pyomo + glpk")
except Exception as e:
    print(f"⚠️  pyomo/glpk: {e}")

print("\n✅ Hoàn tất cài đặt thư viện!")
