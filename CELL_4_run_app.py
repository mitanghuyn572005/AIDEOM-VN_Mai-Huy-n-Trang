# ============================================================
# CELL 4 — CHẠY WEB APP QUA NGROK
# Chạy sau Cell 3. Tự động khởi động Streamlit + tạo URL public
# ============================================================

import subprocess
import time
import threading
import os
import sys

# ── Bước 1: Đảm bảo pyngrok được cài ──────────────────────
try:
    from pyngrok import ngrok, conf
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pyngrok"])
    from pyngrok import ngrok, conf

# ── Bước 2: Cấu hình ngrok ─────────────────────────────────
# (Tuỳ chọn) Nếu bạn có authtoken ngrok cá nhân, paste vào đây:
NGROK_AUTHTOKEN = ""   # ← để trống = dùng anonymous (giới hạn 1 tunnel)

if NGROK_AUTHTOKEN:
    ngrok.set_auth_token(NGROK_AUTHTOKEN)
    print("✅ Đã cấu hình ngrok authtoken.")
else:
    print("ℹ️  Dùng ngrok anonymous (có thể cần xác nhận captcha lần đầu).")

# ── Bước 3: Kill Streamlit cũ nếu đang chạy ───────────────
try:
    subprocess.run(["pkill", "-f", "streamlit"], capture_output=True)
    time.sleep(1)
    print("🔄 Đã dừng Streamlit cũ (nếu có).")
except:
    pass

# ── Bước 4: Khởi động Streamlit trong background ──────────
STREAMLIT_PORT = 8501
LOG_FILE = "/content/streamlit.log"

def start_streamlit():
    with open(LOG_FILE, "w") as logf:
        subprocess.Popen(
            [
                sys.executable, "-m", "streamlit", "run",
                "/content/app.py",
                "--server.port", str(STREAMLIT_PORT),
                "--server.headless", "true",
                "--server.enableCORS", "false",
                "--server.enableXsrfProtection", "false",
                "--server.fileWatcherType", "none",
                "--logger.level", "error",
            ],
            stdout=logf,
            stderr=logf,
        )

t = threading.Thread(target=start_streamlit, daemon=True)
t.start()

print("⏳ Đang khởi động Streamlit...")
time.sleep(8)  # Chờ Streamlit warm-up

# ── Bước 5: Tạo ngrok tunnel ──────────────────────────────
try:
    # Đóng tunnel cũ (nếu có)
    tunnels = ngrok.get_tunnels()
    for tunnel in tunnels:
        ngrok.disconnect(tunnel.public_url)
    time.sleep(1)
except:
    pass

tunnel = ngrok.connect(STREAMLIT_PORT, "http")
public_url = tunnel.public_url

print("\n" + "="*60)
print("🚀  WEB APP ĐÃ SẴN SÀNG!")
print("="*60)
print(f"🌐  URL Public : {public_url}")
print(f"🖥️  Local URL  : http://localhost:{STREAMLIT_PORT}")
print("="*60)
print("📌  Hướng dẫn:")
print("    1. Click vào URL trên để mở web app")
print("    2. Nếu thấy màn hình trống, đợi 5-10 giây rồi reload")
print("    3. URL sẽ hoạt động trong ~2 giờ (ngrok free)")
print("="*60)

# ── Bước 6: Kiểm tra Streamlit đã chạy chưa ──────────────
time.sleep(3)
try:
    import urllib.request
    urllib.request.urlopen(f"http://localhost:{STREAMLIT_PORT}", timeout=5)
    print("\n✅ Streamlit server đang chạy bình thường!")
except Exception as e:
    print(f"\n⚠️  Streamlit chưa phản hồi: {e}")
    print("   Thử đọc log:")
    try:
        with open(LOG_FILE) as lf:
            log_content = lf.read()[-1500:]
            print(log_content)
    except:
        pass

print(f"\n🔗  {public_url}")
