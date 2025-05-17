import os
from dotenv import load_dotenv

# Nạp file .env vào biến môi trường của hệ thống
load_dotenv()

# Lấy các biến nhạy cảm từ .env
MONGODB_CONNECTION_STRING = os.getenv("MONGODB_CONNECTION_STRING")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
URL_NGROK = os.getenv("URL_NGROK")

# Các biến cấu hình khác có thể đặt mặc định nếu không có trong .env
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "rag_database")
MONGODB_COLLECTION_NAME = os.getenv("MONGODB_COLLECTION_NAME", "documents")
NUM_CLASSES = int(os.getenv("NUM_CLASSES", "7"))

# Kiểm tra các biến nhạy cảm đã được nạp chưa
if not MONGODB_CONNECTION_STRING:
    raise ValueError("Thiếu MONGODB_CONNECTION_STRING trong .env")
if not OLLAMA_BASE_URL:
    raise ValueError("Thiếu OLLAMA_BASE_URL trong .env")

# In ra để kiểm tra (có thể xóa dòng in này sau khi kiểm tra)
print("MONGODB_CONNECTION_STRING:", MONGODB_CONNECTION_STRING)
print("OLLAMA_BASE_URL:", OLLAMA_BASE_URL)
print("MONGODB_DB_NAME:", MONGODB_DB_NAME)
print("MONGODB_COLLECTION_NAME:", MONGODB_COLLECTION_NAME)
print("NUM_CLASSES:", NUM_CLASSES)