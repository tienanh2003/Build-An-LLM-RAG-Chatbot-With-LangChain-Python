import os
import sys
import re
import json
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from bs4 import BeautifulSoup
from pymongo import MongoClient
from rank_bm25 import BM25Okapi
import streamlit as st
from dotenv import load_dotenv
import base64
from PIL import Image
import io
import streamlit.components.v1 as components
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_session import Session
# Cập nhật theo khuyến nghị
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import OpenAIEmbeddings
import openai


from config import OPENAI_API_KEY 
openai.api_key = OPENAI_API_KEY


app = Flask(__name__)

# Cấu hình secret key để Flask có thể mã hóa session
app.secret_key = "supersecretkey"  # Thay bằng chuỗi bí mật của riêng bạn

# Cấu hình Flask-Session để lưu session trên filesystem (server-side)
app.config["SESSION_TYPE"] = "filesystem"
# Nếu muốn, bạn có thể chỉ định thư mục lưu file session:
# app.config["SESSION_FILE_DIR"] = "/path/to/your/session/folder"

# Khởi tạo Session
Session(app)

# Import cấu hình từ file config.py (để lấy chuỗi kết nối, số lớp, URL, …)
from config import MONGODB_CONNECTION_STRING, MONGODB_DB_NAME, MONGODB_COLLECTION_NAME, NUM_CLASSES, OLLAMA_BASE_URL, OPENAI_API_KEY

# Import các module hỗ trợ
from classification_model_loader import ClassificationModelLoader
from segmentation_model_loader import SegmentationModelLoader
from mongodb_retriever import MongoDBRAGRetriever, get_query_results
from local_ollama import get_llm_and_agent, normal_model_answer, rag_llm_answer

# --------------------------
# Khởi tạo kết nối MongoDB toàn cục (sử dụng cho pipeline)
# --------------------------
client = MongoClient(MONGODB_CONNECTION_STRING)
db = client[MONGODB_DB_NAME]
collection = db[MONGODB_COLLECTION_NAME]

# Load model phân loại và segmentation
device_local = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classification_model = ClassificationModelLoader("model_multiclass.pth", NUM_CLASSES, device_local).load_weights()
segmentation_model = SegmentationModelLoader("model_segmentation.pth", device_local).load_weights()

# Khởi tạo agent: Agent này tích hợp các tool (như classification, segmentation, BCLIP embedding)
# để hỗ trợ xử lý input phức tạp (ví dụ, khi người dùng gửi ảnh kèm theo text).
# Trong pipeline hiện tại, nếu input chỉ là text, chúng ta sử dụng normal_model_answer hoặc rag_llm_answer.
agent_executor = get_llm_and_agent(
    MongoDBRAGRetriever(MONGODB_CONNECTION_STRING, MONGODB_DB_NAME, MONGODB_COLLECTION_NAME),
    classification_model,
    segmentation_model
)
# Lưu ý: Agent được khởi tạo để mở rộng xử lý input, tuy nhiên trong pipeline hiện tại (text-only)
# agent chưa được gọi trực tiếp.

vietnamese_labels = {
    0: "Tiền ung thư da do tác động của ánh nắng (AKIEC)",
    1: "Ung thư biểu mô đáy (BCC)",
    2: "Tổn thương da dạng keratosis lành tính (BKL)",
    3: "Dermatofibroma (Nốt sần da)",
    4: "Ung thư hắc tố (Melanoma)",
    5: "Nốt mầm da (Nevus)",
    6: "Tổn thương mạch máu (VASC)"
}

# --------------------------
# Hàm làm sạch và tính embedding với PhoBERT
# --------------------------
def advanced_clean_text(raw_html: str) -> str:
    text = BeautifulSoup(raw_html, "html.parser").get_text(separator=" ", strip=True)
    text = re.sub(r'[\u0000-\u001F\u007F-\u009F]', ' ', text)
    text = re.sub(r'[\u200B-\u200F\uFEFF]', '', text)
    text = re.sub(r'\ufeff', '', text)
    text = re.sub(r'[^a-zA-Z0-9À-ỹà-ỹ\s\.\,\;\:\-\?\!\(\)]', ' ', text)
    return ' '.join(text.split())

# Khởi tạo PhoBERT (global)
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2", use_fast=False)
model = AutoModel.from_pretrained("vinai/phobert-base-v2")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

def get_phobert_embedding(text: str):
    cleaned_text = advanced_clean_text(text)
    if not cleaned_text.strip():
        raise ValueError("Text rỗng sau khi làm sạch!")
    try:
        inputs = tokenizer(
            cleaned_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]
        mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        pooled = torch.sum(embeddings * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
        pooled = F.normalize(pooled, p=2, dim=1)
        return pooled[0].cpu().numpy().tolist()
    except Exception as e:
        print("Error in get_phobert_embedding:", e, "for text snippet:", cleaned_text[:100])
        return [0.0] * 768

# --------------------------
# Các hàm BM25, tính cosine similarity, và self-reflection
# --------------------------
def calc_cosine_similarity(vec1, vec2):
    v1 = np.array(vec1, dtype=float)
    v2 = np.array(vec2, dtype=float)
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 * norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

def self_reflection_ollama(query: str) -> str:
    # Có thể tinh chỉnh câu hỏi ở đây, hiện tại chỉ trả về query gốc.
    # query + {"Bổ sung thêm ngữ nghĩa cho câu hỏi trên"}
    return query

def bm25_rerank(query: str, docs: list, top_n):
    corpus = [doc["page_content"] for doc in docs]
    tokenized_corpus = [c.split() for c in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    query_tokens = query.split()
    scores = bm25.get_scores(query_tokens)
    scored_docs = []
    for doc, s in zip(docs, scores):
        doc["bm25_score"] = float(s)
        scored_docs.append(doc)
    scored_docs = sorted(scored_docs, key=lambda x: x["bm25_score"], reverse=True)
    return scored_docs[:top_n]

# 20 câu y học mẫu và tính embedding của chúng
medical_queries = [
    "Ung thư hắc tố da là gì?",
    "Triệu chứng ung thư biểu mô tế bào đáy",
    "Cách điều trị u hắc lành tính",
    "Dấu hiệu da bị tổn thương do tia UV",
    "Nguyên nhân gây ung thư hắc tố da",
    "Phòng ngừa ung thư da hiệu quả",
    "Tác động của tia UV đến sức khỏe da",
    "Biểu hiện của ung thư da sớm",
    "Phương pháp chẩn đoán ung thư hắc tố da",
    "Cách điều trị ung thư biểu mô tế bào đáy",
    "Phẫu thuật ung thư da",
    "Liệu pháp điều trị ung thư da",
    "Tầm quan trọng của kem chống nắng",
    "Rủi ro khi tiếp xúc nhiều với tia UV",
    "Cách chăm sóc da sau điều trị ung thư",
    "Các bước chẩn đoán bệnh da liễu",
    "Phòng tránh các bệnh về da",
    "Tầm quan trọng của kiểm tra da định kỳ",
    "Các dấu hiệu cần gặp bác sĩ da liễu",
    "Phương pháp điều trị bệnh da hiện đại"
]
medical_queries_embeddings = []
for q in medical_queries:
    emb = get_phobert_embedding(q)
    medical_queries_embeddings.append((q, emb))
print("Đã tính embedding cho 20 câu y học.")

def check_need_rag(query_emb, threshold=0.9):
    best_sim = 0
    for (q_text, q_emb) in medical_queries_embeddings:
        sim = calc_cosine_similarity(query_emb, q_emb)
        if sim > best_sim:
            best_sim = sim
    print(f"[check_need_rag] Best similarity = {best_sim:.3f}")
    return best_sim >= threshold

def summarize_history(chat_history, query_text):
    """
    Lấy các đoạn hội thoại liên quan nhất đến câu hỏi mới.
    """
    # Lấy nội dung của từng tin nhắn, loại bỏ những tin nhắn rỗng.
    history_docs = [f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" 
                    for msg in chat_history if msg.get('content', '').strip() != ""]
    
    # Lấy embedding cho query
    query_emb = get_phobert_embedding(query_text)
    
    scored = []
    for doc in history_docs:
        emb = get_phobert_embedding(doc)
        # Nếu embedding không hợp lệ (ví dụ, toàn số 0) thì bỏ qua
        if all(v == 0.0 for v in emb):
            continue
        score = calc_cosine_similarity(query_emb, emb)
        scored.append((doc, score))
    
    if not scored:
        return ""
    
    # Sắp xếp theo điểm similarity giảm dần và chọn 3 đoạn đầu
    top_relevant = sorted(scored, key=lambda x: x[1], reverse=True)[:3]
    
    summary = "\n".join([doc for doc, _ in top_relevant])
    return summary

# --------------------------
# Helper functions để xử lý hình ảnh qua 2 model (phân loại và phân đoạn)
# --------------------------
def process_image(image_data: str):
    try:
        # Giả sử image_data là chuỗi base64, loại bỏ phần header nếu có
        image_bytes = base64.b64decode(image_data.split(",")[-1])
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        image_tensor = transform(image).unsqueeze(0)  # thêm batch dimension
        return image_tensor
    except Exception as e:
        print("Error processing image:", e)
        return None

def get_classification_result(image_data: str) -> str:
    image_tensor = process_image(image_data)
    if image_tensor is None:
        return "No image"
    with torch.no_grad():
        output = classification_model(image_tensor.to(device_local))
        pred = torch.argmax(output, dim=1).item()
        # Giả sử bạn có mapping label, ở đây dùng f-string làm ví dụ
        label = f"Label_{pred}"
        print(label)
        disease_vn = vietnamese_labels.get(pred, "Không xác định")
        return disease_vn

def get_segmentation_result(image_data: str) -> str:
    image_tensor = process_image(image_data)
    if image_tensor is None:
        return "No image"
    with torch.no_grad():
        output = segmentation_model(image_tensor.to(device_local))
        # Giả sử tính trung bình của mask làm ví dụ
        mean_val = output.mean().item()
        return f"Segmentation_mean: {mean_val:.2f}"
    
# --------------------------
# Pipeline chính: Xử lý câu hỏi, tính embedding, so sánh với mẫu, truy vấn MongoDB và trả lời
# --------------------------
def pipeline(query_text: str, chat_history=None, images=[]):
    """
    Xử lý câu hỏi của người dùng, có thể sử dụng lịch sử hội thoại (chat_history)
    để cung cấp ngữ cảnh cho model.
    """
    refined_query = query_text
    image_output_str = ""
    if images:
        latest_img = images[-1]
        class_result = get_classification_result(latest_img)
        # seg_result = get_segmentation_result(latest_img)
        image_output_str = f"Image (mới nhất): Classification: {class_result}"
        refined_query = refined_query + "\nThông tin hình ảnh:\n" + image_output_str

    return refined_query

    # B1: Self Reflection (tinh chỉnh câu hỏi, ở đây trả về query gốc)
    refined_query = self_reflection_ollama(query_text)

    query_emb = get_phobert_embedding(refined_query)
    
    need_rag = check_need_rag(query_emb, threshold=0.9)
    
    # Nếu có chat history, ghép nối làm context (có thể điều chỉnh định dạng theo ý muốn)
    if chat_history:
        context = summarize_history(chat_history, query_text)
        refined_query = f"Lịch sử hội thoại:\n{context}\nCâu hỏi mới: {refined_query}"

    
    print(f"[Self Reflection] refined_query = {refined_query}")
    
    if not need_rag:
        print("[INFO] Similarity < 0.9 => Dùng model bình thường")
        return normal_model_answer(refined_query)
    else:
        print("[INFO] Similarity >= 0.9 => Dùng RAG pipeline")
        vec_docs = get_query_results(query_emb, collection, limit=10)
        print("[INFO] vecdoc đã xong")
        print(vec_docs)
        if not vec_docs:
            return "[RAG] Không tìm thấy tài liệu phù hợp."
        top_docs = bm25_rerank(refined_query, vec_docs, top_n=5)
        print("[INFO] BM25 đã xong")
        if not top_docs:
            return "[RAG] Không có tài liệu sau BM25."
        answer = rag_llm_answer(refined_query, top_docs)
        return answer

def normal_gpt_answer(refined_query: str) -> str:
    """
    Sử dụng GPT để trả lời trực tiếp câu hỏi đã được tinh chỉnh.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Bạn có thể thay đổi model nếu cần
            messages=[
                {"role": "system", "content": "Bạn là một chuyên gia về lĩnh vực y tế"},
                {"role": "user", "content": refined_query}
            ],
            temperature=0,
            max_tokens=128
        )
        answer = response['choices'][0]['message']['content'].strip()
        return answer
    except Exception as e:
        return f"Lỗi khi gọi GPT: {str(e)}"


def rag_llm_gpt_answer(refined_query: str, top_docs: list) -> str:
    """
    Sử dụng GPT để trả lời dựa trên thông tin từ các tài liệu liên quan được truy xuất (top_docs).
    """
    # Ghép nội dung tài liệu từ các tài liệu top_docs. Giả sử mỗi tài liệu có trường "page_content".
    context = "\n".join([doc.get("page_content", "") for doc in top_docs])
    prompt = (
        f"Các tài liệu liên quan:\n{context}\n\n"
        f"Câu hỏi: {refined_query}\n\n"
        "Hãy cung cấp câu trả lời ngắn gọn dựa trên các tài liệu trên."
    )
    print(prompt)
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",  # Bạn có thể thay đổi model nếu cần
            messages=[
                {"role": "system", "content": "Bạn là một chuyên gia về lĩnh vực da liễu"},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=128
        )
        answer = response['choices'][0]['message']['content'].strip()
        return answer
    except Exception as e:
        return f"Lỗi khi gọi GPT: {str(e)}"
      
def pipelinegpt(query_text: str, chat_history=None, images=[]):
    """
    Xử lý câu hỏi của người dùng, có thể sử dụng lịch sử hội thoại (chat_history)
    để cung cấp ngữ cảnh cho model.
    """
    refined_query = query_text

    # image_output_str = ""
    # if images:
    #     for i, img_data in enumerate(images):
    #         class_result = get_classification_result(img_data)
    #         seg_result = get_segmentation_result(img_data)
    #         image_output_str += f"Image {i+1}: Classification: {class_result}, Segmentation: {seg_result}\n"
    #     refined_query += "\nThông tin hình ảnh:\n" + image_output_str
    
    # return refined_query

    query_emb = get_phobert_embedding(refined_query)
    
    need_rag = check_need_rag(query_emb, threshold=0.9)
    
    # Nếu có chat history, ghép nối làm context (có thể điều chỉnh định dạng theo ý muốn)
    if chat_history:
        context = summarize_history(chat_history, query_text)
        refined_query = f"Lịch sử hội thoại:\n{context}\n Thực hiện: {refined_query}"

    
    print(f"Refined_query = {refined_query}")
    
    if not need_rag:
        print("[INFO] Similarity < 0.9 => Dùng model bình thường")
        return normal_gpt_answer(refined_query)
    else:
        print("[INFO] Similarity >= 0.9 => Dùng RAG pipeline")
        vec_docs = get_query_results(query_emb, collection, limit=5)
        print("[INFO] vecdoc đã xong")
        print(vec_docs)
        if not vec_docs:
            return "[RAG] Không tìm thấy tài liệu phù hợp."
        top_docs = bm25_rerank(refined_query, vec_docs, top_n=2)
        print("[INFO] BM25 đã xong")
        print(top_docs)
        if not top_docs:
            return "[RAG] Không có tài liệu sau BM25."
        answer = rag_llm_gpt_answer(refined_query, top_docs)
        return answer
    
# ============= ROUTES FLASK =============
@app.route("/", methods=["GET", "POST"])
def index():
    if "messages" not in session:
        session["messages"] = [
            {"role": "assistant", "content": "Xin chào, tôi có thể giúp gì?"}
        ]
    if request.method == "POST":
        data = request.get_json()
        user_input = data.get("user_input", "").strip()
        images = data.get("images", [])
        if user_input or images:
            # Thêm tin nhắn người dùng (văn bản và hình ảnh) vào session
            session["messages"].append({
                "role": "user",
                "content": user_input,
                "images": images
            })
            # Gọi pipeline để xử lý văn bản, hình ảnh có thể không cần thiết cho model trả lời
            # reply = pipelinegpt(user_input, session["messages"])
            reply = pipelinegpt(user_input, session["messages"], images)
            session["messages"].append({"role": "assistant", "content": reply})
        return jsonify({"chat_history": session["messages"]})
    return render_template("index.html", chat_history=session["messages"])


# ============= HÀM MAIN =============
def main():
    """
    Đây là hàm main - bạn có thể tùy chỉnh host, port, debug v.v. 
    Khi chạy 'python app.py', code trong 'if __name__=="__main__": main()' sẽ được gọi.
    """
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=5000)

if __name__ == "__main__":
    main()
