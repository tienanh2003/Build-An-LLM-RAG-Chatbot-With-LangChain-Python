import os
import sys
import re
import json
import uuid
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from bs4 import BeautifulSoup
from pymongo import MongoClient
from rank_bm25 import BM25Okapi
import base64
from io import BytesIO

import openai
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_session import Session
from torchvision import transforms
from PIL import Image

# Import các thư viện của LangChain
from langchain.tools import BaseTool
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI

from mongodb_retriever import MongoDBRAGRetriever, get_query_results

# --------------------------
# CONFIG & INIT
# --------------------------
from config import (
    OPENAI_API_KEY,
    MONGODB_CONNECTION_STRING,
    MONGODB_DB_NAME,
    MONGODB_COLLECTION_NAME,
    NUM_CLASSES,
    OLLAMA_BASE_URL
)

openai.api_key = OPENAI_API_KEY

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Thay bằng chuỗi bí mật của bạn
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Tạo thư mục để lưu file upload & ảnh kết quả segmentation
UPLOAD_FOLDER = os.path.join("src", "image")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --------------------------
# KẾT NỐI MONGODB
# --------------------------
client = MongoClient(MONGODB_CONNECTION_STRING)
db = client[MONGODB_DB_NAME]
collection = db[MONGODB_COLLECTION_NAME]

# --------------------------
# LOAD MODEL PHÂN LOẠI & PHÂN ĐOẠN
# --------------------------
from classification_model_loader import ClassificationModelLoader
from segmentation_model_loader import SegmentationModelLoader

device_local = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classification_model = ClassificationModelLoader("model_multiclass.pth", NUM_CLASSES, device_local).load_weights()
segmentation_model = SegmentationModelLoader("model_segmentation.pth", device_local).load_weights()

# --------------------------
# HÀM XỬ LÝ ẢNH
# --------------------------
def base64_to_image(base64_string: str) -> Image.Image:
    """
    Chuyển đổi chuỗi base64 thành đối tượng PIL Image.
    Nếu chuỗi chứa header như "data:image/png;base64,", ta sẽ loại bỏ phần đó.
    """
    if "," in base64_string:
        header, base64_data = base64_string.split(",", 1)
    else:
        base64_data = base64_string
    image_data = base64.b64decode(base64_data)
    image = Image.open(BytesIO(image_data)).convert("RGB")
    return image

def load_image_as_tensor(image_path: str):
    """
    Đọc file ảnh từ đường dẫn -> PIL -> transform -> tensor
    """
    try:
        img = Image.open(image_path).convert("RGB")
        transform_ = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        tensor = transform_(img)
        # Thêm batch dimension
        return tensor.unsqueeze(0)
    except Exception as e:
        print("Error loading image:", e)
        return None
    
def image_file_to_base64(image_path: str) -> str:
    """
    Đọc file ảnh từ đường dẫn và chuyển thành chuỗi base64 có thêm header.
    Ví dụ: "data:image/jpeg;base64,...."
    """
    try:
        with open(image_path, "rb") as f:
            image_data = f.read()
        base64_str = base64.b64encode(image_data).decode("utf-8")
        # Ở đây giả sử file ảnh là JPEG
        return f"data:image/jpeg;base64,{base64_str}"
    except Exception as e:
        print("Error converting image to base64:", e)
        return ""
    
import re

def replace_image_paths_with_base64(text: str) -> str:
    """
    Tìm các chuỗi có định dạng đường dẫn ảnh trong text (ví dụ: src\image\...) và thay thế bằng HTML <img> chứa ảnh base64.
    """
    # Regex tìm kiếm chuỗi có dạng "src\image\..." (có thể có dấu / hoặc \)
    pattern = r"(src[\\/]+image[\\/]+[^\s]+)"
    
    def replacer(match):
        image_path = match.group(1)
        # Chuyển đường dẫn thành base64
        base64_img = image_file_to_base64(image_path)
        if base64_img:
            return f"<img src='{base64_img}' style='max-width:300px;'/>";
        else:
            return image_path

    # Thay thế tất cả các đường dẫn ảnh trong text
    return re.sub(pattern, replacer, text)

# Mapping nhãn tiếng Việt cho model phân loại
vietnamese_labels = {
    0: "Tiền ung thư da do tác động của ánh nắng (AKIEC)",
    1: "Ung thư biểu mô đáy (BCC)",
    2: "Tổn thương da dạng keratosis lành tính (BKL)",
    3: "Dermatofibroma (Nốt sần da)",
    4: "Ung thư hắc tố (Melanoma)",
    5: "Nốt mầm da (Nevus)",
    6: "Tổn thương mạch máu (VASC)"
}

def get_classification_result(image_path: str, clf_model, device):
    tensor = load_image_as_tensor(image_path)
    if tensor is None:
        return "Không thể đọc ảnh."
    tensor = tensor.to(device)
    with torch.no_grad():
        output = clf_model(tensor)
        pred = torch.argmax(output, dim=1).item()
        return vietnamese_labels.get(pred, "Không xác định")

def get_segmentation_result(image_path: str, seg_model, device) -> str:

    tensor = load_image_as_tensor(image_path)
    if tensor is None:
        return ""
    with torch.no_grad():
        output = seg_model(tensor.to(device))
    # Giả sử output là tensor có kích thước [1, C, H, W], chuyển sang ảnh
    segmented_image = transforms.ToPILImage()(output.squeeze(0).cpu())
    seg_filename = f"seg_{uuid.uuid4().hex[:8]}.jpg"
    seg_path = os.path.join(UPLOAD_FOLDER, seg_filename)
    segmented_image.save(seg_path)
    return seg_path

# Hàm làm sạch + embedding PhoBERT
def advanced_clean_text(raw_html: str) -> str:
    text = BeautifulSoup(raw_html, "html.parser").get_text(separator=" ", strip=True)
    text = re.sub(r'[\u0000-\u001F\u007F-\u009F]', ' ', text)
    text = re.sub(r'[\u200B-\u200F\uFEFF]', '', text)
    text = re.sub(r'\ufeff', '', text)
    text = re.sub(r'[^a-zA-Z0-9À-ỹà-ỹ\s\.\,\;\:\-\?\!\(\)]', ' ', text)
    return ' '.join(text.split())

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2", use_fast=False)
phobert_model = AutoModel.from_pretrained("vinai/phobert-base-v2")
device = "cuda" if torch.cuda.is_available() else "cpu"
phobert_model = phobert_model.to(device)
phobert_model.eval()

def get_phobert_embedding(text: str):
    cleaned_text = advanced_clean_text(text)
    if not cleaned_text.strip():
        return [0.0]*768
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
            outputs = phobert_model(**inputs)
        embeddings = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]
        mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        pooled = torch.sum(embeddings * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
        pooled = F.normalize(pooled, p=2, dim=1)
        return pooled[0].cpu().numpy().tolist()
    except Exception as e:
        print("Error in get_phobert_embedding:", e)
        return [0.0] * 768

def calc_cosine_similarity(vec1, vec2):
    v1 = np.array(vec1, dtype=float)
    v2 = np.array(vec2, dtype=float)
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 * norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

# --------------------------
# TOOL LANGCHAIN: XỬ LÝ ẢNH Y HỌC (nhận vào path)
# --------------------------
class MedicalImageTool(BaseTool):
    name: str = "MedicalImageTool"
    description: str = "Nhận đường dẫn ảnh y học => trả về kết quả phân loại + segmentation"

    def _run(self, image_path: str) -> str:
        classification = get_classification_result(image_path, classification_model, device_local)
        seg_path = get_segmentation_result(image_path, segmentation_model, device_local)

        if seg_path:
            seg_msg = f"File segmentation: {seg_path}"
        else:
            seg_msg = "Không tạo được ảnh segmentation."
        # Kết hợp kết quả phân loại và segmentation thành một chuỗi văn bản

        return f"Kết quả phân loại: {classification}\n{seg_msg}"
    async def _arun(self, image_path: str) -> str:
        raise NotImplementedError("Async không hỗ trợ")

# --------------------------
# AGENT LANGCHAIN CHO TOOL ẢNH
# --------------------------
llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
tools = [MedicalImageTool()]
image_agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# --------------------------
# HÀM HỖ TRỢ XỬ LÝ VĂN BẢN (BM25, Retrieval, History)
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
    return query

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
    for (_, q_emb) in medical_queries_embeddings:
        sim = calc_cosine_similarity(query_emb, q_emb)
        if sim > best_sim:
            best_sim = sim
    print(f"[check_need_rag] Best similarity = {best_sim:.3f}")
    return best_sim >= threshold

def summarize_history(chat_history, query_text):
    history_docs = [f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" 
                    for msg in chat_history if msg.get('content', '').strip() != ""]
    query_emb = get_phobert_embedding(query_text)
    scored = []
    for doc in history_docs:
        emb = get_phobert_embedding(doc)
        if all(v == 0.0 for v in emb):
            continue
        score = calc_cosine_similarity(query_emb, emb)
        scored.append((doc, score))
    if not scored:
        return ""
    top_relevant = sorted(scored, key=lambda x: x[1], reverse=True)[:3]
    summary = "\n".join([doc for doc, _ in top_relevant])
    return summary

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

def normal_gpt_answer(refined_query: str) -> str:
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
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
    context = "\n".join([doc.get("page_content", "") for doc in top_docs])
    prompt = (
        f"Các tài liệu liên quan:\n{context}\n\n"
        f"Câu hỏi: {refined_query}\n\n"
        "Hãy cung cấp câu trả lời ngắn gọn dựa trên các tài liệu trên."
    )
    print(prompt)
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
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

# --------------------------
# PIPELINE TEXT & ẢNH CHO TRẢ LỜI (SỬ DỤNG GPT VÀ RAG)
# --------------------------
def pipelinegpt(query_text: str, chat_history=None, images=[]):
    results = []
    classification_info_list = []  # Để chứa thông tin phân loại của từng ảnh

    # Nếu có ảnh tải lên thì xử lý qua tool
    if images:
        for image_base64 in images:
            try:
                # Chuyển base64 thành PIL Image
                pil_image = base64_to_image(image_base64)
            except Exception as e:
                results.append(f"Lỗi khi chuyển base64 thành ảnh: {e}")
                continue

            # Lưu ảnh tạm thời ra ổ đĩa
            temp_filename = f"upload_{uuid.uuid4().hex[:8]}.png"
            temp_path = os.path.join(UPLOAD_FOLDER, temp_filename)
            try:
                pil_image.save(temp_path)
            except Exception as e:
                results.append(f"Lỗi khi lưu ảnh: {e}")
                continue

            # Sử dụng tool MedicalImageTool thông qua agent
            try:
                tool_result_str = image_agent.run(temp_path)
                print(f"tool_result_str : {tool_result_str}")
                # Ví dụ kết quả có thể có:
                # "The medical image has been classified as Basal Cell Carcinoma (BCC). The segmentation result can be found in the file path: src\image\seg_82e98f83.jpg."
                if "src\\image" in tool_result_str or "src/image" in tool_result_str:
                    # Dùng regex để tìm đường dẫn ảnh kết thúc bằng .jpg
                    pattern = r"(src[\\/]+image[\\/]+[^\s\"']+\.jpg)"
                    match = re.search(pattern, tool_result_str)
                    if match:
                        seg_path = match.group(1).strip().rstrip(".")
                        # Loại bỏ đường dẫn ảnh khỏi final text để có phần mô tả phân loại
                        classification_text = tool_result_str.replace(match.group(1), "").strip()
                    else:
                        seg_path = ""
                        classification_text = tool_result_str.strip()
                else:
                    seg_path = ""
                    classification_text = tool_result_str.strip()

            except Exception as e:
                results.append(f"Lỗi khi xử lý ảnh qua tool: {e}")
                continue

            classification_info_list.append(classification_text)
            
            if seg_path:
                seg_base64 = image_file_to_base64(seg_path)
                seg_html = f"<strong>Kết quả segmentation:</strong><br><img src='{seg_base64}' alt='segmentation' style='max-width:300px;'/>"
            else:
                seg_html = ""
                
            # Tạo HTML hiển thị kết quả của ảnh này
            img_result_html = (
                f"<strong>Kết quả phân loại:</strong> {classification_text}<br>"
                f"{seg_html}"
            )
            results.append(f"<strong>Kết quả xử lý ảnh:</strong><br>{img_result_html}")

    # Nếu có câu hỏi văn bản thì xử lý phần text
    if query_text:
        # Nếu có thông tin từ ảnh thì bổ sung vào query_text
        if classification_info_list:
            classification_info = "; ".join(classification_info_list)
            query_text = f"{query_text}\nThông tin bổ sung từ hình ảnh: {classification_info}"
        
        # Xử lý văn bản như phần cũ
        refined_query = query_text
        query_emb = get_phobert_embedding(refined_query)
        need_rag = check_need_rag(query_emb, threshold=0.9)
        
        if chat_history:
            context = summarize_history(chat_history, query_text)
            refined_query = f"Lịch sử hội thoại:\n{context}\nCâu hỏi mới: {query_text}"
        
        print(f"[DEBUG] refined_query ban đầu:\n{refined_query}")
        
        if not need_rag:
            print("[INFO] Similarity < 0.9 => Dùng model bình thường")
            text_result = normal_gpt_answer(refined_query)
        else:
            print("[INFO] Similarity >= 0.9 => Dùng RAG pipeline")
            vec_docs = get_query_results(query_emb, collection, limit=5)
            print("[INFO] vecdoc đã xong:", vec_docs)
            if not vec_docs:
                text_result = "[RAG] Không tìm thấy tài liệu phù hợp."
            else:
                top_docs = bm25_rerank(refined_query, vec_docs, top_n=2)
                print("[INFO] BM25 đã xong:", top_docs)
                if not top_docs:
                    text_result = "[RAG] Không có tài liệu sau BM25."
                else:
                    text_result = rag_llm_gpt_answer(refined_query, top_docs)
        
        final_query = f"{query_text}\nThông tin bổ sung từ tài liệu:\n{text_result}"
        final_text_result = normal_gpt_answer(final_query)
        results.append(f"<strong>Kết quả văn bản:</strong><br>{final_text_result}")

    return "<br><br>".join(results)
    
def pipelinetest2model(query_text: str, chat_history=None, images=[]):
    # Xử lý từng ảnh base64 từ client qua model phân loại và segmentation
    results = []
    for image_base64 in images:
        # Chuyển base64 thành PIL Image
        try:
            pil_image = base64_to_image(image_base64)
        except Exception as e:
            results.append(f"Lỗi khi chuyển base64 thành ảnh: {e}")
            continue

        # Lưu ảnh tạm thời ra ổ đĩa
        temp_filename = f"upload_{uuid.uuid4().hex[:8]}.png"
        temp_path = os.path.join(UPLOAD_FOLDER, temp_filename)
        try:
            pil_image.save(temp_path)
        except Exception as e:
            results.append(f"Lỗi khi lưu ảnh: {e}")
            continue

        # Xử lý qua model phân loại và segmentation
        classification_result = get_classification_result(temp_path, classification_model, device_local)
        seg_path = get_segmentation_result(temp_path, segmentation_model, device_local)

        # Nếu có ảnh segmentation, chuyển file segmentation sang base64 để nhúng hiển thị
        if seg_path:
            seg_base64 = image_file_to_base64(seg_path)
            seg_msg = f"Kết quả segmentation: <img src='{seg_base64}' alt='segmentation result' style='max-width:300px;'/>"
        else:
            seg_msg = "Không tạo được ảnh segmentation."

        result = (
            f"<strong>Ảnh gốc:</strong> <br><img src='{image_base64}' alt='input image' style='max-width:300px;'/><br>"
            f"<strong>Kết quả phân loại:</strong> {classification_result}<br>"
            f"{seg_msg}"
        )
        results.append(result)
    return "<br><br>".join(results)

# --------------------------
# ROUTES FLASK
# --------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if "messages" not in session:
        session["messages"] = [{"role": "assistant", "content": "Xin chào, tôi có thể giúp gì?"}]
    if request.method == "POST":
        data = request.get_json()
        user_input = data.get("user_input", "").strip()
        images = data.get("images", [])
        if user_input or images:
            session["messages"].append({
                "role": "user",
                "content": user_input,
                "images": images
            })
            reply = pipelinegpt(user_input, session["messages"], images)
            # reply = pipelinetest2model(user_input, session["messages"], images)
            session["messages"].append({"role": "assistant", "content": reply})
        return jsonify({"chat_history": session["messages"]})
    return render_template("index.html", chat_history=session["messages"])

# --------------------------
# MAIN
# --------------------------
def main():
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=5000)

if __name__ == "__main__":
    main()
