import os
import sys
import requests
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))
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
from pydantic import Field

from openai import OpenAI
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_session import Session
from torchvision import transforms
from PIL import Image
import time

# Import các thư viện của LangChain
from langchain.tools import BaseTool
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
# from langchain.chat_models import ChatOllama
from langchain_ollama import ChatOllama
# from langchain.chat_models import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from flask import send_from_directory


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
    GOOGLE_API_KEY,
    OLLAMA_BASE_URL,
    URL_NGROK
)

client = OpenAI(api_key=OPENAI_API_KEY)
genai.configure(api_key=GOOGLE_API_KEY)

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
mongodb_client = MongoClient(MONGODB_CONNECTION_STRING)
db = mongodb_client[MONGODB_DB_NAME]
collection = db[MONGODB_COLLECTION_NAME]

# --------------------------
# LOAD MODEL PHÂN LOẠI & PHÂN ĐOẠN
# --------------------------
from model.classification_model_loader import ClassificationModelLoader
from model.segmentation_model_loader import SegmentationModelLoader

device_local = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classification_model = ClassificationModelLoader("model/model_multiclass.pth", NUM_CLASSES, device_local).load_weights()
segmentation_model = SegmentationModelLoader("model/model_segmentation.pth", device_local).load_weights()

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

# Thay model chỗ này
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(
    "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base", use_fast=False
)
phobert_model = AutoModel.from_pretrained(
    "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"
).to(device)
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

class ImageGenerationTool(BaseTool):
    name: str = "ImageGenerationTool"
    description: str = "Tạo ảnh từ mô tả văn bản (text-to-image)"
    ngrok_url: str = Field(default=URL_NGROK)

    def __init__(self):
        super().__init__()
        # Khởi tạo ngrok URL từ config
        self.ngrok_url = URL_NGROK

    # def _run(self, text_prompt: str) -> str:
    #     try:
    #         start = time.time()
    #         response = requests.post(self.ngrok_url, json={"prompt": text_prompt}, timeout=60)

    #         if response.status_code == 200:
    #             img_base64 = response.json().get("image")
    #             if not img_base64:
    #                 return "❌ Service không trả về ảnh."

    #             img_data = base64.b64decode(img_base64)
    #             image = Image.open(BytesIO(img_data))

    #             # Tạo đường dẫn lưu ảnh tạm
    #             temp_filename = f"generated_{uuid.uuid4().hex[:8]}.png"
    #             temp_path = os.path.join("src", "image", temp_filename)
    #             os.makedirs(os.path.dirname(temp_path), exist_ok=True)
    #             image.save(temp_path)

    #             duration = time.time() - start

    #             # Trả về ảnh nhúng trong HTML (base64)
    #             return (
    #                 f"<strong>✅ Ảnh đã tạo ({duration:.2f}s):</strong><br>"
    #                 f"<img src='data:image/png;base64,{base64.b64encode(img_data).decode()}' "
    #                 f"alt='Generated Image' style='max-width:300px;'/>"
    #             )
    #         else:
    #             return f"❌ Lỗi khi gọi API: HTTP {response.status_code} - {response.text}"

    #     except Exception as e:
    #         return f"❌ Exception: {str(e)}"
    def _run(self, text_prompt: str) -> str:
        try:
            resp = requests.post(self.ngrok_url, json={"prompt": text_prompt}, timeout=60)
            resp.raise_for_status()

            img_b64 = resp.json().get("image")
            if not img_b64:
                return "❌ Service không trả về ảnh."

            # decode và mở ảnh
            img_data = base64.b64decode(img_b64)
            image = Image.open(BytesIO(img_data))

            # lưu vào thư mục src/image
            filename = f"{uuid.uuid4().hex}.png"
            save_dir = os.path.join("src", "image")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, filename)
            image.save(save_path)

            # chỉ return đúng đường dẫn filesystem như MedicalImageTool
            return save_path

        except Exception as e:
            return f"❌ Exception: {e}"

# --------------------------
# AGENT LANGCHAIN CHO TOOL ẢNH
# --------------------------
# llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
# tools = [MedicalImageTool()]
# image_agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
# Tạo các tools và agents riêng cho từng LLM

def create_agent_for_llm(llm_type):
    """
    Khởi tạo agent với 2 tool:
      - MedicalImageTool: phân tích ảnh y tế (classification + segmentation)
      - ImageGenerationTool: sinh ảnh từ text
    Dùng prefix để đưa system prompt, không pass system_message vào LLM constructor.
    """
    tools = [MedicalImageTool(), ImageGenerationTool()]

    system_prompt = (
        "Bạn là một trợ lý AI có khả năng sử dụng các công cụ sau:\n"
        "- Dùng `MedicalImageTool` để phân tích ảnh y tế nếu người dùng cung cấp đường dẫn ảnh.\n"
        "- `ImageGenerationTool`: sinh ảnh từ mô tả văn bản.\n"
        "Khi nhận prompt yêu cầu tạo ảnh, hãy:\n"
        "  1. Gọi `ImageGenerationTool` với đúng mô tả người dùng.\n"
        "  2. Chỉ trả về đường dẫn file do tool trả về (ví dụ: `src/image/abc123.png`).\n"
        "  3. Không embed Base64, không thêm text hay thẻ khác quanh `<img>`.\n"
        "Nếu prompt không yêu cầu tạo ảnh hoặc phân tích ảnh, chỉ trả về text thuần, không gọi tool.\n"
    )

    # Khởi tạo LLM, bỏ system_message
    if llm_type == "gpt":
        llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
    elif llm_type == "ollama":
        llm = ChatOllama(model="phi3", temperature=0)
    elif llm_type == "gemini":
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0)
    else:
        llm = ChatOpenAI(temperature=0, model_name="gpt-4o")

    # Tạo agent, truyền prompt qua prefix
    return initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        prefix=system_prompt
    )

# Khởi tạo các agents
gpt_agent = create_agent_for_llm("gpt")
ollama_agent = create_agent_for_llm("ollama")
gemini_agent = create_agent_for_llm("gemini")

# Hàm sử dụng agent dựa trên loại model
def process_image_with_agent(image_path, model_type):
    if model_type == "gpt-4":
        return gpt_agent.run(f"Phân tích ảnh y tế tại đường dẫn: {image_path}")
    elif model_type == "ollama":
        return ollama_agent.run(f"Phân tích ảnh y tế tại đường dẫn: {image_path}")
    elif model_type == "gemini":
        return gemini_agent.run(f"Phân tích ảnh y tế tại đường dẫn: {image_path}")
    else:
        return "Loại model không được hỗ trợ"

# Hàm tạo ảnh từ văn bản
def generate_image_with_agent(text_prompt, model_type):
    if model_type == "gpt-4":
        return gpt_agent.run(f"Tạo ảnh từ mô tả: {text_prompt}")
    elif model_type == "ollama":
        return ollama_agent.run(f"Tạo ảnh từ mô tả: {text_prompt}")
    elif model_type == "gemini":
        return gemini_agent.run(f"Tạo ảnh từ mô tả: {text_prompt}")
    else:
        return "Loại model không được hỗ trợ"

# Sửa lại các hàm pipeline để sử dụng agent phù hợp
def process_images_in_pipeline(images, model_type):
    results = []
    classification_info_list = []
    
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

        # Sử dụng agent phù hợp với loại model
        try:
            tool_result_str = process_image_with_agent(temp_path, model_type)
            print(f"tool_result_str : {tool_result_str}")
            
            # Xử lý kết quả tool
            if "src\\image" in tool_result_str or "src/image" in tool_result_str:
                pattern = r"(src[\\/]+image[\\/]+[^\s\"']+\.jpg)"
                match = re.search(pattern, tool_result_str)
                if match:
                    seg_path = match.group(1).strip().rstrip(".")
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
            seg_html = f"<strong> - Kết quả segmentation:</strong><br><img src='{seg_base64}' alt='segmentation' style='max-width:300px;'/>"
        else:
            seg_html = ""
            
        img_result_html = (
            f"<strong>Kết quả phân loại:</strong> - {classification_text}<br>"
            f"{seg_html}"
        )
        results.append(f"<strong>Kết quả xử lý ảnh : </strong><br>{img_result_html}")
    
    return results, classification_info_list

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

def check_need_rag(query_emb, threshold=0.5):
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
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Bạn là một chuyên gia về lĩnh vực y tế"},
                {"role": "user",   "content": refined_query}
            ],
            temperature=0,
            max_tokens=128
        )
        # API mới trả về .choices list
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Lỗi khi gọi GPT: {e}"

def rag_gpt_answer(refined_query: str, top_docs: list) -> str:
    context = "\n".join([doc.get("page_content", "") for doc in top_docs])
    prompt = (
        f"Các tài liệu liên quan:\n{context}\n\n"
        f"Câu hỏi: {refined_query}\n\n"
        "Hãy cung cấp câu trả lời ngắn gọn dựa trên các tài liệu trên."
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system",  "content": "Bạn là một chuyên gia về lĩnh vực da liễu"},
                {"role": "user",    "content": prompt}
            ],
            temperature=0,
            max_tokens=128
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Lỗi khi gọi GPT (RAG): {e}"

# Sửa lại các hàm pipeline để sử dụng tools
# def pipelinegpt(query_text: str, chat_history=None, images=[]):
#     """
#     - images: danh sách base64 image (dành cho ảnh y tế)
#     - chat_history: để summarize history nếu muốn
#     Luồng:
#       1. Xử lý ảnh y tế (nếu có) qua agent → MedicalImageTool
#       2. Xử lý image generation nếu query muốn tạo ảnh
#       3. Xử lý văn bản: RAG (BM25 rồi vector) hoặc normal GPT
#     """
#     print("PipelineGPT")
#     results = []

#     # 1. XỬ LÝ ẢNH Y TẾ
#     if images:
#         image_results, classification_info_list = process_images_in_pipeline(images, "gpt-4")
#         results.extend(image_results)
#     else:
#         classification_info_list = []

#     # 2. XỬ LÝ TẠO ẢNH (nếu cần)
#     text_lower = (query_text or "").lower()
#     keywords = ["tạo ảnh", "sinh ảnh", "generate image", "vẽ", "minh họa", "illustrate"]
#     if any(kw in text_lower for kw in keywords):
#         try:
#             gen_tool = ImageGenerationTool()
#             print(query_text)
#             html = gen_tool._run(query_text)
#             results.append(f"<strong>Kết quả tạo ảnh:</strong><br>{html}")
#         except Exception as e:
#             results.append(f"<strong>Lỗi khi tạo ảnh:</strong><br>{e}")
#         return "<br><br>".join(results)

#     # 3. XỬ LÝ VĂN BẢN
#     if query_text:
#         # 3.1. Gắn thêm context từ ảnh y tế
#         refined_query = query_text
#         if classification_info_list:
#             extra = "; ".join(classification_info_list)
#             refined_query = f"{query_text}\nThông tin bổ sung từ ảnh y tế: {extra}"

#         # 3.2. Gắn thêm lịch sử hội thoại (nếu cần)
#         if chat_history:
#             context = summarize_history(chat_history, query_text)
#             refined_query = f"Lịch sử:\n{context}\n\nCâu hỏi mới: {refined_query}"

#         # 3.3. Tính embedding và quyết định RAG hay không
#         query_emb = get_phobert_embedding(refined_query)
#         need_rag = check_need_rag(query_emb, threshold=0.5)

#         if not need_rag:
#             # Dùng GPT thuần
#             text_result = normal_gpt_answer(refined_query)
#         else:
#             # --- RAG: VECTOR (cosine) trước, rồi BM25 ---
#             # 3.4. Lấy lên tới 20 doc từ vector store
#             all_docs = get_query_results(query_emb, collection, limit=20)

#             if not all_docs:
#                 text_result = "[RAG] Không tìm thấy tài liệu phù hợp."
#             else:
#                 # 3.5. Sort theo cosine similarity, chọn top K (ví dụ 10)
#                 sorted_by_cosine = sorted(
#                     all_docs,
#                     key=lambda d: calc_cosine_similarity(query_emb, d["embedding"]),
#                     reverse=True
#                 )
#                 top_vec_docs = sorted_by_cosine[:10]

#                 # 3.6. Trên 10 doc này, chạy BM25 để chọn top 2
#                 bm25_docs = bm25_rerank(refined_query, top_vec_docs, top_n=2)

#                 if not bm25_docs:
#                     text_result = "[RAG] Không có tài liệu sau BM25."
#                 else:
#                     # 3.7. Dùng 2 doc đã lọc cho RAG
#                     text_result = rag_gpt_answer(refined_query, bm25_docs)

#                     # 3.7. Làm mịn câu trả lời lần cuối
#                     final_query = f"{refined_query}\n\nThông tin bổ sung từ tài liệu:\n{text_result}"
#                     final_text_result = normal_gpt_answer(final_query)
#                     results.append(f"<strong>Kết quả văn bản:</strong><br>{final_text_result}")

#                 return "<br><br>".join(results)
    
def pipelinegpt(query_text: str, chat_history=None, images=[]):
    """
    - images: danh sách base64 image (dành cho ảnh y tế)
    - chat_history: để summarize history nếu muốn
    Luồng:
      1. Xử lý ảnh y tế (nếu có) qua agent → MedicalImageTool
      2. Xử lý image generation nếu query muốn tạo ảnh
      3. Xử lý văn bản: RAG (vector→BM25) hoặc normal GPT
    """
    results = []
    classification_info_list = []

    # 1. XỬ LÝ ẢNH Y TẾ
    if images:
        image_results, classification_info_list = process_images_in_pipeline(images, "gpt-4")
        results.extend(image_results)

    # 2. XỬ LÝ TẠO ẢNH (nếu cần)
    # text_lower = (query_text or "").lower()
    # keywords = ["tạo ảnh", "sinh ảnh", "generate image", "vẽ", "minh họa", "illustrate"]
    # if any(kw in text_lower for kw in keywords):
    #     try:
    #         gen_tool = ImageGenerationTool()
    #         print(query_text)
    #         html = gen_tool._run(query_text)
    #         results.append(f"<strong>Kết quả tạo ảnh:</strong><br>{html}")
    #     except Exception as e:
    #         results.append(f"<strong>Lỗi khi tạo ảnh:</strong><br>{e}")
    #     return "<br><br>".join(results)
    text_lower = (query_text or "").lower()
    keywords = ["tạo ảnh", "sinh ảnh", "generate image", "vẽ", "minh họa", "illustrate"]
    if any(kw in text_lower for kw in keywords):
        try:
            tool_result_str = gpt_agent.run(query_text)

            # tìm đường dẫn .png trong output
            m = re.search(r"(src[\\/]+image[\\/]+[^\s\"']+\.png)", tool_result_str)
            if m:
                img_path = m.group(1).replace("\\", "/")
                b64 = image_file_to_base64(img_path)
                img_html = f"<img src='{b64}' style='max-width:300px;'/>"
                results.append(f"<strong>Kết quả tạo ảnh:</strong><br>{img_html}")
            else:
                # không tìm thấy path, show nguyên text
                results.append(f"<strong>Kết quả tạo ảnh:</strong><br>{tool_result_str}")

        except Exception as e:
            results.append(f"<strong>Lỗi khi tạo ảnh qua agent:</strong><br>{e}")
        return "<br><br>".join(results)
    # 3. XỬ LÝ VĂN BẢN
    if query_text:
        # 3.1. Gắn thêm context từ ảnh y tế
        refined_query = query_text
        if classification_info_list:
            refined_query += "\nThông tin bổ sung từ ảnh y tế: " + "; ".join(classification_info_list)

        # 3.2. Gắn thêm lịch sử hội thoại
        if chat_history:
            context = summarize_history(chat_history, query_text)
            refined_query = f"Lịch sử:\n{context}\n\nCâu hỏi mới: {refined_query}"

        # 3.3. Tính embedding và quyết định RAG
        query_emb = get_phobert_embedding(refined_query)
        need_rag = check_need_rag(query_emb, threshold=0.5)

        if not need_rag:
            # GPT thuần
            text_result = normal_gpt_answer(refined_query)
        else:
            # --- RAG: VECTOR → BM25 ---
            all_docs = get_query_results(query_emb, collection, limit=20)

            # DEBUG: in số lượng và keys của các doc
            print(f"[DEBUG] Lấy được {len(all_docs)} docs từ vector store")
            for i, doc in enumerate(all_docs[:5]):
                print(f"[DEBUG] Doc {i} keys: {list(doc.keys())}")
                if "embedding" in doc:
                    print(f"[DEBUG]  - embedding length: {len(doc['embedding'])}")

            if not all_docs:
                text_result = "[RAG] Không tìm thấy tài liệu phù hợp."
            else:
                # 1) Sort theo cosine, lấy top 10
                sorted_by_cosine = sorted(
                    all_docs,
                    key=lambda d: calc_cosine_similarity(query_emb, d["embedding"]),
                    reverse=True
                )
                top_vec_docs = sorted_by_cosine[:10]

                # DEBUG: in keys của top_vec_docs
                print(f"[DEBUG] Top 10 docs sau cosine:")
                for i, doc in enumerate(top_vec_docs):
                    print(f"[DEBUG]  Doc {i} keys: {list(doc.keys())}")

                # 2) Trên 10 doc, chạy BM25 để chọn 2
                bm25_docs = bm25_rerank(refined_query, top_vec_docs, top_n=2)
                if not bm25_docs:
                    text_result = "[RAG] Không có tài liệu sau BM25."
                else:
                    text_result = rag_gpt_answer(refined_query, bm25_docs)

        # 3.4. Làm mịn câu trả lời cuối cùng
        final_prompt = (
            f"{refined_query}\n\nThông tin bổ sung từ tài liệu:\n{text_result}"
        )
        final_text = normal_gpt_answer(final_prompt)
        results.append(f"<strong>Kết quả văn bản:</strong><br>{final_text}")

    return "<br><br>".join(results)

def ollama_answer(prompt: str, model="phi3") -> str:
    """
    Gửi prompt đến Ollama local và nhận phản hồi.
    """
    try:
        url = f"{OLLAMA_BASE_URL}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7
            }
        }
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "Không nhận được phản hồi từ Ollama")
        else:
            return f"Lỗi khi gọi Ollama: HTTP {response.status_code}"
    except Exception as e:
        return f"Lỗi khi gọi Ollama: {str(e)}"

def rag_ollama_answer(refined_query: str, top_docs: list, model="phi3") -> str:
    """
    Sử dụng Ollama với RAG
    """
    context = "\n".join([doc.get("page_content", "") for doc in top_docs])
    prompt = (
        f"Các tài liệu liên quan:\n{context}\n\n"
        f"Câu hỏi: {refined_query}\n\n"
        "Hãy cung cấp câu trả lời ngắn gọn dựa trên các tài liệu trên."
    )
    print(prompt)
    return ollama_answer(prompt, model)

# Tương tự sửa lại hàm pipelineOllama và pipelineGemini
# def pipelineOllama(query_text: str, chat_history=None, images=[], model="phi3"):
#     results = []
    
#     # Xử lý ảnh (nếu có)
#     if images:
#         image_results, classification_info_list = process_images_in_pipeline(images, "ollama")
#         results.extend(image_results)
#     else:
#         classification_info_list = []
    
#     # Xử lý văn bản (nếu có)
#     if query_text:
#         # Kiểm tra nếu đây là yêu cầu tạo ảnh
#         if "tạo ảnh" in query_text.lower() or "sinh ảnh" in query_text.lower() or "generate image" in query_text.lower():
#             try:
#                 image_gen_result = generate_image_with_agent(query_text, "ollama")
#                 results.append(f"<strong>Kết quả tạo ảnh:</strong><br>{image_gen_result}")
#             except Exception as e:
#                 results.append(f"<strong>Lỗi khi tạo ảnh:</strong><br>{str(e)}")
#         else:
#             # Xử lý văn bản thông thường
#             if classification_info_list:
#                 classification_info = "; ".join(classification_info_list)
#                 query_text = f"{query_text}\nThông tin bổ sung từ hình ảnh: {classification_info}"
            
#             refined_query = query_text
#             query_emb = get_phobert_embedding(refined_query)
#             need_rag = check_need_rag(query_emb, threshold=0.5)
            
#             if chat_history:
#                 context = summarize_history(chat_history, query_text)
#                 refined_query = f"Lịch sử hội thoại:\n{context}\nCâu hỏi mới: {query_text}"
            
#             print(f"[DEBUG] refined_query ban đầu:\n{refined_query}")
            
#             if not need_rag:
#                 print("[INFO] Similarity < 0.9 => Dùng model bình thường")
#                 text_result = ollama_answer(refined_query, model)
#             else:
#                 print("[INFO] Similarity >= 0.9 => Dùng RAG pipeline")
#                 vec_docs = get_query_results(query_emb, collection, limit=5)
#                 print("[INFO] vecdoc đã xong:", vec_docs)
#                 if not vec_docs:
#                     text_result = "[RAG] Không tìm thấy tài liệu phù hợp."
#                 else:
#                     top_docs = bm25_rerank(refined_query, vec_docs, top_n=2)
#                     print("[INFO] BM25 đã xong:", top_docs)
#                     if not top_docs:
#                         text_result = "[RAG] Không có tài liệu sau BM25."
#                     else:
#                         text_result = rag_ollama_answer(refined_query, top_docs, model)
            
#             final_query = f"{query_text}\nThông tin bổ sung từ tài liệu:\n{text_result}"
#             final_text_result = ollama_answer(final_query, model)
#             results.append(f"<strong>Kết quả văn bản:</strong><br>{final_text_result}")

#     return "<br><br>".join(results)
def pipelineOllama(query_text: str, chat_history=None, images=[], model="phi3"):
    results = []
    
    # 1. XỬ LÝ ẢNH Y TẾ (nếu có)
    if images:
        image_results, classification_info_list = process_images_in_pipeline(images, "ollama")
        results.extend(image_results)
    else:
        classification_info_list = []
    
    # 2. XỬ LÝ TẠO ẢNH (nếu cần)
    text_lower = (query_text or "").lower()
    keywords = ["tạo ảnh", "sinh ảnh", "generate image", "vẽ", "minh họa", "illustrate"]
    if any(kw in text_lower for kw in keywords):
        try:
            # Gọi agent để tạo ảnh
            tool_result_str = generate_image_with_agent(query_text, "ollama")
            
            # tìm đường dẫn .png trong output
            m = re.search(r"(src[\\/]+image[\\/]+[^\s\"']+\.png)", tool_result_str)
            if m:
                img_path = m.group(1).replace("\\", "/")
                b64 = image_file_to_base64(img_path)
                img_html = f"<img src='{b64}' style='max-width:300px;'/>"
                results.append(f"<strong>Kết quả tạo ảnh:</strong><br>{img_html}")
            else:
                # fallback: show nguyên text
                results.append(f"<strong>Kết quả tạo ảnh:</strong><br>{tool_result_str}")

        except Exception as e:
            results.append(f"<strong>Lỗi khi tạo ảnh qua agent:</strong><br>{e}")
        return "<br><br>".join(results)
    
    # 3. XỬ LÝ VĂN BẢN (RAG hoặc GPT thuần)
    if query_text:
        refined_query = query_text
        if classification_info_list:
            refined_query += "\nThông tin bổ sung từ hình ảnh: " + "; ".join(classification_info_list)
        if chat_history:
            context = summarize_history(chat_history, query_text)
            refined_query = f"Lịch sử hội thoại:\n{context}\nCâu hỏi mới: {query_text}"

        emb = get_phobert_embedding(refined_query)
        need_rag = check_need_rag(emb, threshold=0.5)

        if not need_rag:
            text_result = ollama_answer(refined_query, model)
        else:
            vec_docs = get_query_results(emb, collection, limit=5)
            if not vec_docs:
                text_result = "[RAG] Không tìm thấy tài liệu phù hợp."
            else:
                top_docs = bm25_rerank(refined_query, vec_docs, top_n=2)
                if not top_docs:
                    text_result = "[RAG] Không có tài liệu sau BM25."
                else:
                    text_result = rag_ollama_answer(refined_query, top_docs, model)

        final_query = f"{query_text}\nThông tin bổ sung từ tài liệu:\n{text_result}"
        final_text = ollama_answer(final_query, model)
        results.append(f"<strong>Kết quả văn bản:</strong><br>{final_text}")

    return "<br><br>".join(results)

def gemini_answer(prompt: str) -> str:
    """
    Gửi prompt đến Gemini và nhận phản hồi.
    """
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Lỗi khi gọi Gemini: {str(e)}"

def rag_gemini_answer(refined_query: str, top_docs: list) -> str:
    """
    Sử dụng Gemini với RAG
    """
    context = "\n".join([doc.get("page_content", "") for doc in top_docs])
    prompt = (
        f"Các tài liệu liên quan:\n{context}\n\n"
        f"Câu hỏi: {refined_query}\n\n"
        "Hãy cung cấp câu trả lời ngắn gọn dựa trên các tài liệu trên."
    )
    print(prompt)
    return gemini_answer(prompt)

# def pipelineGemini(query_text: str, chat_history=None, images=[]):
#     results = []
    
#     # Xử lý ảnh (nếu có)
#     if images:
#         image_results, classification_info_list = process_images_in_pipeline(images, "gemini")
#         results.extend(image_results)
#     else:
#         classification_info_list = []
    
#     # Xử lý văn bản (nếu có)
#     if query_text:
#         # Kiểm tra nếu đây là yêu cầu tạo ảnh
#         if "tạo ảnh" in query_text.lower() or "sinh ảnh" in query_text.lower() or "generate image" in query_text.lower():
#             try:
#                 image_gen_result = generate_image_with_agent(query_text, "gemini")
#                 results.append(f"<strong>Kết quả tạo ảnh:</strong><br>{image_gen_result}")
#             except Exception as e:
#                 results.append(f"<strong>Lỗi khi tạo ảnh:</strong><br>{str(e)}")
#         else:
#             # Xử lý văn bản thông thường
#             if classification_info_list:
#                 classification_info = "; ".join(classification_info_list)
#                 query_text = f"{query_text}\nThông tin bổ sung từ hình ảnh: {classification_info}"
            
#             refined_query = query_text
#             query_emb = get_phobert_embedding(refined_query)
#             need_rag = check_need_rag(query_emb, threshold=0.5)
            
#             if chat_history:
#                 context = summarize_history(chat_history, query_text)
#                 refined_query = f"Lịch sử hội thoại:\n{context}\nCâu hỏi mới: {query_text}"
            
#             print(f"[DEBUG] refined_query ban đầu:\n{refined_query}")
            
#             if not need_rag:
#                 print("[INFO] Similarity < 0.9 => Dùng model bình thường")
#                 text_result = gemini_answer(refined_query)
#             else:
#                 print("[INFO] Similarity >= 0.9 => Dùng RAG pipeline")
#                 vec_docs = get_query_results(query_emb, collection, limit=5)
#                 print("[INFO] vecdoc đã xong:", vec_docs)
#                 if not vec_docs:
#                     text_result = "[RAG] Không tìm thấy tài liệu phù hợp."
#                 else:
#                     top_docs = bm25_rerank(refined_query, vec_docs, top_n=2)
#                     print("[INFO] BM25 đã xong:", top_docs)
#                     if not top_docs:
#                         text_result = "[RAG] Không có tài liệu sau BM25."
#                     else:
#                         text_result = rag_gemini_answer(refined_query, top_docs)
            
#             final_query = f"{query_text}\nThông tin bổ sung từ tài liệu:\n{text_result}"
#             final_text_result = gemini_answer(final_query)
#             results.append(f"<strong>Kết quả văn bản:</strong><br>{final_text_result}")

#     return "<br><br>".join(results)
def pipelineGemini(query_text: str, chat_history=None, images=[]):
    results = []
    
    # 1. XỬ LÝ ẢNH Y TẾ (nếu có)
    if images:
        image_results, classification_info_list = process_images_in_pipeline(images, "gemini")
        results.extend(image_results)
    else:
        classification_info_list = []
    
    # 2. XỬ LÝ TẠO ẢNH (nếu cần)
    text_lower = (query_text or "").lower()
    keywords = ["tạo ảnh", "sinh ảnh", "generate image", "vẽ", "minh họa", "illustrate"]
    if any(kw in text_lower for kw in keywords):
        try:
            tool_result_str = generate_image_with_agent(query_text, "gemini")
            
            m = re.search(r"(src[\\/]+image[\\/]+[^\s\"']+\.png)", tool_result_str)
            if m:
                img_path = m.group(1).replace("\\", "/")
                b64 = image_file_to_base64(img_path)
                img_html = f"<img src='{b64}' style='max-width:300px;'/>"
                results.append(f"<strong>Kết quả tạo ảnh:</strong><br>{img_html}")
            else:
                results.append(f"<strong>Kết quả tạo ảnh:</strong><br>{tool_result_str}")

        except Exception as e:
            results.append(f"<strong>Lỗi khi tạo ảnh qua agent:</strong><br>{e}")
        return "<br><br>".join(results)
    
    # 3. XỬ LÝ VĂN BẢN (RAG hoặc GPT thuần)
    if query_text:
        refined_query = query_text
        if classification_info_list:
            refined_query += "\nThông tin bổ sung từ hình ảnh: " + "; ".join(classification_info_list)
        if chat_history:
            context = summarize_history(chat_history, query_text)
            refined_query = f"Lịch sử hội thoại:\n{context}\nCâu hỏi mới: {query_text}"

        emb = get_phobert_embedding(refined_query)
        need_rag = check_need_rag(emb, threshold=0.5)

        if not need_rag:
            text_result = gemini_answer(refined_query)
        else:
            vec_docs = get_query_results(emb, collection, limit=5)
            if not vec_docs:
                text_result = "[RAG] Không tìm thấy tài liệu phù hợp."
            else:
                top_docs = bm25_rerank(refined_query, vec_docs, top_n=2)
                if not top_docs:
                    text_result = "[RAG] Không có tài liệu sau BM25."
                else:
                    text_result = rag_gemini_answer(refined_query, top_docs)

        final_query = f"{query_text}\nThông tin bổ sung từ tài liệu:\n{text_result}"
        final_text = gemini_answer(final_query)
        results.append(f"<strong>Kết quả văn bản:</strong><br>{final_text}")

    return "<br><br>".join(results)

# --------------------------
# ROUTES FLASK
# --------------------------
def get_model_response(model, query, chat_history=None, images=[]):
    if model == "gpt-4":
        return pipelinegpt(query, chat_history, images)
    elif model == "ollama":
        return pipelineOllama(query, chat_history, images, model="phi3")
    elif model == "gemini":
        return pipelineGemini(query, chat_history, images)
    else:
        return "Mô hình không hợp lệ."

app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'src', 'image')
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
 
@app.route("/", methods=["GET", "POST"])
def index():
    # Kiểm tra và khởi tạo session nếu chưa có
    if "messages" not in session:
        session["messages"] = [{"role": "assistant", "content": "Xin chào, tôi có thể giúp gì?"}]
    
    if request.method == "POST":
        # Lấy dữ liệu JSON từ yêu cầu POST
        data = request.get_json()
        user_input = data.get("user_input", "").strip()
        images = data.get("images", [])
        model = data.get("model", "gpt-4")  # Mặc định chọn GPT-4

        # Thêm tin nhắn của người dùng vào session
        session["messages"].append({
            "role": "user",
            "content": user_input,
            "images": images,
            "model": model
        })

        # Xử lý yêu cầu từ người dùng và trả về kết quả
        reply = get_model_response(model, user_input, session["messages"], images)
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
