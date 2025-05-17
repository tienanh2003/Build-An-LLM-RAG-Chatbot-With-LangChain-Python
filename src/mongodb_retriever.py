import os
from pymongo import MongoClient
from dotenv import load_dotenv
from langchain.document_loaders import UnstructuredURLLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings

from config import MONGODB_CONNECTION_STRING, MONGODB_DB_NAME, MONGODB_COLLECTION_NAME, NUM_CLASSES, OLLAMA_BASE_URL

# Lớp để truy xuất dữ liệu từ MongoDB Atlas (dùng truy vấn text)
class MongoDBRAGRetriever:
    def __init__(self, connection_string, db_name, collection_name):
        self.connection_string = connection_string
        self.db_name = db_name
        self.collection_name = collection_name
        self.client = MongoClient(self.connection_string)
        self.db = self.client[self.db_name]
        self.collection = self.db[self.collection_name]
        print(f"Đã kết nối MongoDB: DB='{self.db_name}', Collection='{self.collection_name}'")
    
    def get_relevant_documents(self, query: str):
        try:
            cursor = self.collection.find({"$text": {"$search": query}})
            docs = list(cursor)
            if not docs:
                print("Không tìm thấy document phù hợp với truy vấn.")
            return [
                Document(
                    page_content=doc.get("page_content", ""),
                    metadata={
                        "url": doc.get("url", ""),
                        "title": doc.get("title", ""),
                        "chunk_index": doc.get("chunk_index", None)
                    }
                )
                for doc in docs
            ]
        except Exception as e:
            print(f"Lỗi khi truy xuất từ MongoDB: {e}")
            return [Document(page_content="Không tìm thấy dữ liệu phù hợp.", metadata={})]
    
    def close(self):
        self.client.close()

# def get_query_results(query_embedding, collection, limit):
#     """
#     Thực hiện truy vấn vector trên collection sử dụng chỉ mục "vector_index".
#     (Lưu ý: Cần cấu hình index vector trong MongoDB để hàm này hoạt động)
#     """
#     pipeline_vector = [
#         {
#             "$vectorSearch": {
#                 "index": "vector_index",
#                 "queryVector": query_embedding,
#                 "path": "embedding",
#                 "exact": True,
#                 "limit": limit
#             }
#         },
#         {
#             "$project": {
#                 "_id": 0,
#                 "url": 1,
#                 "page_content": 1,
#                 "score": {"$meta": "searchScore"}
#             }
#         }
#     ]
#     results_cursor = collection.aggregate(pipeline_vector)
#     results = [doc for doc in results_cursor]
#     return results
def get_query_results(query_embedding, collection, limit):
    pipeline_vector = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "queryVector": query_embedding,
                "path": "embedding",
                "exact": True,
                "limit": limit
            }
        },
        {
            "$project": {
                "_id": 0,
                "url": 1,
                "page_content": 1,
                "score": {"$meta": "searchScore"},
                "embedding": 1        # <-- Thêm trường này
            }
        }
    ]
    results_cursor = collection.aggregate(pipeline_vector)
    return [doc for doc in results_cursor]


def ingest_urls_to_mongo(urls, connection_string, db_name, collection_name):
    """
    Load nội dung từ các URL bằng UnstructuredURLLoader, chia nhỏ văn bản (chunk) với overlap,
    tính embedding (PhoBERT) và lưu vào MongoDB.
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # đảm bảo import lại
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    client = MongoClient(connection_string)
    db = client[db_name]
    collection = db[collection_name]
    
    # Giả sử bạn đã định nghĩa hàm get_phobert_embedding trong file khác (ở main.py hoặc module chung)
    # Nếu chưa, bạn cần nhập hàm đó vào đây.
    from main import get_phobert_embedding  # Ví dụ: nếu bạn lưu get_phobert_embedding trong main.py
    
    for url in urls:
        try:
            loader = UnstructuredURLLoader(urls=[url])
            docs = loader.load()
            if not docs:
                print(f"Không load được nội dung từ {url}")
                continue
            for doc in docs:
                content = doc.page_content
                # Làm sạch HTML
                from bs4 import BeautifulSoup
                content = BeautifulSoup(content, "html.parser").get_text(separator=" ", strip=True)
                content = content.encode("utf-8", "ignore").decode("utf-8")
                chunks = text_splitter.split_text(content)
                if not chunks:
                    print(f"Không có chunk nào ở {url}")
                    continue
                for idx, chunk in enumerate(chunks):
                    if not chunk.strip():
                        print(f"Bỏ qua chunk rỗng (chunk {idx}) ở {url}")
                        continue
                    try:
                        embedding = get_phobert_embedding(chunk)
                        doc_dict = {
                            "url": url,
                            "chunk_index": idx,
                            "page_content": chunk,
                            "metadata": doc.metadata,
                            "embedding": embedding
                        }
                        collection.insert_one(doc_dict)
                        print(f"Đã chèn chunk {idx} từ {url}")
                    except Exception as embed_err:
                        print(f"Lỗi tạo embedding chunk {idx} từ {url}: {embed_err}")
        except Exception as e:
            print(f"Lỗi khi ingest {url}: {e}")
    
    client.close()
    print("Hoàn thành ingest dữ liệu vào MongoDB.")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()  # Đọc biến môi trường nếu có
    connection_string = os.getenv("MONGODB_CONNECTION_STRING") or "mongodb+srv://<username>:<password>@<cluster-address>/test?retryWrites=true&w=majority"
    db_name = "your_database"
    collection_name = "your_collection"
    # Danh sách URL mẫu (bạn có thể thay đổi)
    urls = [
        "https://www.vinmec.com/vie/benh/ung-thu-hac-to-da-4723",
        "https://www.vinmec.com/vie/bai-viet/u-hac-lanh-tinh-cua-da-vi",
        "https://www.fvhospital.com/tin-suc-khoe/ung-thu-sac-to-la-gi-hieu-dung-ve-ung-thu-hac-to-va-benh-ly-sac-to-da/",
        "https://www.vinmec.com/vie/bai-viet/ung-thu-bieu-mo-te-bao-day-chan-doan-va-dieu-tri-vi#:~:text=Ung%20th%C6%B0%20bi%E1%BB%83u%20m%C3%B4%20t%E1%BA%BF%20b%C3%A0o%20%C4%91%C3%A1y%20(Basal%20cell%20carcinoma,s%C3%A2u%20nh%E1%BA%A5t%20c%E1%BB%A7a%20bi%E1%BB%83u%20m%C3%B4.",
        "https://www.vinmec.com/vie/benh/ung-thu-bieu-mo-te-bao-day-4315",
        "https://tamanhhospital.vn/ung-thu-bieu-mo-te-bao-day/",
        "https://suckhoedoisong.vn/ung-thu-bieu-mo-te-bao-day-tai-phat-va-phuong-phap-chua-hieu-qua-16923011211073782.htm",
        "https://www.vinmec.com/vie/bai-viet/dau-hieu-da-bi-day-sung-anh-nang-vi",
        "https://www.nhathuocankhang.com/benh/benh-day-sung-quang-hoa",
        "https://nhathuoclongchau.com.vn/benh/day-sung-anh-sang-520.html",
        "https://www.vinmec.com/vie/bai-viet/nhung-dieu-can-biet-ve-chung-day-sung-tiet-ba-vi",
        "https://www.vinmec.com/vie/bai-viet/day-sung-da-dau-nguyen-nhan-va-trieu-chung-vi",
        "https://nhathuoclongchau.com.vn/bai-viet/benh-day-sung-da-dau-nguyen-nhan-trieu-chung-va-cach-dieu-tri.html",
        "https://www.vinmec.com/vie/bai-viet/tim-hieu-ve-u-xo-kinh-mun-thit-vi",
        "https://www.vinmec.com/vie/bai-viet/noi-u-va-buou-bieu-hien-dieu-gi-tren-da-cua-ban-vi",
        "https://www.vinmec.com/vie/bai-viet/tim-hieu-not-ruoi-la-gi-va-cac-loai-not-ruoi-vi",
        "https://www.vinmec.com/vie/bai-viet/not-ruoi-hinh-thanh-nhu-the-nao-vi",
        "https://tamanhhospital.vn/not-ruoi/",
        "https://medlatec.vn/tin-tuc/vet-thuong-mach-mau-la-gi-nguyen-tac-va-ky-thuat-so-cuu-dung-cach",
        "https://nhathuoclongchau.com.vn/bai-viet/vo-mach-mau-duoi-da-nguyen-nhan-va-cach-dieu-tri-72683.html",
        "https://www.vinmec.com/vie/bai-viet/viem-mach-ngoai-da-nhung-dieu-can-biet-vi",
        "https://hellobacsi.com/suc-khoe/trieu-chung/vo-mach-mau-duoi-da/"
    ]
    ingest_urls_to_mongo(urls, connection_string, db_name, collection_name)
