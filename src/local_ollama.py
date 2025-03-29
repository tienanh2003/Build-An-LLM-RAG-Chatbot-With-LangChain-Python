import torch
from langchain.tools import BaseTool
from langchain.tools.retriever import create_retriever_tool
from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
# Cập nhật theo khuyến nghị
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import OpenAIEmbeddings


from config import MONGODB_CONNECTION_STRING, MONGODB_DB_NAME, MONGODB_COLLECTION_NAME, NUM_CLASSES, OLLAMA_BASE_URL

# --- TOOL FUNCTION CALLING CHO ẢNH ---
class ClassificationTool(BaseTool):
    name: str = "classify_image"
    description: str = "Gọi hàm xử lý ảnh bằng model ClassificationModelLoader để trả về predicted_class và confidence score."

    def _run(self, image_data: bytes):
        # TODO: Thay thế bằng logic xử lý thực tế
        return {"predicted_class": "predicted_class", "confidence": 0.85}

    async def _arun(self, image_data: bytes):
        raise NotImplementedError("Async not implemented")

class SegmentationTool(BaseTool):
    name: str = "segment_image"
    description: str = "Gọi hàm xử lý ảnh bằng model SegmentationModelLoader để trả về segmented_output."

    def _run(self, image_data: bytes):
        # TODO: Thay thế bằng logic xử lý thực tế cho segmentation.
        return {"segmented_output": "segmented_output"}

    async def _arun(self, image_data: bytes):
        raise NotImplementedError("Async not implemented")

class BCLIPEmbeddingTool(BaseTool):
    name: str = "bclip_embed_image"
    description: str = "Dùng model BCLIP để embedding ảnh và trả về chuỗi các số mô phỏng embedding + text embedding"
    
    def _run(self, image_data: bytes):
        # TODO: Thay thế bằng logic gọi model BCLIP thực tế để lấy embedding vector.
        return {"embedding_str": "0.1,0.2,0.3,0.4,0.5"}

    async def _arun(self, image_data: bytes):
        raise NotImplementedError("Async not implemented")

# --- Xây dựng Agent với tích hợp function calling ---
class LocalOllamaAgent:
    """
    Tích hợp LLM Ollama với các tool: classification, segmentation và BCLIP để hỗ trợ RAG từ MongoDB.
    """
    def __init__(self, mongodb_retriever, classification_model, segmentation_model, confidence_threshold=0.8):
        self.mongodb_retriever = mongodb_retriever
        self.classification_model = classification_model
        self.segmentation_model = segmentation_model
        self.confidence_threshold = confidence_threshold
        self.agent_executor = self._create_agent_executor()
    
    def _custom_retriever_tool(self, input_data):
        text_input = input_data.get("text", "")
        image_data = input_data.get("image")
        image_result = ""
        confidence = 0.0
        embedding_str = ""
        
        if image_data is not None:
            classification_response = ClassificationTool()._run(image_data)
            segmentation_response = SegmentationTool()._run(image_data)
            predicted_class = classification_response.get("predicted_class", "")
            confidence = classification_response.get("confidence", 0.0)
            seg_output = segmentation_response.get("segmented_output", "")
            image_result = f"Classification: {predicted_class}; Segmentation: {seg_output}"
            
            bclip_response = BCLIPEmbeddingTool()._run(image_data)
            embedding_str = bclip_response.get("embedding_str", "")
        
        combined_query = f"{text_input}\n{image_result}\nImageEmbedding: {embedding_str}".strip()
        
        if (image_data is not None and confidence >= self.confidence_threshold):
            return self.mongodb_retriever.get_relevant_documents(combined_query)
        else:
            return [Document(page_content="Thông tin đầu vào không đủ để thực hiện truy xuất chuyên sâu.",
                             metadata={"source": "fallback"})]
    
    def _create_agent_executor(self):
        retriever_tool = create_retriever_tool(
            self._custom_retriever_tool,
            "find_documents",
            "Search for information using classification, segmentation, and image embeddings from BCLIP."
        )
        classification_tool = ClassificationTool()
        segmentation_tool = SegmentationTool()
        bclip_tool = BCLIPEmbeddingTool()
        
        llm = ChatOllama(
            model="llama2",
            temperature=0,
            streaming=True
        )
        tools = [retriever_tool, classification_tool, segmentation_tool, bclip_tool]
        system = (
            "Bạn là một chatbot chuyên về y tế dự đoán bệnh ngoài da. "
            "Sử dụng function calling để xử lý ảnh với classification, segmentation và embedding ảnh (BCLIP). "
            "Nếu đầu vào đủ điều kiện thực hiện truy xuất từ MongoDB; nếu không, trả lời thông thường."
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)
        return AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    def get_agent_executor(self):
        return self.agent_executor

def get_llm_and_agent(mongodb_retriever, classification_model, segmentation_model):
    agent_instance = LocalOllamaAgent(mongodb_retriever, classification_model, segmentation_model)
    return agent_instance.get_agent_executor()

def llm_answer_with_ollama(prompt: str) -> str:
    """
    Gọi ChatOllama để trả lời prompt.
    """
    try:
        # Sử dụng model phi3 (ví dụ) với base_url nếu cần điều chỉnh
        llm = ChatOllama(
            model="phi3",
            base_url="http://localhost:11434",
            temperature=0.7,
            streaming=False
        )
        # response = llm.predict(prompt)
        response = llm.invoke(prompt)
        return response
    except Exception as e:
        print("Lỗi khi gọi llm.predict:", e)
        return "[Lỗi] Không thể gọi model."

def normal_model_answer(query: str) -> str:
    """
    Trả lời câu hỏi bằng ChatOllama (không dùng RAG).
    """
    prompt = f"Trả lời câu hỏi sau một cách ngắn gọn và rõ ràng:\nQuery: {query}"
    return llm_answer_with_ollama(prompt)

def rag_llm_answer(query: str, docs: list) -> str:
    """
    Trả lời câu hỏi dựa trên tài liệu liên quan (RAG).
    """
    combined_text = "\n\n".join([f"URL: {doc['url']}\nContent: {doc['page_content']}" for doc in docs])
    prompt = (
        f"Bạn là chuyên gia y tế hàng đầu trong lĩnh vực da liễu, hãy giúp tôi trả lời câu hỏi sau dựa trên các tài liệu liên quan.\n"
        f"Câu hỏi từ người dùng: {query}\n\n"
        f"Tài liệu tham khảo:\n{combined_text}\n\n"
        f"Trả lời một cách ngắn gọn và dễ hiểu."
    )
    return llm_answer_with_ollama(prompt)
