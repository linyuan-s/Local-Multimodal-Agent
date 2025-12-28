import chromadb
from chromadb.config import Settings
import os

class Database:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Database, cls).__new__(cls)
            # 初始化 ChromaDB Client (Persistent)
            # [FIX] 使用用户指定的纯英文路径 (分离存储，避免扫描时冲突)
            db_path = "D:/Multi_model/peizhi/chroma_db"
            if not os.path.exists(db_path):
                os.makedirs(db_path)
            print(f"Database Path: {db_path}")
            cls._instance.client = chromadb.PersistentClient(path=db_path)
            
            # 获取或创建 Collections
            # [FIX] 移除 hnsw:space: cosine，使用默认的 L2 距离。
            # 配合归一化的 Embedding，L2 距离排序与 Cosine 相似度完全一致，且更稳定。
            cls._instance.paper_collection = cls._instance.client.get_or_create_collection(
                name="papers"
            )
            
            cls._instance.image_collection = cls._instance.client.get_or_create_collection(
                name="images"
            )
        return cls._instance

    def get_paper_collection(self):
        return self.paper_collection

    def get_image_collection(self):
        return self.image_collection

# 全局数据库实例
db = Database()
