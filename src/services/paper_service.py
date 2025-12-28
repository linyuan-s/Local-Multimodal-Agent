import os
import shutil
from typing import List
from src.core.database import db
from src.core.model_loader import get_text_embedding
from src.core.processor import Processor
from sentence_transformers.util import cos_sim
import numpy as np

class PaperService:
    @staticmethod
    def add_paper(file_path: str, topics: List[str] = None, root_dir: str = None):
        """
        处理论文：提取 -> 嵌入 -> 分类 -> 存储 -> 移动
        :param root_dir: 如果提供，分类后的文件将移动到 root_dir/Topic 下，而不是 file_path 所在的相对目录下
        """
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} not found.")
            return

        filename = os.path.basename(file_path)
        print(f"Processing: {filename}...")

        # 1. 提取文本
        pages = Processor.extract_text_with_page(file_path)
        if not pages:
            print("Warning: No text extracted. Is it a scanned PDF?")
            return

        # 2. 准备 chunks
        chunks = Processor.chunk_text(pages)
        
        # 3. 准备摘要用于分类和特殊索引
        summary_text = Processor.extract_summary_candidate(pages)
        
        # 4. 自动分类逻辑 (Optimization: Cosine Similarity)
        predicted_topic = "Uncategorized"
        if topics:
            # 定义缩写映射，增强语义匹配准确度
            # Key: 用户输入的Topic (也是文件夹名), Value: 用于生成Embedding的完整描述
            topic_descriptions = {
                "SGG": "Scene Graph Generation in Computer Vision and Images",
                "RL": "Reinforcement Learning and Multi-Agent Systems",
                "Hypergraph": "Hypergraph Neural Networks and Relation Learning",
                "CV": "Computer Vision",
                "NLP": "Natural Language Processing"
            }
            
            # 准备用于 Embedding 的文本列表
            # 如果 topic 在映射中，用描述；否则直接用 topic 本身
            texts_to_score = [topic_descriptions.get(t, t) for t in topics]
            
            # 编码 Topics
            topic_embeddings = get_text_embedding(texts_to_score) # List[List[float]]
            # 编码 摘要
            summary_embedding = get_text_embedding(summary_text) # List[float]
            
            # 计算相似度
            import torch
            t_emb = torch.tensor(topic_embeddings)
            s_emb = torch.tensor(summary_embedding).unsqueeze(0)
            
            scores = cos_sim(s_emb, t_emb)[0] # shape (num_topics,)
            best_topic_idx = torch.argmax(scores).item()
            predicted_topic = topics[best_topic_idx] # 依然返回原始的短 Topic 用于文件夹命名
            print(f" -> Classified as: {predicted_topic} (Score: {scores[best_topic_idx]:.4f})")
        
        # 5. 生成所有 chunks 的 Embedding 并准备入库数据
        #  为了性能，可以批量生成
        texts_to_embed = [c["text"] for c in chunks]
        # 添加摘要作为单独的文档，权重更高(逻辑上，通过 is_summary 标记)
        texts_to_embed.append(summary_text) 
        
        all_embeddings = get_text_embedding(texts_to_embed)
        
        # 分离
        chunk_embeddings = all_embeddings[:-1]
        summary_embedding_final = all_embeddings[-1]
        
        collection = db.get_paper_collection()
        
        # 构造存入的数据
        ids = []
        metadatas = []
        documents = []
        
        # 处理普通 chunks
        for i, chunk in enumerate(chunks):
            chunk_id = f"{filename}_chunk_{chunk['chunk_id']}"
            ids.append(chunk_id)
            documents.append(chunk["text"])
            metadatas.append({
                "filename": filename,
                "path": file_path, # 注意：如果移动了文件，这里稍后需要更新，但在add之前我们还不知道移动后的路径？
                # 实际移动逻辑通常在存库之后或之前。为了简单，我们先移动文件，再存新路径。
                # 但这里我们先保留原始逻辑，稍后执行移动。
                "page_number": chunk["page_number"],
                "topic": predicted_topic,
                "is_summary": False
            })
            
        # 处理摘要 chunk
        ids.append(f"{filename}_summary")
        documents.append(summary_text)
        metadatas.append({
            "filename": filename,
            "path": file_path, 
            "page_number": 1, 
            "topic": predicted_topic,
            "is_summary": True
        })
        
        # 合并 embeddings
        final_embeddings = chunk_embeddings + [summary_embedding_final]
        
        # 6. 移动文件 (如果指定了 topics)
        final_path = file_path
        if topics and predicted_topic != "Uncategorized":
            # 如果指定了 root_dir，则移动到 root_dir/Topic
            # 否则移动到 当前文件目录/Topic
            base_dir = root_dir if root_dir else os.path.dirname(file_path)
            target_dir = os.path.join(base_dir, predicted_topic)
            os.makedirs(target_dir, exist_ok=True)
            target_path = os.path.join(target_dir, filename)
            
            # 如果文件已在目标位置，跳过；否则移动
            # 注意: Windows下路径可能大小写不敏感，但abspath比较是安全的
            if os.path.abspath(file_path).lower() != os.path.abspath(target_path).lower():
                try:
                    shutil.move(file_path, target_path)
                    print(f" -> Moved to: {target_path}")
                    final_path = target_path
                except Exception as e:
                    print(f" -> Move failed: {e}")
                    final_path = file_path # 回退
            for meta in metadatas:
                meta["path"] = final_path

        # 7. 存入 ChromaDB
        collection.add(
            ids=ids,
            embeddings=final_embeddings,
            metadatas=metadatas,
            documents=documents
        )
        print(f" -> Indexed {len(ids)} chunks.")

    @staticmethod
    def search_paper(query: str, top_k: int = 5):
        """
        搜索论文
        """
        print(f"Searching for: {query}")
        query_embedding = get_text_embedding(query)
        
        collection = db.get_paper_collection()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            # 可以过滤掉 is_summary (可选)，或者让 summary 排在前面
            # where={"is_summary": False} 
        )
        
        # 格式化输出
        # results 结构: {'ids': [[]], 'distances': [[]], 'metadatas': [[]], 'documents': [[]]}
        if not results['ids'][0]:
            print("No results found.")
            return

        print(f"\nTop {top_k} Results:")
        print("-" * 50)
        for i in range(len(results['ids'][0])):
            meta = results['metadatas'][0][i]
            doc = results['documents'][0][i]
            score = results['distances'][0][i] # Cosine distance (lower is better usually, depends on space)
            # ChromaDB cosine space: 1 - cosine_similarity. So lower is closer. 0 means identical.
            
            is_summary_tag = "[SUMMARY MATCH]" if meta.get("is_summary") else ""
            
            print(f"File: {meta['filename']} (Page {meta['page_number']}) {is_summary_tag}")
            print(f"Topic: {meta.get('topic', 'N/A')}")
            # print(f"Distance: {score:.4f}")
            print(f"Content: {doc[:200].replace(chr(10), ' ')}...") # 只显示前200字符
            print("-" * 50)
