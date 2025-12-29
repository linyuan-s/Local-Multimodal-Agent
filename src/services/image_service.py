import os
from PIL import Image
from src.core.database import db
from src.core.model_loader import get_image_embedding, get_text_embedding_for_clip
import glob

class ImageService:
    @staticmethod
    def index_images(folder_path: str):
        """
        索引指定文件夹下的所有图片
        """
        if not os.path.isdir(folder_path):
            # 支持单个文件
            if os.path.isfile(folder_path):
                files = [folder_path]
            else:
                print(f"Error: {folder_path} is not a valid path.")
                return
        else:
            # 扫描常见图片格式
            exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
            files = []
            for ext in exts:
                files.extend(glob.glob(os.path.join(folder_path, ext)))
        
        print(f"Found {len(files)} images to index.")
        
        collection = db.get_image_collection()
        
        for file_path in files:
            try:
                filename = os.path.basename(file_path)
                print(f"Indexing: {filename}...", end="", flush=True)
                
                # Load image
                image = Image.open(file_path).convert("RGB")
                
                # Get Embedding
                emb = get_image_embedding(image)
                
                # Add to DB
                collection.add(
                    ids=[filename], # Simple ID
                    embeddings=[emb],
                    metadatas=[{"filename": filename, "path": file_path}],
                    documents=[filename] # Chroma needs a document usually, just use filename
                )
                print(" Done.")
            except Exception as e:
                print(f" Failed: {e}")

    @staticmethod
    def search_image(query: str, top_k: int = 3):
        """
        以文搜图
        """
        print(f"Searching for image: '{query}'")
        
        # 使用 CLIP 的 Text Encoder 获取查询向量
        text_emb = get_text_embedding_for_clip(query)
        
        collection = db.get_image_collection()
        results = collection.query(
            query_embeddings=[text_emb],
            n_results=top_k
        )
        
        if not results['ids'][0]:
            print("No images found.")
            return

        print(f"\nTop {top_k} Matching Images:")
        print("-" * 50)
        for i in range(len(results['ids'][0])):
            meta = results['metadatas'][0][i]
            filename = meta['filename']
            path = meta['path']
            # score = results['distances'][0][i]
            
            print(f"Image: {filename}")
            print("-" * 50)

    @staticmethod
    def answer_question(image_path: str, question: str):
        """
        Visual Question Answering (VQA) using BLIP
        """
        try:
            from src.core.model_loader import ModelLoader
            model, processor = ModelLoader.get_blip_components()
            
            # Load Image
            # Check if path exists
            if not os.path.exists(image_path):
                return "Error: Image file not found."

            raw_image = Image.open(image_path).convert('RGB')
            
            # Prepare inputs
            # encoding
            inputs = processor(raw_image, question, return_tensors="pt")
            
            # Generate answer
            # We move inputs to the same device as model if we were using GPU, 
            # but here default is CPU or auto-handled by transformers if lucky.
            # Ideally: inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            out = model.generate(**inputs)
            answer = processor.decode(out[0], skip_special_tokens=True)
            
            return answer
        except Exception as e:
            print(f"Error in VQA: {e}")
            return f"Error generating answer: {str(e)}"
