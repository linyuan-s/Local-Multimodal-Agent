from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer, BlipProcessor, BlipForQuestionAnswering
import torch

class ModelLoader:
    _text_model = None
    _clip_model = None
    _clip_processor = None
    _clip_tokenizer = None
    _blip_model = None
    _blip_processor = None

    @classmethod
    def get_text_model(cls):
        """Lazy load SentenceTransformer model"""
        if cls._text_model is None:
            print("Loading Text Embedding Model (all-MiniLM-L6-v2)...")
            cls._text_model = SentenceTransformer('all-MiniLM-L6-v2')
        return cls._text_model

    @classmethod
    def get_clip_components(cls):
        """Lazy load CLIP model and processor"""
        if cls._clip_model is None:
            print("Loading CLIP Model (openai/clip-vit-base-patch32)...")
            model_name = "openai/clip-vit-base-patch32"
            cls._clip_model = CLIPModel.from_pretrained(model_name)
            cls._clip_processor = CLIPProcessor.from_pretrained(model_name)
            cls._clip_tokenizer = CLIPTokenizer.from_pretrained(model_name)
        return cls._clip_model, cls._clip_processor, cls._clip_tokenizer

    @classmethod
    def get_blip_components(cls):
        """Lazy load BLIP VQA model"""
        if cls._blip_model is None:
            print("Loading BLIP Model (Salesforce/blip-vqa-base)...")
            model_name = "Salesforce/blip-vqa-base"
            cls._blip_processor = BlipProcessor.from_pretrained(model_name)
            cls._blip_model = BlipForQuestionAnswering.from_pretrained(model_name)
        return cls._blip_model, cls._blip_processor

# 便捷获取函数
def get_text_embedding(text):
    model = ModelLoader.get_text_model()
    # SentenceTransformers 返回的是 numpy array, 需要转 list 存入 ChromaDB
    # [FIX] 强制归一化，配合 ChromaDB 默认的 L2 距离使用，等效于 Cosine 相似度
    return model.encode(text, normalize_embeddings=True).tolist()

def get_image_embedding(image):
    model, processor, _ = ModelLoader.get_clip_components()
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    # 归一化并转 list
    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
    return image_features[0].tolist()

def get_text_embedding_for_clip(text):
    """用于以文搜图的文本 Embedding (使用 CLIP Text Encoder)"""
    model, _, tokenizer = ModelLoader.get_clip_components()
    inputs = tokenizer([text], padding=True, return_tensors="pt")
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    return text_features[0].tolist()
