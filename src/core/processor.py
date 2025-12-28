import fitz  # PyMuPDF
from typing import List, Dict, Tuple

class Processor:
    @staticmethod
    def extract_text_with_page(pdf_path: str) -> List[Tuple[int, str]]:
        """
        使用 PyMuPDF (fitz) 提取 PDF 文本，保留页码信息。
        返回: List[(page_number, page_text)]，页码从 1 开始。
        """
        doc = fitz.open(pdf_path)
        pages_content = []
        for i, page in enumerate(doc):
            text = page.get_text()
            # 简单的清理：去除多余空白
            text = " ".join(text.split())
            if text:
                pages_content.append((i + 1, text))
        return pages_content

    @staticmethod
    def chunk_text(pages_content: List[Tuple[int, str]], chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
        """
        将文本切分为 chunk，并尽量保持语义完整性（这里简化为按字符数切分）。
        同时确保每个 chunk 都能关联到起始页码。
        
        返回: List[{ "text": str, "page_number": int, "chunk_id": int }]
        """
        chunks = []
        current_chunk_id = 0
        
        for page_num, text in pages_content:
            # 如果单页内容过长，需要切分
            if len(text) > chunk_size:
                start = 0
                while start < len(text):
                    end = start + chunk_size
                    # 只要不是最后一段，都往后多取一点作为 overlap，或者按 limit 切
                    chunk_text = text[start:end]
                    
                    chunks.append({
                        "text": chunk_text,
                        "page_number": page_num,
                        "chunk_id": current_chunk_id
                    })
                    current_chunk_id += 1
                    start += (chunk_size - overlap)
            else:
                # 页面内容较少，直接作为一个 chunk
                chunks.append({
                    "text": text,
                    "page_number": page_num,
                    "chunk_id": current_chunk_id
                })
                current_chunk_id += 1
                
        return chunks

    @staticmethod
    def extract_summary_candidate(pages_content: List[Tuple[int, str]]) -> str:
        """
        尝试提取用于分类的摘要候补文本。
        通常取前两页的内容作为摘要分析的依据。
        """
        # 取前两页，最多 2000 字符
        candidate_text = ""
        for _, text in pages_content[:2]:
            candidate_text += text + " "
        return candidate_text[:2000]
