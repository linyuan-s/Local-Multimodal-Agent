import typer
import sys
import os

# 确保 src 在路径中
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.services.paper_service import PaperService
from src.services.image_service import ImageService

app = typer.Typer(
    name="Local Multimodal AI Agent",
    help="智能文献与图像管理助手 CLI",
    add_completion=False
)

@app.command()
def add_paper(
    path: str = typer.Argument(..., help="PDF文件的路径"),
    topics: str = typer.Option(None, help="分类主题列表，用逗号分隔，例如 'SGG,Hypergraph,RL'")
):
    """
    添加论文到数据库，并根据 Topics 自动分类移动。
    """
    topic_list = None
    if topics:
        topic_list = [t.strip() for t in topics.split(",") if t.strip()]
    
    PaperService.add_paper(path, topic_list)

@app.command()
def search_paper(
    query: str = typer.Argument(..., help="搜索查询语句")
):
    """
    语义搜索论文。
    """
    PaperService.search_paper(query)

@app.command()
def index_image(
    path: str = typer.Argument(..., help="图片文件或文件夹路径")
):
    """
    索引一张图片或整个文件夹的图片。
    """
    ImageService.index_images(path)

@app.command()
def search_image(
    query: str = typer.Argument(..., help="图片描述")
):
    """
    以文搜图。
    """
    ImageService.search_image(query)

@app.command()
def ingest(
    folder_path: str = typer.Argument(..., help="要扫描的文件夹路径"),
    topics: str = typer.Option(None, help="分类主题列表 (仅对论文有效)")
):
    """
    [新增] 批量导入：自动递归扫描文件夹，识别 PDF 和图片并入库。
    """
    if not os.path.exists(folder_path):
        print(f"Error: Path {folder_path} does not exist.")
        return

    topic_list = None
    if topics:
        topic_list = [t.strip() for t in topics.split(",") if t.strip()]

    print(f"Scanning folder: {folder_path} ...")
    
    pdf_count = 0
    img_count = 0
    
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower()
            
            # 处理 PDF
            if ext == ".pdf":
                print(f"\n[Found PDF] {file}")
                try:
                    # 传入 folder_path 作为 root_dir，确保移动到主文件夹下的分类目录
                    PaperService.add_paper(file_path, topic_list, root_dir=folder_path)
                    pdf_count += 1
                except Exception as e:
                    print(f"Failed to process PDF {file}: {e}")
            
            # 处理图片
            elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                # ImageService.index_images 默认是处理整个 folder，这里我们也可以复用它，或者改一下
                # 现在的 ImageService.index_images 是接收 folder 或 file 的，所以可以直接传 file_path
                print(f"[Found Image] {file}")
                try:
                    ImageService.index_images(file_path)
                    img_count += 1
                except Exception as e:
                    print(f"Failed to process Image {file}: {e}")

    print("\n" + "="*50)
    print(f"Ingestion Complete.")
    print(f"Papers processed: {pdf_count}")
    print(f"Images processed: {img_count}")
    print("="*50)

if __name__ == "__main__":
    app()
