import streamlit as st
import os
import pandas as pd
from PIL import Image
import time

# 导入后端服务
# 注意：确保运行目录在项目根目录，或者添加 sys.path
import sys
sys.path.append(os.getcwd())

from src.services.paper_service import PaperService
from src.services.image_service import ImageService
from src.core.processor import Processor
from src.core.database import db

# --- 页面配置 ---
st.set_page_config(
    page_title="Multimodal AI Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 自定义 CSS 美化 ---
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        font-weight: bold;
    }
    .main-header {
        font-size: 2.5rem;
        color: #4B4B4B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .card {
        background-color: #f9f9f9;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
        margin-bottom: 15px;
        border-left: 5px solid #6C63FF;
        color: #333333; /* [FIX] 强制黑色字体，防止夜间模式看不见 */
    }
    .card h4 {
        color: #000000 !important;
        margin-top: 0;
    }
    .highlight {
        background-color: #fffacd;
        padding: 2px 5px;
        border-radius: 3px;
        color: #333333; /* [FIX] 高亮块内文字也强制深色 */
    }
</style>
""", unsafe_allow_html=True)

# --- 侧边栏导航 ---
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/brain.png", width=100)
    st.title("本地多模态助手")
    
    page = st.radio(
        "选择功能模块",
        ["📁 智能整理 (Auto-Org)", "🔍 文献深度搜索 (Deep Search)", "🖼️ 以文搜图 (Image Search)"],
        index=1 # 默认进搜索页
    )
    
    st.markdown("---")
    st.info(f"📚 Database Path:\n`{db.client._system.settings.require('persist_directory')}`")
    
    # 状态重置
    if st.button("清除缓存 / Reload"):
        st.cache_data.clear()
        st.rerun()

# --- 页面 A: 智能整理 ---
if page == "📁 智能整理 (Auto-Org)":
    st.markdown("<h1 class='main-header'>📁 智能文献与图像整理</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        folder_path = st.text_input("输入要扫描的文件夹路径", value="D:\\Multi_model\\test")
        topics_str = st.text_input("分类 Topics (逗号分隔)", value="SGG,Hypergraph,RL")
    
    with col2:
        st.write("") # Spacer
        st.write("")
        start_btn = st.button("🚀 开始自动清理与分类", type="primary")

    if start_btn:
        if not os.path.exists(folder_path):
            st.error("文件夹路径不存在！")
        else:
            status_text = st.empty()
            progress_bar = st.progress(0)
            
            files = []
            for root, _, fs in os.walk(folder_path):
                for f in fs:
                    files.append(os.path.join(root, f))
            
            total_files = len(files)
            processed_data = []
            
            status_text.write(f"🔍 发现 {total_files} 个文件，开始处理...")
            
            topic_list = [t.strip() for t in topics_str.split(",") if t.strip()]
            
            for idx, file_path in enumerate(files):
                filename = os.path.basename(file_path)
                ext = os.path.splitext(filename)[1].lower()
                
                status_text.write(f"Processing: **{filename}**")
                progress_bar.progress((idx + 1) / total_files)
                
                try:
                    predicted_topic = "N/A"
                    if ext == ".pdf":
                        # Hack: 重定向 stdout 来捕获分类结果 (通常不建议，但为了展示方便)
                        # 这里还是直接调用 Service 比较好，但 Service 没有返回值，只有 Print
                        # 我们先简单调用，假设成功。实际要获取分类结果需要改 Service 返回值。
                        # 为了演示效果，我们这里先执行，再看文件去哪了。
                        PaperService.add_paper(file_path, topic_list, root_dir=folder_path)
                        # 简单推断一下归类（根据新路径）
                        time.sleep(0.5) # 模拟处理时间
                        
                        # 检查文件被移到哪了
                        new_topic = "Unknown"
                        for t in topic_list:
                            if os.path.exists(os.path.join(folder_path, t, filename)):
                                new_topic = t
                                break
                        processed_data.append({
                            "Filename": filename,
                            "Type": "PDF",
                            "Topic": new_topic,
                            "Status": "✅ Success"
                        })
                        
                    elif ext in ['.jpg', '.png', '.jpeg']:
                        ImageService.index_images(file_path)
                        processed_data.append({
                            "Filename": filename,
                            "Type": "Image",
                            "Topic": "Image Index",
                            "Status": "✅ Indexed"
                        })
                    else:
                        processed_data.append({"Filename": filename, "Type": ext, "Topic": "-", "Status": "Skipped"})
                        
                except Exception as e:
                    processed_data.append({"Filename": filename, "Type": ext, "Topic": "Error", "Status": f"❌ {str(e)}"})
            
            progress_bar.progress(100)
            status_text.success("🎉 整理完成！")
            
            # 展示结果表格
            df = pd.DataFrame(processed_data)
            st.dataframe(df, use_container_width=True)
            
            # 图表统计
            st.subheader("📊 分类统计")
            if not df.empty and "Topic" in df.columns:
                chart_data = df["Topic"].value_counts()
                st.bar_chart(chart_data)

# --- 页面 B: 文献深度搜索 ---
elif page == "🔍 文献深度搜索 (Deep Search)":
    st.markdown("<h1 class='main-header'>🔍 深度语义搜索</h1>", unsafe_allow_html=True)

    query = st.text_input("", placeholder="💡 试着问: What is the core idea of Scene Graph Generation?", label_visibility="collapsed")
    st.markdown("---")

    if query:
        # 获取搜索结果
        results = db.get_paper_collection().query(
            query_embeddings=[Processor.get_text_embedding_safe(query)], # 需要一个小 helper 或者直接调 model_loader
            n_results=3,
            include=['documents', 'metadatas', 'distances']
        )
        
        # 布局
        c1, c2 = st.columns([1, 1])
        
        # 使用 Session State 记录当前选中的论文以便在右侧展示
        if "selected_paper" not in st.session_state:
            st.session_state.selected_paper = None
        
        with c1:
            st.subheader("📄 搜索结果")
            if not results['ids'][0]:
                st.warning("没有找到相关结果。")
            
            for i, doc in enumerate(results['documents'][0]):
                meta = results['metadatas'][0][i]
                score = 1 - results['distances'][0][i] # Cosine Distance -> Similarity (Approx)
                
                filename = meta.get('filename', 'Unknown')
                page_num = meta.get('page_number', 1)
                file_path = meta.get('path', '')
                
                # 卡片容器
                with st.container():
                    st.markdown(f"""
                    <div class="card">
                        <h4>📄 {filename}</h4>
                        <p class="highlight">...{doc[:200]}...</p>
                        <p style="font-size:0.8em; color:gray">
                            Score: {score:.4f} | Page: {page_num}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 预览按钮
                    if st.button(f"👁️ 预览第 {page_num} 页", key=f"preview_{i}"):
                        st.session_state.selected_paper = {
                            "path": file_path,
                            "page": page_num,
                            "doc": doc
                        }

        # 右侧预览区
        with c2:
            st.subheader("👁️ 实时阅读")
            if st.session_state.selected_paper:
                p_info = st.session_state.selected_paper
                st.info(f"正在查看: {os.path.basename(p_info['path'])} (第 {p_info['page']} 页)")
                
                # [MODIFIED] 用户要求移除预览图，仅显示文字
                st.markdown("**本页命中内容:**")
                st.info(p_info['doc'])
            else:
                st.markdown("""
                <div style="text-align: center; padding: 50px; color: gray;">
                    👈 点击左侧结果的“预览”按钮<br>在此处查看 PDF 原文
                </div>
                """, unsafe_allow_html=True)

# --- 页面 C: 以文搜图 ---
elif page == "🖼️ 以文搜图 (Image Search)":
    st.markdown("<h1 class='main-header'>🖼️ 图像搜索</h1>", unsafe_allow_html=True)
    
    img_query = st.text_input("", placeholder="💡 描述你想找的图片: A dog running on grass...", label_visibility="collapsed")
    
    if img_query:
        st.write(f"Searching for: **{img_query}**")
        
        # 搜索 (复用 ImageService 逻辑)
        # 以前的 ImageService 直接 print 了，我们需要稍微改一下或者直接在这里调 DB (更灵活)
        from src.core.model_loader import get_text_embedding_for_clip
        
        q_emb = get_text_embedding_for_clip(img_query)
        results = db.get_image_collection().query(
            query_embeddings=[q_emb],
            n_results=6,
            include=['metadatas', 'distances']
        )
        
        # 瀑布流展示 (每行3张)
        cols = st.columns(3)
        for i, meta in enumerate(results['metadatas'][0]):
            img_path = meta.get('path')
            score = 1 - results['distances'][0][i]
            
            with cols[i % 3]:
                if os.path.exists(img_path):
                    st.image(img_path, use_container_width=True)
                    st.caption(f"{os.path.basename(img_path)} (Sim: {score:.2f})")
                else:
                    st.error(f"Image not found: {img_path}")

# --- Helper function patch ---
# 因为直接 import model_loader 可能会有相对路径问题，我们在 app.py 开头处理了 sys.path
# 但为了简单起见，我们给 Processor 加一个临时方法或者直接调用
def get_text_embedding_safe(text):
    from src.core.model_loader import get_text_embedding
    return get_text_embedding(text)

Processor.get_text_embedding_safe = staticmethod(get_text_embedding_safe)
