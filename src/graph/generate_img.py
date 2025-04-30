# 生成工作流图
from IPython.display import display
from PIL import Image as PILImage
import io
from langgraph.graph import StateGraph

def generate_img(parallelWorkflow: StateGraph, output_path: str = "output.png", max_retries: int = 5, retry_delay: float = 2.0) -> None:
    """生成工作流图
    
    Args:
        parallelWorkflow: StateGraph 工作流图对象
        output_path: 输出图片路径
        max_retries: 重试次数
        retry_delay: 重试延迟时间(秒)
    """
    try:
        # 尝试使用在线API生成
        image_data = parallelWorkflow.get_graph().draw_mermaid_png(
            max_retries=max_retries,
            retry_delay=retry_delay
        )
    except Exception:
        # 在线API失败时使用本地浏览器渲染
        image_data = parallelWorkflow.get_graph().draw_mermaid_png()
        
    image = PILImage.open(io.BytesIO(image_data))
    image.save(output_path)
