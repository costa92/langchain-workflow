 # 生成工作流图
from IPython.display import display
from PIL import Image as PILImage
import io
from langgraph.graph import StateGraph

def generate_img(parallelWorkflow: StateGraph, output_path: str = "output.png") -> None:
  """生成工作流图"""  
  image_data = parallelWorkflow.get_graph().draw_mermaid_png()
  image = PILImage.open(io.BytesIO(image_data))
  image.save(output_path)




