from langchain_core.runnables.graph import MermaidDrawMethod
from langgraph.graph.state import CompiledGraph


def save_graph_as_png(app: CompiledGraph, output_file_path) -> None:
    # png_image = app.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.PYPPETEER, max_retries=3, retry_delay=1.0)
    # png_image = app.get_graph().draw_mermaid_png()
    # file_path = output_file_path if len(output_file_path) > 0 else "graph.png"
    # with open(file_path, "wb") as f:
    #     f.write(png_image)
    print(app.get_graph().draw_mermaid())
