import gradio as gr

def build_ui(pipeline_fn):
    def answer_fn(question):
        return pipeline_fn(question)

    return gr.Interface(fn=answer_fn, inputs="text", outputs="text", title="Simple RAG Demo")
