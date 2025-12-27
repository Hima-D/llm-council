import gradio as gr
from council import AdvancedLLMCouncil  # Adjust import as needed
import os
import json

def deliberate_query(question):
    api_key = os.getenv("OPENAI_API_KEY")  # User sets in Space secrets
    if not api_key:
        return {"error": "Set OPENAI_API_KEY in Space secrets."}
    
    council = AdvancedLLMCouncil(api_key)
    decision = council.deliberate(question, max_reflection_rounds=1)
    return json.dumps(decision.model_dump(), indent=2)

iface = gr.Interface(
    fn=deliberate_query, 
    inputs=gr.Textbox(label="Query", placeholder="e.g., What is ML?"),
    outputs=gr.JSON(label="Decision Object"),
    title="LLM Council Demo",
    description="3 agents + 2 judges deliberate your query."
)

if __name__ == "__main__":
    iface.launch()
