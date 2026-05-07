#!/usr/bin/env python3
"""
Gradio Web Demo for TinyMultimodal — image captioning, VQA, and retrieval.
Usage:
  python demo/web_app.py --checkpoint ../checkpoints_phase6_vqa/best.pt
"""

import os, sys, argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.model import TinyMultimodal
from core.tokenizer import SimpleTokenizer
from core.config import resolve_config
from utils import load_checkpoint_adaptive


def load_model(checkpoint_path, device):
    tok = SimpleTokenizer(max_vocab=10000)
    cfg = resolve_config(checkpoint_path, tok,
        defaults={'img_generation': True, 'use_audio': True, 'use_video': True,
                  'use_memory_bank': True, 'n_mem_tokens': 16})
    model = TinyMultimodal(cfg).to(device)
    if os.path.exists(checkpoint_path):
        load_checkpoint_adaptive(model, checkpoint_path, device)
    model.eval()
    return model, tok, cfg


def preprocess_image(image, size=224):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.convert('RGB').resize((size, size), Image.LANCZOS)
    tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 127.5 - 1.0
    return tensor


class ModelSession:
    def __init__(self, checkpoint, device):
        self.model, self.tok, self.cfg = load_model(checkpoint, device)
        self.device = device
        self.ctx = None  # multi-turn context

    def caption(self, image):
        img_tensor = preprocess_image(image).to(self.device)
        gen = self.model.generate_text(img_tensor, self.tok, max_len=48,
                                       temperature=0.7, top_k=30)
        return gen.strip()

    def start_chat(self, image):
        img_tensor = preprocess_image(image).to(self.device)
        self.ctx = self.model.start_conversation(img_tensor)
        return "Chat started! Ask a question about the image."

    def chat(self, question):
        if self.ctx is None:
            return "Please upload an image first and click 'Start Chat'."
        answer, self.ctx = self.model.chat(self.ctx, question, self.tok,
                                           max_len=48, temperature=0.7, top_k=30)
        return answer.strip()

    def ask(self, image, question):
        img_tensor = preprocess_image(image).to(self.device)
        prompt = f"问：{question}\n答："
        text_ids = torch.tensor([[2] + self.tok.encode(prompt)], dtype=torch.long,
                                device=self.device)[:, :64]
        gen = self.model.generate_text(img_tensor, self.tok, max_len=48,
                                       temperature=0.7, top_k=30)
        answer = gen.split('答：', 1)[1].strip() if '答：' in gen else gen.strip()
        return answer

    def model_info(self):
        params = sum(p.numel() for p in self.model.parameters()) / 1e6
        return f"TinyMultimodal {self.cfg.describe()} | {params:.1f}M params"


def build_demo(checkpoint, share=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    session = ModelSession(checkpoint, device)
    print(f"Model loaded: {session.model_info()}")

    try:
        import gradio as gr

        with gr.Blocks(title="TinyMultimodal Demo") as demo:
            gr.Markdown(f"# TinyMultimodal — Web Demo\n{session.model_info()}")

            with gr.Tabs():
                # Tab 1: Image Captioning
                with gr.TabItem("Image Captioning"):
                    with gr.Row():
                        img_input = gr.Image(type="pil", label="Upload Image")
                        with gr.Column():
                            caption_output = gr.Textbox(label="Generated Caption", lines=3)
                            caption_btn = gr.Button("Generate Caption")
                    caption_btn.click(session.caption, inputs=img_input, outputs=caption_output)

                # Tab 2: VQA
                with gr.TabItem("Visual Q&A"):
                    with gr.Row():
                        vqa_img = gr.Image(type="pil", label="Upload Image")
                        with gr.Column():
                            vqa_question = gr.Textbox(label="Question (Chinese OK)")
                            vqa_answer = gr.Textbox(label="Answer", lines=3)
                            vqa_btn = gr.Button("Ask")
                    vqa_btn.click(session.ask, inputs=[vqa_img, vqa_question], outputs=vqa_answer)

                # Tab 3: Multi-turn Chat
                with gr.TabItem("Multi-turn Chat"):
                    with gr.Row():
                        chat_img = gr.Image(type="pil", label="Upload Image")
                        with gr.Column():
                            chat_status = gr.Textbox(label="Status")
                            start_btn = gr.Button("Start Chat")
                            chat_question = gr.Textbox(label="Your Question")
                            chat_answer = gr.Textbox(label="Response", lines=3)
                            chat_btn = gr.Button("Send")
                    start_btn.click(session.start_chat, inputs=chat_img, outputs=chat_status)
                    chat_btn.click(session.chat, inputs=chat_question, outputs=chat_answer)

                # Tab 4: Model Info
                with gr.TabItem("About"):
                    gr.Markdown(f"""
                    ## TinyMultimodal
                    - Architecture: {session.cfg.describe()}
                    - Parameters: {sum(p.numel() for p in session.model.parameters())/1e6:.1f}M
                    - Memory Bank: {session.cfg.use_memory_bank} ({session.cfg.n_mem_tokens} tokens)
                    - KV Cache: enabled
                    - Modalities: text + image + audio + video

                    Built with PyTorch. 100% offline, no API calls.
                    """)

        demo.launch(share=share, server_name="0.0.0.0")

    except ImportError:
        print("Gradio not installed. Run: pip install gradio")
        print("\nFalling back to CLI mode...")
        _cli_mode(session)


def _cli_mode(session):
    print(f"\n{session.model_info()}")
    print("CLI mode: type 'q' to quit\n")
    while True:
        path = input("Image path: ").strip()
        if path.lower() == 'q':
            break
        if not os.path.exists(path):
            print(f"  File not found: {path}")
            continue

        img = Image.open(path)
        prompt = input("  Question (empty for auto-caption): ").strip()
        if prompt:
            answer = session.ask(img, prompt)
        else:
            answer = session.caption(img)
        print(f"  → {answer}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="../checkpoints_phase6_vqa/best.pt")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    args = parser.parse_args()
    build_demo(args.checkpoint, args.share)


if __name__ == '__main__':
    main()
