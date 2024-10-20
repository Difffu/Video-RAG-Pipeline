# Import necessary libraries
from transformers import LlavaProcessor, LlavaForConditionalGeneration, LlamaForConditionalGeneration
import torch
from langchain.document_loaders import VideoLoader
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import cv2
import requests
from PIL import Image
import uuid

# Video Data Ingestion - Sample Frames
def sample_frames(url, num_frames=6):
    """
    Sample a given number of frames from the video at the provided URL.
    """
    response = requests.get(url)
    path_id = str(uuid.uuid4())
    path = f"./{path_id}.mp4"

    with open(path, "wb") as f:
        f.write(response.content)

    video = cv2.VideoCapture(path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = total_frames // num_frames
    frames = []

    for i in range(total_frames):
        ret, frame = video.read()
        if not ret:
            continue
        if i % interval == 0:
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(pil_img)
    video.release()
    return frames

# Initialize the models for video-text summarization
model_id = "llava-hf/llava-interleave-qwen-0.5b-hf"

processor = LlavaProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16)
model.to("cuda")

# Create a utility to replace video tokens with sampled image frames
def replace_video_with_images(text, frames):
    return text.replace("<video>", "<image>" * frames)

# Define the LangChain RAG pipeline (VideoLoader, Retriever, and Llama 3.2 as generator)
def get_rag_pipeline(video_url):
    # 1. Load and process video
    video_frames = sample_frames(video_url)

    # 2. Set up Retrieval using LangChain
    # Load the video data into LangChain's video document loader
    video_loader = VideoLoader(video_url)
    docs = video_loader.load()

    # Use FAISS for vector similarity search
    embedding = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(docs, embedding)

    # 3. Set up Retrieval QA pipeline
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # Llama 3.2 model for answer generation
    llm = LlamaForConditionalGeneration.from_pretrained("llama-3.2", torch_dtype=torch.float16)
    
    # Build the retrieval-augmented pipeline
    qa_chain = RetrievalQA(llm=llm, retriever=retriever)

    return qa_chain

# Run inference using the RAG pipeline
video_url = "https://huggingface.co/spaces/merve/llava-interleave/resolve/main/cats_1.mp4"
rag_pipeline = get_rag_pipeline(video_url)

# Ask a question about the video and generate a summary
query = "What are the cats doing in the video?"
result = rag_pipeline.run(query)

print(result)

num_f = 5
video_frames = sample_frames(video_url, num_f)

# Summarize the video using Llava interleave model
user_prompt = "Summarize the content of the video."
toks = "<image>" * len(video_frames)
prompt = f"<|im_start|>user{toks}\n{user_prompt}<|im_end|><|im_start|>assistant"

inputs = processor(prompt, images=video_frames).to(model.device, model.dtype)
output = model.generate(**inputs, max_new_tokens=100, do_sample=False)

# Print the generated summary
summary = processor.decode(output[0], skip_special_tokens=True)
print("Summary:", summary)
