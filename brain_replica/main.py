from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import json
import uvicorn
import os
import requests
from typing import List
from pathlib import Path

try:
    from huggingface_hub import InferenceClient
except ImportError:  # optional — only needed for /api/chat
    InferenceClient = None

app = FastAPI(title="3D Brain Visualization", description="Interactive 3D brain model with pathway animations")

# Get the directory where this file is located
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"

# Add CORS middleware for web development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (HTML, GLTF, etc.)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Brain region mappings (from your mesh_download.py)
region_to_node = {
    "Cerebellum": 5,
    "Left Occipital Lobe": 23,
    "Right Occipital Lobe": 22,
    "Left Temporal Lobe": 4,
    "Right Temporal Lobe": 13,
    "Left Parietal Lobe": 28,
    "Right Parietal Lobe": 25,
    "Left Frontal Lobe": 14,
    "Right Frontal Lobe": 9,
    "Brain Stem": 18,
    "Pituitary Gland": 24,
}

# Pathway definitions
pathways = {
    "listening": [
        "Brain Stem",
        "Left Temporal Lobe", 
        "Right Temporal Lobe",
        "Left Parietal Lobe",
        "Right Parietal Lobe"
    ],
    "speaking": [
        "Left Frontal Lobe",
        "Right Frontal Lobe", 
        "Left Temporal Lobe",
        "Right Temporal Lobe",
        "Brain Stem"
    ],
    "thinking": [
        "Left Frontal Lobe",
        "Right Frontal Lobe",
        "Left Parietal Lobe", 
        "Right Parietal Lobe",
        "Left Temporal Lobe", 
        "Right Temporal Lobe",
    ]
}

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # Remove dead connections
                self.active_connections.remove(connection)

client = (
    InferenceClient(model="HuggingFaceTB/SmolLM3-3B", token=os.environ.get("HF_TOKEN"))
    if InferenceClient is not None
    else None
)

import re
def call_hf_model(prompt, max_tokens=1024):
    """Call Hugging Face Inference API for chat completions using chat.completions.create logic and remove <think> sections."""
    if client is None:
        return "Chat is unavailable (install huggingface_hub to enable the assistant)."
    try:
        system_message = {
            "role": "system",
            "content": "You are a neuroscience explainer bot. For any explanations only use these brain parts: Cerebellum, Left Occipital Lobe, Right Occipital Lobe, Left Temporal Lobe, Right Temporal Lobe, Left Parietal Lobe, Right Parietal Lobe, Left Frontal Lobe, Right Frontal Lobe, Brain Stem, Pituitary Gland. Please be as succinct as possible."
        }
        user_message = {
            "role": "user",
            "content": prompt
        }
        chat_response = client.chat.completions.create(
            messages=[system_message, user_message],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        # Extract the assistant's reply
        text = ""
        if hasattr(chat_response, "choices") and chat_response.choices:
            text = chat_response.choices[0].message.content
        # Remove <think>...</think> sections
        if text:
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
            text = text.strip()
        # Fallback if empty
        if not text:
            text = "Sorry, I couldn't get a response from the AI model."
        return text
    except Exception as e:
        print("Hugging Face API call failed:", e)
        return "Sorry, I couldn't get a response from the AI model."

manager = ConnectionManager()

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return FileResponse(str(STATIC_DIR / "index.html"))

@app.get("/bci", response_class=HTMLResponse)
async def bci_page():
    """Closed-loop motor-imagery BCI demo (connects to decode_live.py :8765)."""
    # Read the HTML file and inject decoder URL if provided
    decoder_url = os.environ.get("DECODER_URL", "")
    html_content = (STATIC_DIR / "bci.html").read_text()
    if decoder_url:
        html_content = html_content.replace(
            "const WS_HOST = typeof DECODER_URL !== 'undefined' ? DECODER_URL : `${location.hostname || 'localhost'}:8765`;",
            f"const WS_HOST = '{decoder_url}';"
        )
    return Response(content=html_content, media_type="text/html")

@app.get("/api/regions")
async def get_regions():
    """Get available brain regions"""
    return {"regions": list(region_to_node.keys())}

@app.get("/api/pathways")
async def get_pathways():
    """Get available pathways"""
    return {"pathways": list(pathways.keys())}

@app.post("/api/highlight/{region_name}")
async def highlight_region(region_name: str):
    """Highlight a specific brain region"""
    if region_name not in region_to_node:
        return {"error": f"Region '{region_name}' not found"}
    
    message = {
        "type": "highlight_region",
        "region": region_name,
        "node_id": region_to_node[region_name]
    }
    await manager.broadcast(json.dumps(message))
    return {"success": True, "region": region_name}

@app.post("/api/animate/{pathway_name}")
async def animate_pathway(pathway_name: str):
    """Animate a specific pathway"""
    if pathway_name not in pathways:
        return {"error": f"Pathway '{pathway_name}' not found"}
    
    message = {
        "type": "animate_pathway", 
        "pathway": pathway_name,
        "regions": pathways[pathway_name]
    }
    await manager.broadcast(json.dumps(message))
    return {"success": True, "pathway": pathway_name}

@app.post("/api/stop-animation")
async def stop_animation():
    """Stop current animation"""
    message = {"type": "stop_animation"}
    await manager.broadcast(json.dumps(message))
    return {"success": True}

@app.post("/api/chat")
async def chat_with_ai(request: Request):
    """Chat with the AI using Hugging Face model"""
    try:
        body = await request.json()
        user_message = body.get("message", "")
        if not user_message:
            return {"error": "No message provided"}

        # Call the Hugging Face API
        response = call_hf_model(user_message)

        if response:
            return {"response": response}
        else:
            return {"response": "Sorry, I couldn't get a response from the AI model."}

    except Exception as e:
        print(f"Chat API error: {e}")
        return {"error": f"Chat failed: {str(e)}"}

@app.get("/api/bci/prediction")
async def get_bci_prediction():
    """Get mock BCI prediction for Vercel polling (serverless-compatible)"""
    import time
    import random

    # Cycle through predictions like the original mock mode
    predictions = ['left', 'right', 'rest']
    cycle_length = 10  # Hold each prediction for 10 requests (2.5s at 250ms polling)
    current_time = int(time.time() * 1000)  # Current time in ms
    cycle_index = (current_time // 2500) % len(predictions)  # Change every 2.5s
    pred = predictions[cycle_index]

    conf = 0.7 + random.random() * 0.25

    # More realistic values
    current_seconds = int(time.time())  # Use seconds instead of ms
    trial_number = 50 + (current_seconds // 30)  # Increment trial every 30 seconds
    rolling_acc = 0.70 + random.random() * 0.06  # 70-76%
    c3_voltage = 90 + random.random() * 360  # 90-450 mV
    c4_voltage = 90 + random.random() * 360  # 90-450 mV

    return {
        "type": "prediction",
        "prediction": pred,
        "confidence": conf,
        "probs": {
            "left": conf if pred == "left" else random.random() * 0.2,
            "right": conf if pred == "right" else random.random() * 0.2,
            "rest": conf if pred == "rest" else random.random() * 0.2
        },
        "rolling_accuracy": rolling_acc,
        "n_scored": trial_number,
        "marker": "T1" if pred == "left" else "T2" if pred == "right" else "T0",
        "band_power": {
            "C3": c3_voltage,
            "C4": c4_voltage
        },
        "timestamp": time.time()
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle incoming messages if needed
            await websocket.send_text(f"Message received: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)