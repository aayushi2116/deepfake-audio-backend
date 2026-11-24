from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import json
import asyncio

from signaling import signaling_manager
from audio_processing import audio_processor
from deepfakeaudio_detection import resolve_checkpoint_path, get_model_threshold

import os

app = FastAPI(title="WebRTC Signaling & Audio Detection API")

# Configure CORS origins dynamically from environment variables
cors_env = os.getenv("CORS_ORIGINS", "").strip()
if cors_env:
    allowed_origins = [orig.strip() for orig in cors_env.split(",") if orig.strip()]
else:
    allowed_origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*"
    ]

frontend_url = os.getenv("FRONTEND_URL", "").strip()
if frontend_url and frontend_url not in allowed_origins:
    allowed_origins.append(frontend_url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "WebRTC & Audio Detection Server is running"}

@app.get("/health")
def health_check():
    return {"status": "ok", "active_rooms": len(signaling_manager.rooms)}

@app.get("/api/ice-servers")
def get_ice_servers():
    """
    Returns configured WebRTC ICE servers (STUN & TURN).
    Reads TURN credentials securely from backend environment variables.
    """
    ice_servers = [
        {"urls": "stun:stun.l.google.com:19302"},
        {"urls": "stun:stun1.l.google.com:19302"},
        {"urls": "stun:stun2.l.google.com:19302"}
    ]

    turn_urls = os.getenv("TURN_URLS", os.getenv("TURN_URL", "")).strip()
    turn_username = os.getenv("TURN_USERNAME", "").strip()
    turn_credential = os.getenv("TURN_CREDENTIAL", os.getenv("TURN_PASSWORD", "")).strip()

    if turn_urls:
        server_entry = {"urls": [u.strip() for u in turn_urls.split(",") if u.strip()]}
        if turn_username:
            server_entry["username"] = turn_username
        if turn_credential:
            server_entry["credential"] = turn_credential
        ice_servers.append(server_entry)

    return {"iceServers": ice_servers}

# ----------------- Signaling WebSocket -----------------
@app.websocket("/ws/signaling/{room_id}/{client_id}")
async def websocket_signaling(websocket: WebSocket, room_id: str, client_id: str):
    await signaling_manager.connect(room_id, client_id, websocket)
    try:
        while True:
            raw_data = await websocket.receive_text()
            data = json.loads(raw_data)
            
            target_id = data.get("target")
            if target_id:
                # Forward direct message (e.g. SDP offer/answer, ICE candidate)
                await signaling_manager.send_to_client(room_id, target_id, {
                    **data,
                    "sender": client_id
                })
            else:
                # Broadcast to others in the room
                await signaling_manager.broadcast(room_id, {
                    **data,
                    "sender": client_id
                }, exclude_client=client_id)
                
    except WebSocketDisconnect:
        signaling_manager.disconnect(room_id, client_id)
        await signaling_manager.broadcast(room_id, {
            "type": "user-left",
            "clientId": client_id
        })

# ----------------- Audio Detection WebSocket -----------------
ALLOWED_DETECTION_MODELS = {"webrtc", "mic", "original", "cond_4b"}

@app.websocket("/ws/detection/{client_id}")
async def websocket_detection(
    websocket: WebSocket,
    client_id: str,
    model: Optional[str] = Query("webrtc")
):
    # Extract and normalize requested model parameter
    raw_param = model or websocket.query_params.get("model", "webrtc") or "webrtc"
    requested_model = raw_param.strip().lower()

    # 1. Validate against explicit allowlist
    if requested_model not in ALLOWED_DETECTION_MODELS:
        await websocket.accept()
        err_msg = f"Invalid model '{requested_model}'. Allowed models: {sorted(list(ALLOWED_DETECTION_MODELS))}"
        await websocket.send_text(json.dumps({
            "status": f"Configuration Error: {err_msg}",
            "error": err_msg,
            "is_anomaly": False,
            "confidence": 0.0
        }))
        await websocket.close(code=4003, reason=err_msg[:120])
        return

    # 2. Check checkpoint existence; do not silently fall back if requested checkpoint is missing
    target_path, resolved_tag = resolve_checkpoint_path(requested_model)
    if not target_path.exists():
        await websocket.accept()
        err_msg = f"Model checkpoint for '{requested_model}' not found at '{target_path.name}'"
        await websocket.send_text(json.dumps({
            "status": f"Model Error: {err_msg}",
            "error": err_msg,
            "is_anomaly": False,
            "confidence": 0.0
        }))
        await websocket.close(code=4004, reason=err_msg[:120])
        return

    # 3. Accept connection and bind client session to requested model
    await websocket.accept()
    audio_processor.set_client_model(client_id, requested_model)

    try:
        while True:
            # Receive binary audio chunk or JSON message
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if "bytes" in message and message["bytes"]:
                audio_bytes = message["bytes"]
                try:
                    # Offload rolling buffer accumulation & model inference to worker thread
                    result = await asyncio.to_thread(
                        audio_processor.process_client_stream, client_id, audio_bytes, requested_model
                    )
                    await websocket.send_text(json.dumps(result))
                except ValueError as ve:
                    await websocket.send_text(json.dumps({
                        "status": f"Audio Error: {str(ve)}",
                        "error": str(ve),
                        "is_anomaly": False,
                        "confidence": 0.0
                    }))
                except Exception as ex:
                    await websocket.send_text(json.dumps({
                        "status": f"Inference Error: {str(ex)}",
                        "error": str(ex),
                        "is_anomaly": False,
                        "confidence": 0.0
                    }))
            elif "text" in message and message["text"]:
                await websocket.send_text(json.dumps({
                    "status": "ready",
                    "client_id": client_id,
                    "model": requested_model,
                    "threshold": get_model_threshold(resolved_tag)
                }))
    except WebSocketDisconnect:
        pass
    finally:
        audio_processor.clear_client_buffer(client_id)

# ----------------- Direct Audio File Upload Route -----------------
MAX_UPLOAD_SIZE = 25 * 1024 * 1024  # 25 MB

@app.post("/api/detect")
async def detect_audio(
    file: UploadFile = File(...),
    model: Optional[str] = Query(None, description="Optional model checkpoint alias ('webrtc', 'mic', 'original', 'retrained', or 'cond_4b')"),
    aggregation: Optional[str] = Query("mean", description="Aggregation method across windows ('mean' or 'median')"),
    threshold: Optional[float] = Query(None, description="Optional custom decision threshold override")
):
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(contents) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum upload size of {MAX_UPLOAD_SIZE // (1024 * 1024)} MB."
        )

    try:
        model_ver = model if isinstance(model, str) else None
        agg_ver = aggregation if isinstance(aggregation, str) and aggregation.lower() in ("mean", "median") else "mean"
        thresh_val = float(threshold) if isinstance(threshold, (int, float, str)) and str(threshold).strip() != "" else None
        result = await asyncio.to_thread(
            audio_processor.process_audio_file, contents, file.filename or "uploaded_file", model_ver, agg_ver, threshold=thresh_val
        )
        return {"filename": file.filename, "result": result}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
