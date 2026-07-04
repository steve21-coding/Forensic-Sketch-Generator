import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager

from backend.schemas import GenerateRequest, SearchRequest, BuildIndexRequest, GenerateAndSearchRequest
from backend.errors import ForensicException, forensic_exception_handler
from backend.config import models_singleton
import backend.services as svc

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Safe boot loading exactly once
    models_singleton.load_all_models()
    yield

app = FastAPI(title="Forensic Sketch Generator API", version="1.2.0", lifespan=lifespan)
app.add_exception_handler(ForensicException, forensic_exception_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = "/content/forensic_sketch_generator_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": models_singleton.pipe is not None or models_singleton.is_mocked,
        "mock_mode": models_singleton.is_mocked,
        "index_size": svc.index.ntotal if svc.index is not None else 0,
    }

@app.post("/api/generate")
def generate(req: GenerateRequest):
    try:
        fname = svc.generate_composite_image(req.prompt, req.steps, req.guidance, OUTPUT_DIR)
        fpath = os.path.join(OUTPUT_DIR, fname)
        return {
            "success": True,
            "filename": fname,
            "image_b64": svc.img_to_b64(fpath),
        }
    except Exception as e:
        raise HTTPException(500, f"Generation failure: {str(e)}")

@app.post("/api/search")
def search(req: SearchRequest):
    gender, age, matches = svc.execute_vector_search(req.image_b64, req.k)
    for m in matches:
        path = m.get('local_path', '')
        m['mugshot_b64'] = svc.img_to_b64(path) if path and os.path.exists(path) else None
        
    return {
        "success": True,
        "query_gender": gender,
        "query_age": age,
        "matches": matches,
    }

@app.post("/api/generate_and_search")
def generate_and_search(req: GenerateAndSearchRequest):
    fname = svc.generate_composite_image(req.prompt, req.steps, req.guidance, OUTPUT_DIR)
    fpath = os.path.join(OUTPUT_DIR, fname)
    b64 = svc.img_to_b64(fpath)

    gender, age, matches = svc.execute_vector_search(b64, req.k)
    for m in matches:
        path = m.get('local_path', '')
        m['mugshot_b64'] = svc.img_to_b64(path) if path and os.path.exists(path) else None

    return {
        "success": True,
        "generated_image_b64": b64,
        "generated_filename": fname,
        "query_gender": gender,
        "query_age": age,
        "matches": matches,
    }

@app.post("/api/build_index")
def build_index(req: BuildIndexRequest):
    import cv2, numpy as np, faiss
    if not os.path.isdir(req.folder_path):
        raise HTTPException(400, f"Folder path not found: {req.folder_path}")

    all_files = [os.path.join(root, f) for root, _, files in os.walk(req.folder_path) for f in files if not f.startswith('.')]
    new_embeddings, new_metadata = [], []
    failed = 0

    for path in all_files:
        img = cv2.imread(path)
        if img is None: failed += 1; continue
        
        if models_singleton.is_mocked:
            new_embeddings.append(np.random.rand(512).astype('float32'))
            new_metadata.append({"suspect_id": os.path.splitext(os.path.basename(path))[0], "local_path": path, "gender": "Male"})
            continue

        faces = models_singleton.fa_app.get(img)
        if not faces: failed += 1; continue
        try:
            _, skin_hsv, hair_rgb, has_beard = svc.parse_face(img)
        except Exception:
            failed += 1; continue

        new_embeddings.append(faces[0].normed_embedding)
        new_metadata.append({
            "suspect_id": os.path.splitext(os.path.basename(path))[0], "local_path": path,
            "skin_hsv": skin_hsv, "hair_rgb": hair_rgb, "has_beard": has_beard,
            "gender": "Male" if faces[0].gender == 1 else "Female",
        })

    if not new_embeddings:
        raise HTTPException(400, "No valid faces discovered in database pathway.")

    mat = np.array(new_embeddings).astype('float32')
    new_index = faiss.IndexFlatL2(mat.shape[1])
    new_index.add(mat)

    svc.index = new_index
    svc.db_metadata = new_metadata

    return {"success": True, "indexed": len(new_embeddings), "failed": failed, "total": len(all_files)}

@app.get("/api/image/{filename}")
def serve_image(filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(404, "Target image resource not found.")
    return FileResponse(path, media_type="image/png")