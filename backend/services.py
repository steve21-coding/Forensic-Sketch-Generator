import os, sys, uuid, base64
import cv2, numpy as np, faiss, torch
from torchvision.transforms.functional import to_tensor
from backend.errors import FaceNotFoundError, IndexNotBuiltError
from backend.config import models_singleton as models

# Vector database global tracking variables
index = None
db_metadata = []

# ─── EXTRACTION LOGIC ────────────────────────────────────────────────────────
def get_mask_color(img_rgb, mask, label_id):
    pixels = img_rgb[mask == label_id]
    if len(pixels) < 10: return None
    return np.mean(pixels, axis=0).astype(int).tolist()

def get_mask_color_hsv(img_bgr, mask, label_id):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    pixels = hsv[mask == label_id]
    if len(pixels) < 10: return None
    return np.mean(pixels[:, :2], axis=0).astype(int).tolist()

def get_hsv_similarity(hsv1, hsv2):
    if hsv1 is None or hsv2 is None: return 0.5
    dst = np.linalg.norm(np.array(hsv1) - np.array(hsv2))
    return max(0.0, 1 - (dst / 312.0))

def parse_face(img_bgr):
    if models.is_mocked:
        return None, [20, 150], [40, 30, 20], False
        
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (512, 512))
    inp = to_tensor(img_resized).unsqueeze(0).cuda()
    with torch.no_grad():
        out = models.net(inp)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
    mask = cv2.resize(parsing.astype(np.uint8), (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    skin_hsv  = get_mask_color_hsv(img_bgr, mask, 1)
    hair_rgb  = get_mask_color(img_rgb, mask, 17)
    has_beard = bool(np.sum(mask == 11) > 500)
    return mask, skin_hsv, hair_rgb, has_beard

def img_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def generate_composite_image(prompt: str, steps: int, guidance: float, output_dir: str) -> str:
    fname = f"suspect_{uuid.uuid4().hex[:8]}.png"
    fpath = os.path.join(output_dir, fname)
    
    if models.is_mocked:
        img = Image.new('RGB', (832, 1216), color=(128, 128, 128))
        from PIL import ImageDraw
        d = ImageDraw.Draw(img)
        d.text((100,100), "MOCK SKETCH GENERATION (CPU)", fill=(255,255,255))
        img.save(fpath)
        return fname

    forensic_prompt = (
        f"Mugshot portrait of a suspect: {prompt}. Front-facing view, neutral expression, looking at camera. "
        "Plain grey background, harsh flat lighting, raw photography, hyper-realistic, police photography style."
    )
    neg = "cgi, 3d, render, cartoon, anime, illustration, painting, blurry, smile, hat, sunglasses"
    
    image = models.pipe(prompt=forensic_prompt, negative_prompt=neg, num_inference_steps=steps, guidance_scale=guidance, width=832, height=1216).images[0]
    image.save(fpath)
    return fname

def execute_vector_search(image_b64: str, k: int) -> dict:
    global index, db_metadata
    if index is None or len(db_metadata) == 0:
        raise IndexNotBuiltError()

    img_bytes = base64.b64decode(image_b64)
    img_np = np.frombuffer(img_bytes, np.uint8)
    img_bgr = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not decode image.")

    if models.is_mocked:
        return "Male", 35, [{
            "rank": i+1, "suspect_id": m["suspect_id"], "final_score": 0.15, "bio_dist": 0.05,
            "skin_match": 95.0, "hair_match": 92.0, "gender": m["gender"], "local_path": m["local_path"]
        } for i, m in enumerate(db_metadata[:k])]

    faces = models.fa_app.get(img_bgr)
    if not faces:
        raise FaceNotFoundError()
        
    face_vec = faces[0].normed_embedding
    gender = "Male" if faces[0].gender == 1 else "Female"
    age = int(faces[0].age)

    _, skin_hsv, hair_rgb, has_beard = parse_face(img_bgr)
    qvec = face_vec.reshape(1, -1).astype('float32')
    distances, indices = index.search(qvec, min(300, index.ntotal))

    results = []
    for i, idx in enumerate(indices[0]):
        m = db_metadata[idx]
        skin_sim  = get_hsv_similarity(skin_hsv,  m.get('skin_hsv'))
        hair_sim  = get_hsv_similarity(hair_rgb,  m.get('hair_rgb'))
        bio_dist  = float(distances[0][i])

        skin_penalty  = (1 - skin_sim) * 3.0
        beard_penalty = 0.6 if (has_beard and not m.get('has_beard')) else 0.0
        hair_penalty  = (1 - hair_sim)  * 2.0
        gender_penalty = 0.5 if (gender != m.get('gender', gender)) else 0.0
        final_score   = bio_dist + skin_penalty + beard_penalty + hair_penalty + gender_penalty

        results.append({
            "rank": i + 1, "suspect_id": m.get('suspect_id', f'UNK-{idx:05d}'), "final_score": round(final_score, 3),
            "bio_dist": round(bio_dist, 3), "skin_match": round(skin_sim * 100, 1), "hair_match": round(hair_sim * 100, 1),
            "gender": m.get('gender', '—'), "local_path": m.get('local_path', ''),
        })

    results = sorted(results, key=lambda x: x['final_score'])[:k]
    for rank, r in enumerate(results, 1):
        r['rank'] = rank
    return gender, age, results