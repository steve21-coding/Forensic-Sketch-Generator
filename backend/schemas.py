from pydantic import BaseModel
from typing import Optional, List

class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    steps: Optional[int] = 35
    guidance: Optional[float] = 5.5

class BuildIndexRequest(BaseModel):
    folder_path: str

class SearchRequest(BaseModel):
    image_b64: str  # base64-encoded PNG/JPG
    k: Optional[int] = 5

class MatchResult(BaseModel):
    rank: int
    suspect_id: str
    final_score: float
    bio_dist: float
    skin_match: float
    hair_match: float
    gender: str
    local_path: str
    mugshot_b64: Optional[str] = None

class GenerateAndSearchRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    steps: Optional[int] = 35
    guidance: Optional[float] = 5.5
    k: Optional[int] = 5