from fastapi import Request
from fastapi.responses import JSONResponse

class ForensicException(Exception):
    """Base exception for all Forensic Sketch Generator system errors"""
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

class FaceNotFoundError(ForensicException):
    def __init__(self, message: str = "No face detected in the query image."):
        super().__init__(message, status_code=400)

class IndexNotBuiltError(ForensicException):
    def __init__(self, message: str = "Database index not built. Call /api/build_index first."):
        super().__init__(message, status_code=503)

async def forensic_exception_handler(request: Request, exc: ForensicException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "error": exc.message}
    )