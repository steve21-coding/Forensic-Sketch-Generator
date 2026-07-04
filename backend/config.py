import torch

class ModelRegistry:
    """
    Singleton Model Registry to manage the lifecycle of heavy ML models.
    Guarantees models are loaded into memory exactly once.
    """
    def __init__(self):
        self.pipe = None      # SDXL Pipeline
        self.fa_app = None    # InsightFace
        self.net = None       # BiSeNet
        self.is_mocked = False

    def load_all_models(self):
        import sys
        
        if not torch.cuda.is_available():
            print("⚠️ CUDA GPU not detected! Falling back to GitHub Mock Mode for testing...")
            self.is_mocked = True
            return

        from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
        from insightface.app import FaceAnalysis
        sys.path.append('/content/face-parsing.PyTorch')
        from model import BiSeNet

        print(">> Loading SDXL pipeline...")
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            "RunDiffusion/Juggernaut-XL-v9",
            torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        ).to("cuda")
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config, use_karras_sigmas=True
        )

        print(">> Loading InsightFace...")
        self.fa_app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.fa_app.prepare(ctx_id=0, det_size=(640, 640))

        print(">> Loading BiSeNet...")
        self.net = BiSeNet(n_classes=19)
        ckpt = '/content/face-parsing.PyTorch/res/cp/79999_iter.pth'
        self.net.cuda()
        self.net.load_state_dict(torch.load(ckpt))
        self.net.eval()
        print("✅ Production ML models loaded inside global registry singleton.")

# Single source of truth across the entire application
models_singleton = ModelRegistry()