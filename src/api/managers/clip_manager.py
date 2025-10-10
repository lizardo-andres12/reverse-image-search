import torch
from config import CLIPConfig
from transformers import CLIPModel, CLIPProcessor


class CLIPManager:
    """Manages lifecycle of CLIP embedding model"""

    def __init__(self, config: CLIPConfig):
        self.model_name = config.MODEL_NAME
        self.supported_formats = config.SUPPORTED_FORMATS
        self.cache_dir = config.CACHE_DIR
        
        device = config.DEVICE
        if device == "":
            if torch.cuda.is_available():
                device = "cuda"
            device = "cpu"
        self.device = device

    def initialize(self):
        """Loads the CLIP model into memory"""

        self.model = CLIPModel.from_pretrained(
            self.model_name, cache_dir=self.cache_dir
        )
        self.processor = CLIPProcessor.from_pretrained(
            self.model_name, cache_dir=self.cache_dir
        )

        self.model.to(self.device)  # type: ignore[arg-type]
        self.model.eval()

        if self.device == "cuda":
            self.model = torch.compile(self.model)

    def teardown(self):
        """Unload model and clear caches"""

        if self.device == "cuda":
            torch.cuda.empty_cache()
