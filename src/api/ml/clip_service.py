import logging
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

logger = logging.getLogger(__name__)


@dataclass
class CLIPConfig:
    """CLIP model config"""

    model_name: str = "openai/clip-vit-base-patch32"
    device: str | None = None
    cache_dir: str | None = None
    supported_formats: tuple = "RGB", "RGBA", "L"


class CLIPModelService:
    """Class encapsulating functionality for interacting with CLIP model"""

    def __init__(self, config: CLIPConfig | None = None):
        self.config = config if config else CLIPConfig()
        self.device = self._get_device()
        self.model: CLIPModel | None = None
        self.processor: CLIPProcessor | None = None
        self._is_loaded = False

    def extract_image_features(self, image: Image.Image) -> np.ndarray | None:
        """
        Extract features from image using clip as either 512 or 768 dimension vector.

        Args:
            image (Image.Image): The PIL Image object.

        Returns:
            np.ndarray | None: Normalized feature vector as numpy array or None if there is a failure.
        """

        if not self.is_ready():
            logger.error("Model not loaded. Call load_model() first.")
            return None

        try:
            if image.mode not in self.config.supported_formats:
                image = image.convert("RGB")

            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                normalized_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )

            return normalized_features.cpu().numpy().flatten()
        except Exception as e:
            logger.error(f"Failed to extract image features: {e}")
            return None

    def extract_text_features(self, text: str) -> np.ndarray | None:
        """
        Extract features from text using CLIP.

        Args:
            text (str): Text string to encode.
        Returns:
            np.ndarray | None: Normalized text feature vector as numpy array or None if there is a failure.
        """

        if not self.is_ready():
            logger.error("Model not loaded. Call load_model() first.")
            return None

        try:
            inputs = self.processor(text=[text], return_tensors="pt").to(self.device)
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                normalized_features = text_features / text_features.norm(
                    dim=-1, keepdim=True
                )

            return normalized_features.cpu().numpy().flatten()
        except Exception as e:
            logger.error(f"Failed to extract text features: {e}")
            return None

    def extract_batch_image_features(
        self, images: list[Image.Image], batch_size: int = 32
    ) -> list[np.ndarray | None] | None:
        """
        Extract features from multiple images in batches for efficiency.

        Args:
            images (list[Image.Image]): The list of PIL image objects to batch process.
            batch_size (int): The number of images to process in one batch
        Returns:
            list[np.ndarray | None] | None: List of feature vectors (or None for failed extractions)
            in the same input order or None if the model was not loaded at invocation.
        """

        if not self.is_ready():
            logger.error("Model not loaded. Call load_model() first.")
            return None

        total_batches = (len(images) + batch_size - 1) // batch_size
        results: list[np.ndarray | None] = []

        for i in range(0, len(images), batch_size):
            batch_num = (i // batch_size) + 1

            batch = images[i : i + batch_size]
            batch_results = [None] * len(batch)

            try:
                processed_images = []
                processed_idxs = []
                for idx, image in enumerate(batch):
                    try:
                        if image.mode not in self.config.supported_formats:
                            image = image.convert("RGB")
                        processed_images.append(image)
                        processed_idxs.append(idx)
                    except Exception as e:
                        logger.error(f"Failed to preprocess image in batch: {e}")

                if processed_images:
                    inputs = self.processor(
                        images=processed_images, return_tensors="pt"
                    ).to(self.device)
                    with torch.no_grad():
                        image_features = self.model.get_image_features(**inputs)
                        normalized_features = image_features / image_features.norm(
                            dim=-1, keepdim=True
                        )

                    for j, orig_idx in enumerate(processed_idxs):
                        batch_results[orig_idx] = normalized_features[j].cpu().numpy()

            except Exception as e:
                logger.error(
                    f"Batch {batch_num}/{total_batches} failed to preprocess: {e}"
                )

            results.extend(batch_results)
            logger.info(f"Batch {batch_num}/{total_batches} processed successfully")

        return results

    def load_model(self) -> bool:
        """
        Loads the CLIP model into memory.

        Returns:
            bool: True if model loads successfully, False otherwise
        """

        try:
            self.model = CLIPModel.from_pretrained(
                self.config.model_name, cache_dir=self.config.cache_dir
            )
            self.processor = CLIPProcessor.from_pretrained(
                self.config.model_name, cache_dir=self.config.cache_dir
            )

            self.model.to(self.device)  # type: ignore[arg-type]
            self.model.eval()
            if self.device == "cuda":
                self.model = torch.compile(self.model)

            self._is_loaded = True
            return True

        except Exception as e:
            self._is_loaded = False
            return False

    def is_ready(self) -> bool:
        """Convenience method to check if model is ready for use."""
        return self._is_loaded and self.model is not None and self.processor is not None

    def unload_model(self):
        """Unload model and clear caches."""
        if self.model:
            del self.model
            self.model = None

        if self.processor:
            del self.processor
            self.processor = None

        self._is_loaded = False

        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_model_info(self) -> dict:
        """Get information about model"""

        return {
            "model_name": self.config.model_name,
            "device": self.device,
            "is_loaded": self._is_loaded,
            "feature_dimensions": 512 if "base" in self.config.model_name else 768,
        }

    def _get_device(self) -> str:
        if self.config.device:
            return self.config.device

        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
