import logging

import numpy as np
import torch
from managers import CLIPManager
from PIL import Image

logger = logging.getLogger(__name__)


class CLIPModelService:
    """Class encapsulating functionality for interacting with CLIP model"""

    def __init__(self, manager: CLIPManager):
        self.device = manager.device
        self.supported_formats = set(manager.supported_formats)
        self.model = manager.model
        self.processor = manager.processor

    def extract_image_features(self, image: Image.Image) -> list[float, ...] | None:
        """Extract features from image using clip as either 512 or 768 dimension vector.

        Args:
            image (Image.Image): The PIL Image object.
        Returns:
            np.ndarray | None: Normalized feature vector as numpy array or None if there is a failure.
        """
        if not image:
            return None

        if image.mode not in self.supported_formats:
            image = image.convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            normalized_features = image_features / image_features.norm(
                dim=-1, keepdim=True
            )

        return normalized_features.cpu().numpy().flatten().tolist()

    def extract_text_features(self, text: str) -> list[float, ...] | None:
        """Extract features from text using CLIP.

        Args:
            text (str): Text string to encode.
        Returns:
            np.ndarray | None: Normalized text feature vector as numpy array or None if there is a failure.
        """
        if text == "":
            return None

        inputs = self.processor(text=[text], return_tensors="pt").to(self.device)
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            normalized_features = text_features / text_features.norm(
                dim=-1, keepdim=True
            )

        return normalized_features.cpu().numpy().flatten().tolist()

    def extract_batch_image_features(
        self, images: list[Image.Image], batch_size: int = 32
    ) -> list[list[float, ...] | None] | None:
        """Extract features from multiple images in batches for efficiency.

        Args:
            images (list[Image.Image]): The list of PIL image objects to batch process.
            batch_size (int): The number of images to process in one batch
        Returns:
            list[np.ndarray | None] | None: List of feature vectors (or None for failed extractions)
            in the same input order or None if the model was not loaded at invocation.
        """
        total_batches = (len(images) + batch_size - 1) // batch_size
        results: list[np.ndarray | None] = []

        for i in range(0, len(images), batch_size):
            batch_num = (i // batch_size) + 1
            batch = images[i : i + batch_size]
            batch_results = [list()] * len(batch)

            try:
                processed_images = []
                processed_idxs = []
                for idx, image in enumerate(batch):
                    try:
                        if image.mode not in self.supported_formats:
                            image = image.convert("RGB")
                        processed_images.append(image)
                        processed_idxs.append(idx)
                        logger.info(f"Preprocessed image {idx}")
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
                        batch_results[orig_idx] = (
                            normalized_features[j].cpu().numpy().flatten().tolist()
                        )

            except Exception as e:
                logger.error(
                    f"Batch {batch_num}/{total_batches} failed to preprocess: {e}"
                )

            results.extend(batch_results)
            logger.info(f"Batch {batch_num}/{total_batches} processed successfully")

        return results

    def info(self) -> dict:
        """Get information about model"""

        return {
            "model_name": self.config.model_name,
            "device": self.device,
            "feature_dimensions": 512 if "base" in self.config.model_name else 768,
        }
