import os
import json
import logging
import torch
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from typing import Dict
from io import BytesIO

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class Model:
    def __init__(self, s3_client, bucket_name, model_s3_key):
        self.s3_client = s3_client
        self.bucket_name = bucket_name
        self.model_s3_key = model_s3_key
        self.device = torch.device("cpu")  # Lambda CPU-only
        self.model = None
        self.cache_dir = "/tmp"  # Lambda writable directory
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # Load class mapping from JSON
        self._load_class_mapping()

    def _load_class_mapping(self):
        """Load class mapping and human-readable names"""
        CLASS_MAPPING_PATH = os.path.join(os.path.dirname(__file__), "class_mapping.json")
        with open(CLASS_MAPPING_PATH, "r") as f:
            mapping = json.load(f)

        self.class_ids = mapping["classes"]
        self.class_to_idx = mapping["class_to_idx"]
        self.idx_to_class = {int(k): v for k, v in mapping["idx_to_class"].items()}

        pose_name_map = {
            "1": "Front Relaxed",
            "2": "Back Relaxed",
            "5": "Back Double Biceps",
            "6": "Front Double Biceps",
            "7": "Front Lat Spread",
            "11": "Abs & Thighs",
            "14": "Side Chest",
            "15": "Side Tricep",
            "16": "Quarter Turn"
        }
        self.class_names = [pose_name_map.get(cid, f"Unknown Pose {cid}") for cid in self.class_ids]
        logger.info(f"✅ Loaded {len(self.class_names)} classes from mapping")

    def download_model(self) -> str:
        """Download model from S3 if not cached locally"""
        os.makedirs(self.cache_dir, exist_ok=True)
        local_path = os.path.join(self.cache_dir, os.path.basename(self.model_s3_key))

        if os.path.exists(local_path):
            logger.info(f"✅ Using cached model: {local_path}")
            return local_path

        logger.info(f"🔄 Downloading model from s3://{self.bucket_name}/{self.model_s3_key}")
        self.s3_client.download_file(self.bucket_name, self.model_s3_key, local_path)
        logger.info(f"✅ Model downloaded to: {local_path}")
        return local_path

    def load_model(self):
        """Load the model and prepare for predictions"""
        if self.model is not None:
            logger.info("✅ Model already loaded in memory")
            return self.model

        model_path = self.download_model()
        logger.info("🔄 Loading model with memory optimization...")

        try:
            checkpoint = torch.load(model_path, map_location=self.device)

            state_dict = checkpoint.get("model_state_dict", checkpoint)
            num_classes = len(self.class_ids)

            # Build ResNet50
            self.model = models.resnet50(weights=None)
            self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)

            # Load weights (ignore fc if shapes mismatch)
            fc_shape_checkpoint = state_dict["fc.weight"].shape[0]
            if fc_shape_checkpoint == num_classes:
                self.model.load_state_dict(state_dict)
            else:
                filtered_state_dict = {k: v for k, v in state_dict.items() if "fc." not in k}
                self.model.load_state_dict(filtered_state_dict, strict=False)

            self.model.to(self.device)
            self.model.eval()
            logger.info("✅ Model ready")
            return self.model

        except Exception as e:
            self.model = None
            import gc; gc.collect()
            raise Exception(f"Failed to load model: {e}")
        

    def predict_from_file(self, image_path: str) -> Dict:
        """
        Make prediction from an image file path.

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise Exception("Model not loaded.")

        # Open image and convert to RGB
        image = Image.open(image_path).convert('RGB')

        # Apply transforms and add batch dimension
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Make prediction
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

        # Convert results to numpy
        prob_values = probabilities.cpu().numpy()[0]
        predicted_class = int(np.argmax(prob_values))
        confidence = float(prob_values[predicted_class])

        return {
            "predicted_class_index": predicted_class,
            "predicted_class_id": self.idx_to_class.get(predicted_class, str(predicted_class)),
            "predicted_class_name": self.class_names[predicted_class],
            "confidence": confidence,
            "all_probabilities": {
                self.class_names[i]: float(prob_values[i])
                for i in range(len(self.class_names))
            },
            "all_probabilities_with_ids": {
                f"{self.idx_to_class.get(i, str(i))} - {self.class_names[i]}": float(prob_values[i])
                for i in range(len(self.class_names))
            }
        }
    
    def predict_from_s3(self, bucket: str, key: str) -> Dict:
        """
        Make prediction from an image stored in S3.

        Args:
            bucket: S3 bucket name
            key: S3 object key (filename)

        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise Exception("Model not loaded. Call load_model() first.")

        # Download image from S3 into memory
        try:
            s3_response = self.s3_client.get_object(Bucket=bucket, Key=key)
            image_data = s3_response['Body'].read()
        except Exception as e:
            raise Exception(f"Failed to fetch image from S3: {e}")

        # Open image from bytes and convert to RGB
        image = Image.open(BytesIO(image_data)).convert('RGB')

        # Apply transforms and add batch dimension
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

        # Convert results to numpy
        prob_values = probabilities.cpu().numpy()[0]
        predicted_class = int(np.argmax(prob_values))
        confidence = float(prob_values[predicted_class])

        return {
            "predicted_class_index": predicted_class,
            "predicted_class_id": self.idx_to_class.get(predicted_class, str(predicted_class)),
            "predicted_class_name": self.class_names[predicted_class],
            "confidence": confidence,
            "all_probabilities": {
                self.class_names[i]: float(prob_values[i])
                for i in range(len(self.class_names))
            },
            "all_probabilities_with_ids": {
                f"{self.idx_to_class.get(i, str(i))} - {self.class_names[i]}": float(prob_values[i])
                for i in range(len(self.class_names))
            }
        }
