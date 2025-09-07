import boto3
import logging
from typing import Any, List, Dict
from model.model import Model

logger = logging.getLogger(__name__)
_loader: Model = None

def load_model() -> Model:
    """
    Lazy-loads the model using Model.
    Returns the loaded Model instance.
    """
    global _loader
    if _loader is not None:
        return _loader

    # S3 config from env vars
    bucket_name = "physiq-models"
    model_s3_key = "pose_classifier.pth"

    s3_client = boto3.client("s3")
    _loader = Model(
        s3_client=s3_client,
        bucket_name=bucket_name,
        model_s3_key=model_s3_key,
    )

    _loader.load_model()  # actually download and load
    return _loader


def run_inference(model: Model, bucket: str, filenames: List[str]):
    """
    Runs inference on a list of images and always returns a list of results.
    """
    results = []
    for filename in filenames:
        result = model.predict_from_s3(bucket, filename)
        results.append({"filename": filename, "result": result})
    return results

def test_inference(model: Model):
    """Test function to verify model loading and inference"""
    try:
        filepath = 'test_images/front-double.jpg'
        result = model.predict_from_file(filepath)
        print("✅ Model loaded successfully")
        return result

    except Exception as e:
        print(f"❌ Error during testing: {e}")