import boto3
import logging
import time
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
        logger.info("✅ Using cached model instance")
        return _loader

    start_time = time.time()
    logger.info("🔄 Initializing new model instance...")

    # S3 config from env vars
    bucket_name = "physiq-models"
    model_s3_key = "pose_classifier.pth"
    
    logger.info(f"Model location: s3://{bucket_name}/{model_s3_key}")

    try:
        s3_client = boto3.client("s3")
        logger.info("✅ S3 client created")
        
        _loader = Model(
            s3_client=s3_client,
            bucket_name=bucket_name,
            model_s3_key=model_s3_key,
        )
        logger.info("✅ Model instance created")

        _loader.load_model()  # actually download and load
        load_time = time.time() - start_time
        logger.info(f"✅ Model loaded and ready in {load_time:.2f}s")
        return _loader
        
    except Exception as e:
        load_time = time.time() - start_time
        logger.error(f"❌ Failed to load model after {load_time:.2f}s: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


def run_inference(model: Model, bucket: str, filenames: List[str]):
    """
    Runs inference on a list of images and always returns a list of results.
    """
    logger.info(f"🔄 Running inference on {len(filenames)} files from bucket '{bucket}'")
    results = []
    
    for i, filename in enumerate(filenames):
        try:
            logger.info(f"🔄 Processing image {i+1}/{len(filenames)}: {filename}")
            start_time = time.time()
            result = model.predict_from_s3(bucket, filename)
            inference_time = time.time() - start_time
            logger.info(f"✅ Processed {filename} in {inference_time:.2f}s")
            results.append({"filename": filename, "result": result, "inference_time": inference_time})
        except Exception as e:
            logger.error(f"❌ Error processing {filename}: {str(e)}")
            results.append({"filename": filename, "error": str(e), "inference_time": 0})
    
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