import json
import logging
from model import load_model, run_inference #test_inference
from pprint import pprint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def pose_inference_handler(event, context):
    """Lambda entry point for pose inference"""

    try:
        logger.info("✅ Starting lambda function")
        logger.info("🔄 Loading in model...")
        model = load_model()
        logger.info("✅ Model loaded successfully")

        # logger.info("🔄 Running test inference...")
        # result = test_inference(model)

        logger.info("🔄 Running interence inference...")
        result = run_inference(model, event["bucket"], event["filenames"])
        logger.info(f"✅ Inference finished successfully")

        print("\n===== Inference Result =====")
        pprint(result)
        print("============================\n")

        return {
            "statusCode": 200,
            "body": json.dumps(result)
        }


    except Exception as e:
        logger.error(f"❌ Error during inference: {e}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }