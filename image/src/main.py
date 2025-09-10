import json
import logging
import time
from model import load_model, run_inference
from pprint import pprint

# Configure logging for Lambda
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def pose_inference_handler(event, context):
    """Lambda entry point for pose inference"""
    start_time = time.time()
    
    try:
        logger.info("✅ Starting lambda function")
        logger.info(f"Event: {json.dumps(event)}")
        logger.info(f"Context: {context}")
        logger.info(f"Remaining time: {context.get_remaining_time_in_millis()}ms")
        
        logger.info("🔄 Loading in model...")
        model_start = time.time()
        model = load_model()
        model_load_time = time.time() - model_start
        logger.info(f"✅ Model loaded successfully in {model_load_time:.2f}s")

        # Check if we have the required event parameters
        if "bucket" not in event or "filenames" not in event:
            raise ValueError("Missing required parameters: 'bucket' and 'filenames' must be provided in the event")

        logger.info(f"🔄 Running inference on {len(event['filenames'])} images from bucket {event['bucket']}...")
        inference_start = time.time()
        result = run_inference(model, event["bucket"], event["filenames"])
        inference_time = time.time() - inference_start
        logger.info(f"✅ Inference finished successfully in {inference_time:.2f}s")

        total_time = time.time() - start_time
        logger.info(f"✅ Total execution time: {total_time:.2f}s")

        print("\n===== Inference Result =====")
        pprint(result)
        print("============================\n")

        return {
            "statusCode": 200,
            "body": json.dumps({
                "results": result,
                "execution_time": {
                    "model_load_time": model_load_time,
                    "inference_time": inference_time,
                    "total_time": total_time
                }
            })
        }

    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"❌ Error during inference after {total_time:.2f}s: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": str(e),
                "error_type": type(e).__name__,
                "execution_time": total_time
            })
        }