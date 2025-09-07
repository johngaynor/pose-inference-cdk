# PhysiQ Pose Inference Lambda

A serverless machine learning application built with AWS CDK that performs pose classification on bodybuilding images using a PyTorch ResNet50 model.

## Overview

This project deploys a Docker-based AWS Lambda function that can classify different bodybuilding poses from images stored in S3. The model recognizes 9 different pose types commonly used in bodybuilding competitions.

### Supported Poses

- Front Relaxed
- Back Relaxed
- Quarter Turn (Left/Right)
- Back Double Biceps
- Front Double Biceps
- Front Lat Spread
- Side Chest (Left)
- Abs & Thighs

## Architecture

- **AWS Lambda**: Serverless compute using Docker container images
- **Docker**: Custom container with PyTorch, torchvision, and pose classification model
- **S3**: Model storage (`physiq-models` bucket) and image input storage
- **CloudWatch**: Logging and monitoring
- **IAM**: Proper permissions for S3 access and Lambda execution

## Project Structure

```
├── bin/                          # CDK app entry point
├── lib/                          # CDK stack definitions
│   └── docker-lambda-aws-stack.ts
├── image/                        # Docker container source
│   ├── Dockerfile
│   ├── requirements.txt
│   └── src/
│       ├── main.py               # Lambda handler
│       └── model/
│           ├── model.py          # PyTorch model wrapper
│           ├── model_utils.py    # Utility functions
│           └── class_mapping.json # Pose class definitions
├── test/                         # Jest unit tests
├── cdk.json                      # CDK configuration
└── package.json                  # Node.js dependencies
```

## Lambda Configuration

- **Function Name**: `pose-inference`
- **Memory**: 3008 MB (optimized for PyTorch model loading)
- **Timeout**: 15 minutes (900 seconds)
- **Runtime**: Docker container with Python 3.9+
- **Permissions**: Full S3 access and CloudWatch logging

## Prerequisites

- Node.js 18+ and npm
- AWS CLI configured with appropriate permissions
- Docker installed and running
- AWS CDK CLI (`npm install -g aws-cdk`)

## Setup and Deployment

1. **Install dependencies**:

   ```bash
   npm install
   ```

2. **Bootstrap CDK** (first time only):

   ```bash
   npx cdk bootstrap
   ```

3. **Build the TypeScript**:

   ```bash
   npm run build
   ```

4. **Deploy the stack**:
   ```bash
   npx cdk deploy
   ```

## Relevant Scripts

### Building the Docker Image Locally

If you want to test the Docker image locally before deploying:

```bash
# Navigate to the image folder
cd image

# Build the Docker image
docker build -t docker-image:test .

# Test the image locally (optional)
docker run --rm docker-image:test
```

### CDK Deployment Commands

```bash
# Deploy the entire stack
cdk deploy

# Deploy with confirmation prompts disabled
cdk deploy --require-approval never

# Deploy and watch the progress
cdk deploy --progress events

# Deploy to a specific AWS profile
cdk deploy --profile your-aws-profile
```

### Other Useful Commands

```bash
# Preview changes before deployment
cdk diff

# Generate CloudFormation template
cdk synth

# Destroy the stack and all resources
cdk destroy

# List all stacks in the app
cdk list
```

## Usage

The Lambda function expects an event with the following structure:

```json
{
  "bucket": "your-image-bucket",
  "filenames": ["image1.jpg", "image2.jpg"]
}
```

### Response Format

```json
[
  {
    "filename": "image1.jpg",
    "result": {
      "predicted_class_index": 0,
      "predicted_class_id": "6",
      "predicted_class_name": "Front Double Biceps",
      "confidence": 0.8945,
      "all_probabilities": {
        "Front Relaxed": 0.0123,
        "Back Relaxed": 0.0098,
        "Front Double Biceps": 0.8945,
        ...
      }
    }
  }
]
```

## Testing the Lambda

You can test the deployed function using the AWS CLI:

```bash
aws lambda invoke \
  --function-name pose-inference \
  --payload '{"bucket":"your-bucket","filenames":["test-image.jpg"]}' \
  --cli-binary-format raw-in-base64-out \
  response.json
```

## Model Details

- **Architecture**: ResNet50 with custom classification head
- **Input Size**: 224x224 pixels
- **Preprocessing**: Standard ImageNet normalization
- **Classes**: 9 bodybuilding pose categories
- **Storage**: Model weights stored in S3 bucket `physiq-models/pose_classifier.pth`

## Development Commands

- `npm run build` - Compile TypeScript to JavaScript
- `npm run watch` - Watch for changes and compile automatically
- `npm run test` - Run Jest unit tests
- `npx cdk deploy` - Deploy stack to AWS
- `npx cdk diff` - Compare deployed stack with current state
- `npx cdk synth` - Generate CloudFormation template
- `npx cdk destroy` - Remove all resources from AWS

## Monitoring

- **CloudWatch Logs**: `/aws/lambda/pose-inference`
- **Metrics**: Function duration, error rate, invocations
- **Debugging**: Check CloudWatch logs for detailed error messages and inference results

## Cost Optimization

- Model caching in `/tmp` directory reduces S3 download costs
- CPU-only PyTorch inference (no GPU required)
- Pay-per-request Lambda pricing model
- Automatic scaling based on demand

## Troubleshooting

### Common Issues

1. **Timeout**: Increase memory allocation if model loading is slow
2. **Out of Memory**: The ResNet50 model requires sufficient RAM
3. **S3 Access**: Verify IAM permissions for model and image buckets
4. **Cold Start**: First invocation may take longer due to model loading

### Logs

Check CloudWatch logs for detailed error messages:

```bash
aws logs tail /aws/lambda/pose-inference --follow
```
