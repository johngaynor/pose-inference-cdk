FROM public.ecr.aws/lambda/python:3.11

# Copy function code
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install dependencies
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt


# Copy all files in ./src
COPY src/ ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler
CMD ["main.pose_inference_handler"]