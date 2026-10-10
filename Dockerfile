FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    #no display inside the container, so matplotlib renders off-screen
    MPLBACKEND=Agg

WORKDIR /project

#installed before copying the code so this layer stays cached when only the code changes
#torch/torchvision come from the CPU-only PyTorch index, which avoids pulling several GB of CUDA libraries
COPY requirements.txt .
RUN pip install -r requirements.txt \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple

COPY app/ app/
#trained weights, CHECKPOINT_PATH resolves to the project root
COPY digit_classifier.pth .
#default test image for predict.py
COPY sample_digit.jpg .

#arguments passed to `docker run` are forwarded to predict.py, e.g. an image path
ENTRYPOINT ["python", "app/predict.py"]
