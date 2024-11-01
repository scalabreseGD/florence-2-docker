FROM nvcr.io/nvidia/cuda:12.2.2-devel-ubuntu22.04 AS base

# Set the working directory in the container
WORKDIR /app

RUN apt update && apt-get install -y \
    git \
    python3.10 \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip wheel setuptools

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \

FROM base AS builder

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Final stage to build the lightweight image
FROM base AS final

# Copy installed dependencies from the builder stage
COPY --from=builder /root/.local /root/.local

# Update PATH to use locally installed packages
ENV PATH=/root/.local/bin:$PATH

# Copy the rest of the application code
COPY ./app /app

# Expose the port (dynamic via ENV)
EXPOSE ${PORT:-8000}

# Set environment variable for the port, fallback to 8000
ENV PORT=8000

# Run the Uvicorn app
ENTRYPOINT ["python3", "app.py"]