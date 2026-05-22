FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# System deps for gymnasium (mujoco, atari, opencv)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libosmesa6-dev \
    libgl1-mesa-dev \
    libglfw3 \
    patchelf \
    swig \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /workspace
RUN pip install --no-cache-dir -e .

# Default to a quick CartPole run as smoke test
CMD ["python", "-m", "ppo.ppo", "--env-id", "CartPole-v1", "--total-timesteps", "100000"]
