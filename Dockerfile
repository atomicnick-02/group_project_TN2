FROM nvidia/cuda:12.3.0-base-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

# ── System dependencies ────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    pkg-config \
    coinor-libipopt-dev \
    libblas-dev \
    liblapack-dev \
    python3.10 python3.10-dev python3.10-venv python3-pip \
    build-essential cmake git wget curl \
    libgl1 libgl1-mesa-glx libgl1-mesa-dev \
    libglfw3 libglfw3-dev \
    libegl1 libegl-dev \
    libegl1-mesa \
    libglu1-mesa libglu1-mesa-dev \
    libgles2-mesa-dev \
    libx11-6 libx11-dev \
    libxrandr-dev libxinerama-dev \
    libxcursor-dev libxi-dev \
    xauth x11-utils \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    libxcb-keysyms1 libxcb-keysyms1-dev \
    xvfb \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ── VirtualGL ──────────────────────────────────────────────────
RUN wget -q https://github.com/VirtualGL/virtualgl/releases/download/3.1/virtualgl_3.1_amd64.deb \
    && dpkg -i virtualgl_3.1_amd64.deb \
    && rm virtualgl_3.1_amd64.deb

# ── Python setup ───────────────────────────────────────────────
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && wget https://bootstrap.pypa.io/get-pip.py \
    && python get-pip.py \
    && rm get-pip.py

WORKDIR /workspace

# ── egl_probe CMake fix ────────────────────────────────────────
ENV CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5"

# ── Clone repos ───────────────────────────────────────────────
# robosuite 1.5.2 — requires MuJoCo 3.x (installed via pip below)
RUN git clone https://github.com/ARISE-Initiative/robosuite.git \
    && cd robosuite \
    && git checkout v1.5.2

# robomimic + mimicgen stay pinned to 1.4.x-compatible commits
# (use the HDF5 patcher + compat wrapper when running mimicgen generation)
RUN git clone https://github.com/ARISE-Initiative/robomimic.git \
    && cd robomimic \
    && git checkout d0b37cf214bd24fb590d182edb6384333f67b661

RUN git clone https://github.com/NVlabs/mimicgen.git \
    && cd mimicgen \
    && git checkout 72bd767c255545f462e7ccfb2731f2e5d4c1d9bb

RUN git clone https://github.com/ARISE-Initiative/robosuite-task-zoo.git \
    && cd robosuite-task-zoo \
    && git checkout 74eab7f88214c21ca1ae8617c2b2f8d19718a9ed

# ── Core pip packages ──────────────────────────────────────────
# MuJoCo 3.x is required by robosuite 1.5.x (replaces mujoco-py)
# gymnasium + sb3 installed here; sb3 >=2.0 targets gymnasium not gym
RUN pip install --upgrade pip \
    && pip install \
        "mujoco>=3.1.1" \
        "gymnasium>=0.29.0" \
        "stable-baselines3>=2.3.0" \
        "shimmy>=1.3.0"\
        "jax"\
        "cyipopt"\
        "dm_control"\
        "matplotlib"
# shimmy provides the gym<->gymnasium compatibility shim
# robomimic/mimicgen internally import old `gym` — shimmy bridges that

# ── Project requirements ───────────────────────────────────────
COPY requirements_clean.txt .
RUN pip install -r requirements_clean.txt

# ── Install editable repos ─────────────────────────────────────
# robosuite first so its MuJoCo 3.x deps are resolved before the others
RUN pip install -e robosuite/
RUN pip install -e robomimic/
RUN pip install -e mimicgen/
RUN pip install -e robosuite-task-zoo/

# ── NVIDIA / rendering config ─────────────────────────────────
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV MUJOCO_GL=egl
ENV DISPLAY=:99
ENV VGL_DISPLAY=:99

# ── Robust runtime display + VirtualGL wrappers ────────────────
# Keep /usr/bin/python* untouched; provide wrappers in /usr/local/bin
# (higher PATH priority) to avoid brittle python*_real assumptions.
RUN printf '%s\n' \
    '#!/usr/bin/env bash' \
    'set -euo pipefail' \
    'display="${DISPLAY:-:99}"' \
    'if ! xdpyinfo -display "$display" >/dev/null 2>&1; then' \
    '  Xvfb "$display" -screen 0 1920x1080x24 +extension GLX +render -noreset >/tmp/xvfb.log 2>&1 &' \
    '  sleep 1' \
    'fi' \
    'exec "$@"' \
    > /usr/local/bin/with-display \
    && chmod +x /usr/local/bin/with-display

RUN printf '%s\n' \
    '#!/usr/bin/env bash' \
    'set -euo pipefail' \
    'pybin="/usr/bin/python3"' \
    'display="${VGL_DISPLAY:-${DISPLAY:-:99}}"' \
    'if command -v vglrun >/dev/null 2>&1 && xdpyinfo -display "$display" >/dev/null 2>&1; then' \
    '  exec vglrun -d "$display" "$pybin" "$@"' \
    'fi' \
    'exec "$pybin" "$@"' \
    > /usr/local/bin/python3 \
    && chmod +x /usr/local/bin/python3 \
    && ln -sf /usr/local/bin/python3 /usr/local/bin/python

ENTRYPOINT ["/usr/local/bin/with-display"]

WORKDIR /workspace