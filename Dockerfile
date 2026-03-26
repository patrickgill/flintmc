# syntax=docker/dockerfile:1
FROM python:3.10-slim

# 1. Cache APT packages and lists
# We remove the 'docker-clean' configuration to allow persistent caching of .deb files
RUN rm -f /etc/apt/apt.conf.d/docker-clean; \
    echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y \
    llvm \
    clang \
    build-essential

WORKDIR /app

# 2. Leverage a cache mount for pip
# This prevents re-downloading wheels (like LLVM-based ones) on every build
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip

# 1. Copy the metadata first (good for caching)
COPY pyproject.toml /app/
COPY README.md /app/

# 2. Copy the actual source code (CRITICAL for -e install)
# Replace 'flintmc' with whatever your actual package folder is named
COPY flintmc/ /app/flintmc/

# 3. Now run the install
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -e ".[dev,compat]"

# 4. Finally, copy everything else (tests, docs, etc.)
COPY . /app
