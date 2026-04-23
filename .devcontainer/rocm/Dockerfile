FROM rocm/dev-ubuntu-22.04:6.2.4

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git make cmake clang libclang-dev ninja-build \
    pkg-config libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain stable --profile default
ENV PATH="/root/.cargo/bin:$PATH"

WORKDIR /workspace
