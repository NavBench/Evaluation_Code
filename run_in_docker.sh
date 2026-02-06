#!/usr/bin/env bash
# Generic Docker launcher: mounts the repo root to /code inside the container.
# Make sure the image exists first: docker image ls | grep navbench

IMAGE=${IMAGE:-navbench:v2}
CONTAINER_NAME=${CONTAINER_NAME:-navbench-dev}
TZ_VALUE=${TZ:-UTC}

# Repo root directory (directory of this script)
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "[info] Image: $IMAGE"
echo "[info] Container name: $CONTAINER_NAME"
echo "[info] Timezone TZ: $TZ_VALUE"
echo "[info] Mount repo root to /code: $ROOT_DIR"

docker run --rm -it \
  --ipc host \
  --shm-size=1024m \
  -e TZ="$TZ_VALUE" \
  --name "$CONTAINER_NAME" \
  --volume "$ROOT_DIR:/code" \
  -w "/code" \
  ${GPU:+--gpus "$GPU"} \
  --entrypoint /bin/bash \
  "$IMAGE"

# Usage (run this script from the repo root):
#   bash run_in_docker.sh
# Inside the container:
#   # Run Comprehension:
#   python run_eval_comprehension.py --max_items 1
#   # Or run Execution:
#   cd Exec_code
#   bash scripts/gpt4o-easy.sh
# With GPU: GPU=all bash run_in_docker.sh

