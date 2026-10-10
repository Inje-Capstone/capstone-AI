#!/usr/bin/env bash
# GPU 서버에서 영상 판독 VLM을 직접 띄운다 (NVIDIA 무료 크레딧을 다 쓴 뒤).
#
# 어디서 돌리나: vLLM 0.20.0이 깔린 GPU 머신.
#   - RunPod 등: 템플릿 이미지를 vllm/vllm-openai:v0.20.0 으로 골라 띄우고 이 스크립트 실행
#   - 일반 GPU 서버(docker): HANDOFF.md의 docker run 예시 참고
# GPU: FP8(기본) = L40S 48GB 이상 · BF16 = H100/A100 80GB (MODEL_VARIANT=BF16)
#
# 쓰는 법:
#   VLM_API_KEY=아무-긴-문자열 bash scripts/gpu_serve.sh
#   → 분석 PC에서: NVIDIA_API_KEY=같은-문자열 python scripts/analyze_video.py <ID> game.mp4 --base-url http://<서버IP>:8000/v1
#
# 끝나면 반드시 GPU를 끈다 (켜 둔 시간만큼 과금).
set -euo pipefail

: "${VLM_API_KEY:?VLM_API_KEY가 없다 — 포트가 공개되므로 아무나 못 쓰게 키를 건다}"
VARIANT="${MODEL_VARIANT:-FP8}"
PORT="${PORT:-8000}"

# 영상 속 음성(해설)을 쓰려면 audio 추가 패키지가 필요하다 (vLLM 이미지엔 기본으로 없음).
pip install --quiet "vllm[audio]==0.20.0"

EXTRA=()
if [ "$VARIANT" != "BF16" ]; then
  EXTRA+=(--kv-cache-dtype fp8)
fi

# served-model-name을 호스팅 API의 모델 이름과 같게 둬서 앱 코드는 그대로 쓴다.
exec vllm serve "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-${VARIANT}" \
  --served-model-name nvidia/nemotron-3-nano-omni-30b-a3b-reasoning \
  --trust-remote-code \
  --max-model-len 131072 \
  --tensor-parallel-size 1 \
  --media-io-kwargs '{"video": {"fps": 2, "num_frames": 256}}' \
  --api-key "$VLM_API_KEY" \
  --host 0.0.0.0 --port "$PORT" \
  "${EXTRA[@]}"
