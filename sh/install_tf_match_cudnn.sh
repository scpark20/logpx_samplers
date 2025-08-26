#!/usr/bin/env bash
set -euo pipefail

# ---- 설정(필요시 환경변수로 덮어쓰기) ----------------------------------------
PYTHON_BIN="${PYTHON_BIN:-python3}"
PIP_BIN="${PIP_BIN:-pip3}"
# PREFER_SYSTEM_CUDNN=1  로 주면 시스템에 깔린 cuDNN만 쓰도록 설치(권장 X)
# TF_TARGET=2.20.*       로 주면 자동 매핑을 무시하고 원하는 TF로 고정
# -----------------------------------------------------------------------------

detect_py='
import ctypes, os, subprocess, sys
ver_str = ""
# 1) libcudnn.so에서 버전 직접 읽기 (cudnnGetVersion)
try:
    cudnn = ctypes.cdll.LoadLibrary("libcudnn.so")
    cudnn.cudnnGetVersion.restype = ctypes.c_size_t
    v = int(cudnn.cudnnGetVersion())  # ex) 8907 -> 8.9.7
    maj, min_, patch = v//1000, (v%1000)//100, v%100
    ver_str = f"{maj}.{min_}.{patch}"
except Exception:
    pass
# 2) pip 메타데이터(nvidia-cudnn-cu12)에서 버전 읽기
if not ver_str:
    try:
        out = subprocess.check_output([os.environ.get("PIP_BIN","pip3"), "show", "nvidia-cudnn-cu12"], text=True)
        for line in out.splitlines():
            if line.startswith("Version:"):
                ver_str = line.split(":")[1].strip()
                break
    except Exception:
        pass
print(ver_str)
'

CUDNN_VER="$($PYTHON_BIN -c "$detect_py" || true)"

choose_tf() {
  local mm="$1"  # major.minor
  # 간단 매핑 표 (공식 문서/커뮤니티 정보 기반):
  #  - TF 2.16.1 ↔ CUDA 12.3 + cuDNN 8.9.7  (공식 pip 설치 문서) 
  #  - TF 2.18.x ↔ CUDA 12.5 + cuDNN 9.3   (커뮤니티/가이드 표준)
  #  - TF 2.19.x ↔ CUDA 12.5 + cuDNN 9.3+  (커뮤니티/가이드 표준)
  case "$mm" in
    8.9) echo "${TF_TARGET:-2.16.1}" ;;
    9.0|9.1|9.2|9.3) echo "${TF_TARGET:-2.18.*}" ;;
    9.4|9.5|9.6|9.7|9.8|9.9) echo "${TF_TARGET:-2.19.*}" ;;
    *) echo "${TF_TARGET:-}";;
  esac
}

if [[ -z "$CUDNN_VER" ]]; then
  echo "[info] cuDNN 버전을 찾지 못했습니다(libcudnn.so 미발견)."
  echo "[info] pip가 의존성(CUDA/cuDNN)을 함께 설치하도록 진행합니다."
  TF_SPEC="${TF_SPEC:-tensorflow[and-cuda]}"
else
  echo "[info] 감지된 cuDNN: $CUDNN_VER"
  MAJ="${CUDNN_VER%%.*}"; REST="${CUDNN_VER#*.}"; MIN="${REST%%.*}"
  MM="${MAJ}.${MIN}"
  TFV="$(choose_tf "$MM")"
  if [[ -z "$TFV" ]]; then
    echo "[warn] 알 수 없는 cuDNN 버전($MM). 최신 호환 패키지를 pip에 맡깁니다."
    TF_SPEC="tensorflow[and-cuda]"
  else
    if [[ -n "${PREFER_SYSTEM_CUDNN:-}" ]]; then
      TF_SPEC="tensorflow==${TFV}"          # 시스템에 설치된 cuDNN만 사용 시도
    else
      TF_SPEC="tensorflow[and-cuda]==${TFV}" # pip가 맞는 CUDA/cuDNN까지 설치(권장)
    fi
  fi
fi

echo "[run] pip install: $TF_SPEC"
$PIP_BIN install --upgrade pip
$PIP_BIN install "$TF_SPEC"

# (선택) 시스템 cuDNN만 쓰는 모드에서 GPU 식별 실패 시, 공식 가이드의 심볼릭링크 워크어라운드
if [[ -n "${PREFER_SYSTEM_CUDNN:-}" ]]; then
  $PYTHON_BIN - <<'PY'
import site, subprocess, tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if not gpus:
    tfdir = __import__('tensorflow').__path__[0]
    print("[fix] GPU 미식별 → NVIDIA so를 TF 패키지 폴더로 심볼릭 링크")
    subprocess.call(f"bash -lc 'pushd {tfdir}; ln -svf ../nvidia/*/lib/*.so* .; popd'", shell=True)
    print("[check]", tf.config.list_physical_devices('GPU'))
else:
    print("[ok] GPUs:", gpus)
PY
fi
