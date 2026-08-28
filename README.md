# RayNeo SoS — Finalized Build

RayNeo X3 Pro (`ARGF20`, Android 12)용 최종 SoS 앱이다. 실험과 중간 버전은 기존
`/Users/silver/Desktop/sos_glass`에 보존하고, 이 프로젝트에는 검증 후 채택한 실행
경로와 필요한 모델만 둔다.

## 기본 실행 경로

```text
Camera2 RAW10 4032×3024
→ SENSOR_TIMESTAMP 기반 RAW/metadata 매칭
→ CFA별 dynamic black-level 차감 + 2× subsampling
→ P-step: N=4 RAW burst / K=9 exposure×gain 후보
   non-P-step: physical single RAW / K=3 gain 후보
→ fused native ARM path
   (first-N 합산 → 2×2 demosaic → digital gain → sRGB OETF
    → 90° 회전 → 640 letterbox/normalization → tensor)
→ 5-class YOLOv8n FP16, OpenCL GPU fixed batch B=3
   (K=9은 안전하게 3×B=3으로 실행)
→ allocation-reuse Decode/NMS → Σconfidence argmax
→ selected candidate만 Bitmap/overlay 생성
→ 다음 sensor request, UI, logging, optional cloud offload
```

## 원본 SoS에서 유지한 기능

- exposure×gain candidate grid와 `P` 주기의 full-grid probing
- burst 합산으로 exposure 후보, digital re-gain으로 gain 후보 생성
- 전체 후보의 batched YOLO 추론과 confidence-sum 기반 최적 cell 선택
- 검출이 전혀 없을 때 entropy/laplacian/tenengrad/Crete/safe-cell fallback
- 선택 exposure를 실제 sensor에 적용하고 다음 step의 anchor로 사용
- Fixed, phone/custom AE, AE-quantized, Proposed 모드
- cross-exposure consistency 기반 비동기 cloud advisory/offload
- P=5/9/12, 회전 턴테이블 시작 동기화, 터치패드 조작
- 후보별 detection JSONL, frame/candidate/router/timing CSV, 선택 JPEG 저장

## RayNeo 최적화

- RayNeo camera 0의 RAW10, 4032×3024, 90° orientation을 사용한다.
- 수신 순서가 아니라 sensor timestamp와 실제 exposure/ISO metadata로 RAW를 승인한다.
- 12-frame 고정 guard 대신 적용 metadata가 확인된 첫 RAW를 사용한다.
- finite single sequence와 다음 burst prefetch로 capture wait를 처리와 겹친다.
- AWB/CCM은 제거하고 원본 SoS와 같은 digital gain + sRGB OETF를 사용한다.
- Bayer sum, demosaic, gain, sRGB, rotation, resize를 하나의 native formation 경로로 합친다.
- 모든 후보의 ARGB/Bitmap을 만들지 않고 tensor로 직접 추론한 뒤 선택 후보만 표시한다.
- tensor/RAW/LUT/decode/NMS 버퍼를 재사용한다.
- 모델은 검증된 640 FP16 COCO5 B=1/3만 포함한다. clean-cache B=9 delegate가 장시간
  정체되는 현상을 피하기 위해 K=9은 3×B=3으로 실행한다. 정확도 저하와 CPU fallback이
  확인된 Full-INT8 512/640 및 NNAPI/NPU 경로는 포함하지 않는다.

COCO5 출력 순서는 `cup`, `wine glass`, `banana`, `bus`, `dining table`이며 앱 내부에서
원래 COCO class ID로 다시 매핑한다.

## 검증된 steady-state 기준

선택 Bitmap/overlay와 비동기 저장을 제외한 detection-ready 기준:

| Step | 평균 지연 |
| --- | ---: |
| K=3 single probing | 약 **0.31 s** |
| K=9 burst probing (`3×B=3`) | 약 **1.02 s** |

새 최종 APK는 독립 빌드와 정적 구성을 확인했다. glass가 현재 연결되어 있지 않아 설치
후 camera/GPU smoke test는 아래 순서로 진행해야 한다.

## 빌드와 설치

```bash
./gradlew :app:assembleDebug --offline
adb -s <DEVICE_SERIAL> install -r app/build/outputs/apk/debug/app-debug.apk
```

- Application ID: `com.sos.rayneox3.final`
- 앱 이름: `RayNeo SoS`
- 기본 설정: Proposed, P=5, Safe fallback, digital boost 2×, confidence 0.25

최초 Start 시 camera와 FP16 GPU B=1/3을 로드·워밍업하므로 첫 실행은 오래 걸릴 수
있다. 이후 처리 시간은 steady-state 값으로 수렴한다.

## 실기 확인 체크리스트

1. 앱 목록에서 `RayNeo SoS`가 보이고 camera 권한 요청이 뜨는지 확인한다.
2. Proposed/P=5 Start 후 `K=9 burst` 1회와 `K=3 single` 4회가 반복되는지 확인한다.
3. 화면 방향, 선택 이미지, bounding box 좌표가 일치하는지 확인한다.
4. `metadata_match=1`이 유지되고 exposure가 선택된 shutter로 바뀌는지 확인한다.
5. GPU backend가 B=1/3 모두 GPU이고 crash/thermal throttling이 없는지 확인한다.
6. 앱 전용 files 디렉터리의 run 폴더에서 `manifest.json`, `frames.csv`, `exp55.csv`,
   `candidates.csv`, `candidate_dets.jsonl`, `router.csv`, `img/`를 확인한다.

세부 구조는 [docs/PIPELINE.md](docs/PIPELINE.md), 터치 조작은
[docs/TOUCHPAD_CONTROLS.md](docs/TOUCHPAD_CONTROLS.md)를 참고한다.
