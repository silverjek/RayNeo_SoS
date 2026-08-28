# 논문 및 RayNeo 최종 구현 대응표

| 요구사항 | 최종 구현 | 구분 |
| --- | --- | --- |
| exposure×gain full-grid probe | N=4 RAW로 K=9 형성 | 원본 SoS 유지 |
| gain-only step | 선택 shutter의 physical N=1 RAW로 K=3 형성 | 원본 SoS 유지 |
| `P` 주기 shutter 재탐색 | P=5/9/12 | 원본 SoS 유지 |
| candidate 선택 | threshold 이상 `Σconfidence` argmax | 원본 SoS 유지 |
| all-zero fallback | 5종 metric 또는 safe-cell | 기존 앱 기능 유지 |
| batched detector | YOLOv8n 640 FP16 GPU B=3; K=9은 3×B=3 | 원본 원리 유지 |
| non-blocking advisory cloud | cross-exposure router + async HTTP | 원본 SoS 유지 |
| camera 입력 | camera 0, RAW10 4032×3024 | RayNeo 적응 |
| RAW/result 결합 | `SENSOR_TIMESTAMP` + exposure/ISO 승인 | RayNeo 적응 |
| black correction | CFA 위치별 dynamic black level | RayNeo 적응 |
| setting 적용 | 고정 12-frame 대기 없이 일치 metadata의 첫 RAW | RayNeo 최적화 |
| capture schedule | finite singles + next burst prefetch | RayNeo 최적화 |
| image formation | fused first-N sum→demosaic→gain→sRGB→rotate→640 tensor | 최적화 |
| 색 처리 | AWB/CCM 미적용 | 원본 처리와 통일 |
| candidate representation | tensor direct; 선택 후보만 Bitmap | 최적화 |
| decode/NMS | 출력 복사 제거, 배열/heap 재사용, pre-NMS Top-K | 최적화 |
| detector domain | COCO5 head + 원 COCO ID remap | 검증 후 채택 |

저신뢰 detection tail(`confidence >= 0.01`)은 candidate JSONL과 offload 판단에 남지만,
화면·selection에는 UI의 operating threshold(기본 0.25)를 적용한다.

## 색과 방향

```text
CFA black subtraction
→ first-N Bayer sum
→ 2×2 block demosaic
→ digital gain + clipping
→ sRGB OETF
→ SENSOR_ORIENTATION=90° 회전
→ 640×640 letterbox + [0,1] normalization
```

AWB와 camera color correction matrix는 적용하지 않는다. 이는 green-cast 보정을 위해
추가했던 RayNeo 전용 A 경로를 제거하고, 원본 SoS의 minimal image formation과 통일한
결과다. Tensor에 들어간 640 image를 화면용으로 재사용하며 detection box도 같은
letterbox 좌표로 변환한다.

## 모델 결정

- 채택: COCO5 YOLOv8n, 640, FP16, TFLite GPU OpenCL, fixed B=1/3.
- B=9 asset은 hot 상태에서 약간 빠르지만 새 패키지의 clean cache에서 delegate 생성이
  장시간 정체되어 제외했다. K=9은 동일 B=3 모델을 세 번 실행한다.
- 제외: Full INT8 640은 FP16 기준 detection regression이 있었고 GPU도 빨라지지 않았다.
- 제외: Full INT8 512는 GPU latency는 줄었지만 recall/confidence 안정성이 더 낮았다.
- 제외: NNAPI 요청 시 대부분의 operator가 XNNPACK CPU로 fallback되어 Qualcomm NPU를
  실질적으로 사용하지 못했다.

## 저장 위치

권한이 필요 없는 앱 전용 외부 저장소 아래에 run별로 기록한다.

```text
/sdcard/Android/data/com.sos.rayneox3.final/files/sos/run_.../
```

주요 파일은 `manifest.json`, `frames.csv`, `exp55.csv`, `candidates.csv`,
`candidate_dets.jsonl`, `router.csv`, `dets.jsonl`, `img/`이다.

## 남은 실기 검증

현재 glass가 연결되어 있지 않아 final package에서는 다음이 아직 필요하다.

- 새 application ID의 launcher 노출 및 camera permission
- OpenCL B=1/3 warm cache와 K=9의 3×B=3 실행
- 300-frame Proposed P=5 smoke/soak
- metadata match, 방향, preview-box 좌표, async log completeness
- 선택 후보 Bitmap 생성이 steady-state detection throughput에 주는 작은 후처리 영향
