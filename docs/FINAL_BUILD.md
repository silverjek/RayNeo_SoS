# Final build record

- Built: 2026-08-27 (Asia/Seoul)
- Project: `/Users/silver/Desktop/sos_glass_final`
- APK: `dist/rayneo-sos-final-debug.apk`
- Application ID: `com.sos.rayneox3.final`
- Version: `1.1-final-safe` (`versionCode=2`)
- ABI: `arm64-v8a`
- SHA-256: `5936ed9b8958f66628e9d65fc241306e80d7c5490ed6fc26f03afb73439631e3`

## Completed checks

- `clean` build succeeded.
- Debug unit tests succeeded.
- APK manifest reports the final application ID, label `RayNeo SoS`, minSdk 29 and targetSdk 36.
- APK contains only the selected 640 FP16 COCO5 B=1/B=3 model assets.
- Experimental GPU probe activity is not exported or packaged from source.
- Existing `/Users/silver/Desktop/sos_glass` experiment project was not modified by finalization.

## Completed on connected glass (2026-08-28)

- APK install, launcher start, camera permission and RAW10 stream open
- OpenCL delegate B=1/B=3 initialization
- Three compute-only repeated runs and three 300-frame actual-app runs
- 1,050/1,050 metadata matches, zero log/image drops, zero thermal throttling
- Clean-cache B=9 hang identified; final K=9 uses 3×B=3

See `experiment_results/final_device_validation_20260828/SUMMARY.md` for measurements.
