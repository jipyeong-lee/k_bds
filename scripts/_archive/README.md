# scripts/_archive — 폐기 스크립트 (이력 보존)

현행 파이프라인에서 쓰지 않는다. `git mv` 로 이력은 보존. 참조는 docs/worklog·README 이력에만.

## 콜드스타트 빌더 진화 (현행은 `scripts/build_mixed_coldstart.py`)
- `build_sft.py`            — 초기 답-only SFT (gold CoT 없음). 추론 저해로 폐기.
- `build_coldstart_sft.py`  — v1. VLAA clevr_math 단일 → 도메인 단일, 일반화 실패로 폐기.
- `build_rft_coldstart.py`  — v2. 자기증류(RFT) 727건. **필수성은 ablation 으로 입증**했으나
                              데이터 자체 format_think=0.473(느슨한 closed() 게이트 + "가장 짧은 것"
                              선별) 이 RL 형식보상의 천장이었음 → v3 로 대체.

## 기타
- `_autostop_57249.sh`      — job 57249(Stage-2 baseline) 전용 자동중단 스크립트. 일회성.
