---
description: Audio Deep Learning 뉴스 및 연구 업데이트 자동화 워크플로우
---

// 이 워크플로우는 최신 오디오 딥러닝 논문을 수집하고 블로그에 포스팅하는 과정을 정의합니다.

1. **연구 수집**: ArXiv API를 통해 `cs.SD`(사운드)와 `eess.AS`(오디오 및 음성) 카테고리의 최신 논문 5개를 수집합니다.
2. **요약 작업**: 각 논문의 요약을 전문가의 관점에서 한국어로 한국어로 요약합니다. (핵심 기여도, 기술적 특징 포함)
3. **포스트 생성**: `scripts/update_audio_news.py`를 실행하여 `_posts/` 폴더에 `YYYY-MM-DD-audio-deeplearning-research-update.md` 형식으로 파일을 생성합니다.
4. **검토 및 배포**: 생성된 내용을 검토하고 로컬 Jekyll 빌드 확인 후 git push를 통해 배포합니다.
