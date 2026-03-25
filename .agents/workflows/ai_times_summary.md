---
description: AI Times 뉴스 요약 및 포스팅 자동화 워크플로우
---

// 이 워크플로우는 AI 에이전트가 외부 뉴스를 수집하고 콘텐츠를 생성하는 과정을 정의합니다.

1. **뉴스 수집**: AI Times 또는 관련 기술 뉴스 사이트에서 최신 기능/소식 5개를 수집합니다.
2. **요약 작업**: 각 기능을 전문가의 관점에서 한국어로 요약합니다. (핵심 이점, 기술적 특징 포함)
3. **포스트 생성**: `_posts/` 폴더에 `YYYY-MM-DD-ai-times-summary.md` 형식으로 파일을 생성합니다.
4. **메타데이터 설정**: 
   - layout: single
   - title: "AI Times 주요 업데이트 요약 (5가지)"
   - categories: [AI News, Sound Tech]
5. **검토 및 배포**: 생성된 내용을 검토하고 로컬 Jekyll 빌드 확인 후 git push를 통해 배포합니다.
