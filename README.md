# Audio Deep Learning Archive 🎵🤖

오디오와 인공지능의 융합을 연구하고 기록하는 개인 공간입니다.

## 🚀 주요 기능 및 연구 환경

본 프로젝트는 단순한 블로그를 넘어, 인공지능 에이전트와 연동된 **지능형 연구 환경**을 지향합니다.

### 1. 자동화된 연구 브리핑 (Audio DL Updates)
- **arXiv 크롤러**: 매일 최신 오디오 딥러닝(cs.SD) 및 오디오 처리(eess.AS) 논문을 자동으로 수집합니다.
- **Gemini 요약**: 수집된 논문의 핵심 기여도와 오디오 공학적 가치를 한국어로 요약하여 포스팅합니다.
- **GitHub Actions 연동**: 별도의 조작 없이 매일 자동으로 업데이트가 이루어집니다.

### 2. 프리미엄 대시보드 (Agentic Dashboard)
- **실시간 지표**: 에이전트의 효율성과 시스템 가동 상태를 시각화하여 보여줍니다.

### 3. AI 에이전트 워크플로우
- **Workflow Automation**: `.agents/workflows`에 정의된 규격에 따라 에이전트가 복잡한 문서 작업, 포스팅 요약 등을 수행합니다.

## 🛠 기술 스택

- **Static Site Generator**: [Jekyll](https://jekyllrb.com/) (Minimal Mistakes Theme 기반)
- **Styling**: Vanilla CSS (Sass) - Premium Glassmorphism Design
- **Automation**: GitHub Actions & Python
- **AI Engine**: Google Gemini API (2.0 Flash)

## 📁 프로젝트 구조

```text
.
├── .agents/              # AI 에이전트 스킬 및 워크플로우 정의
├── .github/workflows/    # 자동화 (논문 수집 및 배포) 워크플로우
├── _posts/               # 블로그 포스트 저장소 (자동 생성되는 논문 요약 포함)
├── _sass/                # 프리미엄 대시보드 스타일
├── scripts/              # 파이썬 기반 데이터 수집/처리 스크립트
└── index.html            # 메인 대시보드 인터페이스
```

## ⚙️ 로컬 실행 방법

이 저장소는 로컬에서 실행하여 작업을 확인하고 에이전트에게 명령을 내릴 수 있도록 구성되어 있습니다.

```bash
# 종속성 설치 (Jekyll)
bundle install

# 로컬 서버 실행
bundle exec jekyll serve
```

## 📝 라이선스

© 2026 hshlalla. All rights reserved.
이 페이지의 디자인과 소스 코드는 연구 및 학습 목적으로 자유롭게 참고 가능합니다.
