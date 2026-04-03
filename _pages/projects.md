---
layout: single
title: "연구 분야"
permalink: /projects/
author_profile: true
classes:
  - content-page
---

<div class="page-intro">
  <p class="section-kicker">Research Tracks</p>
  <p class="page-lead">
    여기에는 현재 깊게 추적하는 사운드 딥러닝 분야를 정리해 두었습니다.
    개인 프로젝트 소개라기보다, 어떤 오디오 문제를 어떤 관점으로 보고 있는지 드러내는 연구 트랙에 가깝습니다.
  </p>
</div>

<div class="project-grid project-grid--page">
  <section class="content-card project-card tone-signal">
    <p class="project-card__index">01</p>
    <h2>Real-time Speech Enhancement</h2>
    <p>
      DTLN 계열 구조를 참고해 저지연 음성 향상 파이프라인을 다듬는 작업입니다.
      모델 정확도뿐 아니라 실제 스트리밍 환경에서의 체감 품질을 중요하게 봅니다.
    </p>
    <ul class="detail-list">
      <li><strong>Focus:</strong> low latency, denoising, speech clarity</li>
      <li><strong>Stack:</strong> PyTorch, Torchaudio, STFT</li>
      <li><strong>Why it matters:</strong> 실사용 가능한 음성 품질 개선 도구에 가까워지기 위해</li>
    </ul>
    <div class="pill-list">
      <span class="pill">Streaming</span>
      <span class="pill">Speech</span>
      <span class="pill">Torch</span>
    </div>
  </section>

  <section class="content-card project-card tone-sky">
    <p class="project-card__index">02</p>
    <h2>Neural Audio Codec</h2>
    <p>
      오디오를 더 압축하면서도 중요한 질감을 잃지 않도록 신경망 기반 표현 방식을 실험합니다.
      복원 품질과 효율을 동시에 보기 위한 구조를 탐색하는 프로젝트입니다.
    </p>
    <ul class="detail-list">
      <li><strong>Focus:</strong> representation learning, reconstruction, compression</li>
      <li><strong>Stack:</strong> autoencoders, spectrogram features, evaluation tooling</li>
      <li><strong>Why it matters:</strong> 저장과 전송, 생성 파이프라인을 연결하는 기반 기술이 되기 때문에</li>
    </ul>
    <div class="pill-list">
      <span class="pill">Codec</span>
      <span class="pill">Latent</span>
      <span class="pill">Audio</span>
    </div>
  </section>

  <section class="content-card project-card tone-coral">
    <p class="project-card__index">03</p>
    <h2>Music Information Retrieval</h2>
    <p>
      음악과 음향 신호를 검색 가능한 정보로 바꾸는 흐름을 추적합니다.
      분류, 태깅, 구조 분석, 특징 추출처럼 소리를 해석 가능한 데이터로 바꾸는 문제를 다룹니다.
    </p>
    <ul class="detail-list">
      <li><strong>Focus:</strong> feature extraction, tagging, retrieval, structure analysis</li>
      <li><strong>Stack:</strong> Librosa, embedding models, analysis pipelines</li>
      <li><strong>Why it matters:</strong> 소리 이해를 검색과 분류, 제작 지원으로 연결할 수 있기 때문에</li>
    </ul>
    <div class="pill-list">
      <span class="pill">MIR</span>
      <span class="pill">Retrieval</span>
      <span class="pill">Tagging</span>
    </div>
  </section>

  <section class="content-card project-card tone-amber">
    <p class="project-card__index">04</p>
    <h2>Generative and Neural Audio</h2>
    <p>
      생성 모델, representation learning, neural processing 계열에서 실제 오디오 품질과 제어 가능성을 함께 봅니다.
      단순히 멋져 보이는 결과보다, 음향적으로 어떤 의미가 있는지를 중요하게 봅니다.
    </p>
    <ul class="detail-list">
      <li><strong>Focus:</strong> generative audio, latent representations, controllable synthesis</li>
      <li><strong>Stack:</strong> neural codecs, audio embeddings, generation pipelines</li>
      <li><strong>Why it matters:</strong> 생성과 압축, 복원, 표현 학습이 점점 하나의 생태계로 이어지고 있기 때문에</li>
    </ul>
    <div class="pill-list">
      <span class="pill">Generation</span>
      <span class="pill">Representation</span>
      <span class="pill">Neural Audio</span>
    </div>
  </section>
</div>

<section class="content-card section-callout tone-signal">
  <p class="card-kicker">More</p>
  <h2>업데이트는 사운드 딥러닝 범위 안에서만</h2>
  <p>
    이 페이지에는 오디오와 직접 연결되는 분야만 남겨 두었습니다.
    세부 구현과 실험 흔적은 <a href="https://github.com/hshlalla">GitHub 저장소</a>와 글 모음 페이지에서 이어서 정리해 나갈 예정입니다.
  </p>
</section>
