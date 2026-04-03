---
layout: single
title: "GitHub 랩"
permalink: /github-lab/
author_profile: false
classes:
  - content-page
---

{% assign github_sync = site.data.github_sync %}
{% assign github_repo_data = site.data.github_repos %}
{% assign github_repos = github_repo_data.items | default: empty %}

<div class="page-intro">
  <p class="section-kicker">GitHub Sync</p>
  <p class="page-lead">
    공개 GitHub 저장소와 README 스냅샷을 모아 두는 탭입니다.
    새 repo가 생기거나 README가 바뀌면 동기화 워크플로우가 이 페이지 데이터를 다시 갱신합니다.
  </p>
</div>

<div class="content-grid content-grid--split">
  <section class="content-card tone-signal">
    <p class="card-kicker">Source</p>
    <h2>어디를 가져오나</h2>
    <ul class="detail-list">
      <li><strong>대상 계정:</strong> {{ github_sync.username }}</li>
      <li><strong>대상 범위:</strong> 공개 저장소, 설명, 토픽, README, 최근 업데이트 시간</li>
      <li><strong>제외 저장소:</strong> {{ github_sync.exclude_repos | join: ", " }}</li>
      <li><strong>마지막 동기화:</strong> {% if github_repo_data.generated_at %}{{ github_repo_data.generated_at | date: "%Y.%m.%d %H:%M UTC" }}{% else %}아직 동기화 전{% endif %}</li>
    </ul>
  </section>

  <section class="content-card tone-sky">
    <p class="card-kicker">Update Flow</p>
    <h2>어떻게 반영되나</h2>
    <p>
      저장소가 새로 공개되거나 README가 올라오면 GitHub 동기화 워크플로우가 데이터를 다시 가져옵니다.
      이 페이지는 그 결과를 `_data/github_repos.json`에서 읽어 카드 형태로 보여줍니다.
    </p>
    <div class="pill-list">
      <span class="pill">Public repos</span>
      <span class="pill">README excerpts</span>
      <span class="pill">Hourly sync</span>
      <span class="pill">Manual trigger</span>
    </div>
  </section>
</div>

{% if github_repos and github_repos.size > 0 %}
  <div class="repo-grid">
    {% for repo in github_repos %}
      <article class="content-card repo-card tone-{% cycle 'signal', 'sky', 'coral', 'amber' %}">
        <div class="repo-card__header">
          <div>
            <p class="project-card__index">Repo {{ forloop.index }}</p>
            <h2>{{ repo.name }}</h2>
            <p class="repo-card__timestamp">최근 반영 {{ repo.pushed_at | date: "%Y.%m.%d" }}</p>
          </div>
          <a class="text-link" href="{{ repo.repo_url }}">GitHub</a>
        </div>

        {% if repo.description %}
          <p>{{ repo.description }}</p>
        {% else %}
          <p>GitHub 저장소 설명은 비어 있고, README 중심으로 내용을 확인할 수 있습니다.</p>
        {% endif %}

        {% if repo.readme_excerpt %}
          <p class="repo-card__excerpt">{{ repo.readme_excerpt }}</p>
        {% else %}
          <p class="repo-card__excerpt repo-card__excerpt--muted">README가 아직 없거나, 현재 동기화 시점에 본문을 가져오지 못했습니다.</p>
        {% endif %}

        <div class="pill-list">
          {% if repo.language %}<span class="pill">{{ repo.language }}</span>{% endif %}
          <span class="pill">Stars {{ repo.stargazers_count }}</span>
          <span class="pill">Forks {{ repo.forks_count }}</span>
          {% for topic in repo.topics limit: 4 %}
            <span class="pill">{{ topic }}</span>
          {% endfor %}
        </div>

        <div class="repo-card__links">
          {% if repo.readme_html_url %}
            <a class="text-link" href="{{ repo.readme_html_url }}">README 보기</a>
          {% endif %}
          {% if repo.homepage %}
            <a class="text-link" href="{{ repo.homepage }}">배포 링크</a>
          {% endif %}
        </div>
      </article>
    {% endfor %}
  </div>
{% else %}
  <section class="content-card section-callout tone-coral repo-empty">
    <p class="card-kicker">Waiting</p>
    <h2>동기화된 GitHub 저장소가 아직 없습니다</h2>
    <p>
      첫 동기화가 실행되면 여기에 공개 저장소 카드가 생성됩니다.
      워크플로우를 수동으로 한 번 실행하면 바로 채울 수 있습니다.
    </p>
  </section>
{% endif %}
