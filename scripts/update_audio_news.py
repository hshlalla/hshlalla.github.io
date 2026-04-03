import os
import urllib.request
import urllib.parse
import json
import xml.etree.ElementTree as ET
from datetime import datetime

# Configuration
ARXIV_URL = "http://export.arxiv.org/api/query?search_query=cat:cs.SD+OR+cat:eess.AS&sortBy=submittedDate&sortOrder=descending&max_results=5"
POSTS_DIR = "_posts"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

def fetch_latest_papers():
    print(f"Fetching: {ARXIV_URL}")
    try:
        with urllib.request.urlopen(ARXIV_URL) as response:
            data = response.read().decode('utf-8')
            root = ET.fromstring(data)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            results = []
            for entry in root.findall("atom:entry", ns):
                title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
                summary = entry.find("atom:summary", ns).text.strip().replace("\n", " ")
                link = entry.find("atom:id", ns).text
                published = entry.find("atom:published", ns).text
                results.append({
                    "title": title,
                    "summary": summary,
                    "url": link,
                    "date": published[:10]
                })
            return results
    except Exception as e:
        print(f"[ERROR] Fetch failed: {e}")
        return []

def summarize_with_gemini(paper):
    if not GEMINI_API_KEY:
        print("[WARNING] GEMINI_API_KEY not found. Using raw abstract.")
        return paper["summary"][:300] + "..."

    prompt = f"다음은 'Audio Deep Learning' 최신 논문 정보입니다.\n제목: {paper['title']}\n요약(영어): {paper['summary']}\n\n이 내용을 한국어로 3줄 요약해줘. 핵심 기여도와 오디오 연구 관점에서의 중요성을 강조해줘."
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    try:
        req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read().decode('utf-8'))
            return result["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        print(f"[ERROR] Gemini summary failed: {e}")
    
    return paper["summary"][:300] + "..."

def create_post(papers):
    today = datetime.now().strftime("%Y-%m-%d")
    filename = f"{today}-audio-deeplearning-research-update.md"
    filepath = os.path.join(POSTS_DIR, filename)

    content = f"""---
layout: single
title: "Audio Deep Learning 최신 연구 브리핑 ({today})"
categories: [Deep Learning, Audio]
tags: [Audio Research, Deep Learning, arXiv]
last_modified_at: {today}
---

오늘 배포된 Audio Deep Learning 및 Audio Signal Processing 분야의 최신 논문 5가지를 요약하여 소개합니다.

"""
    for i, paper in enumerate(papers):
        print(f"Summarizing {i+1}/5: {paper['title'][:50]}...")
        ko_summary = summarize_with_gemini(paper)
        content += f"### {i+1}. {paper['title']}\n"
        content += f"- **원문 링크**: [{paper['url']}]({paper['url']})\n"
        content += f"- **요약**: \n{ko_summary}\n\n---\n\n"

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"[SUCCESS] Created post: {filepath}")

def main():
    print("Fetching latest papers from arXiv...")
    papers = fetch_latest_papers()
    if papers:
        print(f"Found {len(papers)} papers. Summarizing and creating post...")
        create_post(papers)
    else:
        print("No new papers found.")

if __name__ == "__main__":
    main()
