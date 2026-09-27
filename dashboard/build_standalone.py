"""대시보드를 독립 실행용 HTML 로 감싼다.

dashboard/index.html 은 Claude Artifact 용이라 <html>·<head>·<body> 없이 본문만 담는다.
Vercel 같은 정적 호스팅에 올리려면 문서 골격과 charset·viewport·기본 리셋이 필요하므로
같은 내용을 감싸서 dashboard/dist/index.html 을 만든다. 원본은 하나로 유지한다.

실행: python dashboard/build_standalone.py
"""
from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).parent / "index.html"
OUT = Path(__file__).parent / "dist" / "index.html"

HEAD_TAGS = re.compile(r"<title\b[^>]*>.*?</title>|<style\b[^>]*>.*?</style>|<link\b[^>]*>", re.S | re.I)

SHELL_HEAD = """<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="description" content="항공유 가격·환율·수송 물량을 바꿔 대한항공 연간 영업이익을 계산하는 시나리오 대시보드">
<meta property="og:title" content="항공사 연료 시나리오">
<meta property="og:description" content="유가·환율·물량이 바뀌면 영업이익은 어떻게 되나 — 대한항공 2019~2025년 실적 기준">
<style>
  :root { color-scheme: light dark; padding-top: env(safe-area-inset-top, 0px); padding-bottom: env(safe-area-inset-bottom, 0px); }
  * { box-sizing: border-box; }
  body { margin: 0; font-size: 14px; }
  img { max-width: 100%; }
  [hidden] { display: none !important; }
</style>
"""


def build() -> Path:
    src = SRC.read_text(encoding="utf-8")
    head_parts: list[str] = []      # <title>, <link>, <style> → head
    body_parts: list[str] = []      # 나머지 → body
    pos = 0
    for m in HEAD_TAGS.finditer(src):
        chunk = src[pos:m.start()].strip()
        if chunk:
            body_parts.append(chunk)
        head_parts.append(m.group(0))
        pos = m.end()
    tail = src[pos:].strip()
    if tail:
        body_parts.append(tail)

    html = (SHELL_HEAD + "\n".join(head_parts) + "\n</head>\n<body>\n"
            + "\n".join(body_parts) + "\n</body>\n</html>\n")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    return OUT


if __name__ == "__main__":
    out = build()
    print(f"생성: {out}  ({out.stat().st_size:,} bytes)")
