"""분기 데이터셋: DART 정기보고서에서 분기별 연결 손익과 유류비를 모은다.

유류비는 재무제표 API 에 없고 '비용의 성격별 분류' 주석에만 있다. 보고서 양식이 시기마다 달라서
아래 규칙으로 읽는다.

  - 연결 표를 고른다. (신양식: 캡션에 "(연결)", 구양식: 표 아래 "연결포괄손익계산서상의" 문구)
  - 표의 당분기 칸에서 **첫 번째 숫자 = 3개월(해당 분기) 금액**이다.
    구양식은 "3개월 | 누적" 두 칸이 나란히 오고, 신양식은 당분기/전분기 블록이 따로 온다.
  - 단위는 표 근처의 "(단위: 천원)" 또는 "(단위 : 백만원)" 으로 판별한다.

4분기 값은 공시되지 않으므로 `연간 − (1~3분기 합)` 으로 만든다.
연간 합계와 대조하는 검증은 build_quarterly_fuel() 이 함께 돌려준다.
"""
from __future__ import annotations

import io
import re
import zipfile

from .sources import http

EOK_PER_UNIT = {"천원": 1e5, "백만원": 1e2}   # 해당 단위 → 억원 환산 나눗셈 계수


def report_text(key: str, rcept_no: str) -> list[str]:
    """정기보고서 원문(zip) 안의 파일들을 태그를 걷어낸 한 줄 텍스트로 돌려준다."""
    resp = http.get("https://opendart.fss.or.kr/api/document.xml",
                    {"crtfc_key": key, "rcept_no": rcept_no}, timeout=180)
    out = []
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        for name in z.namelist():
            raw = z.read(name)
            try:
                txt = raw.decode("utf-8")
            except UnicodeDecodeError:
                txt = raw.decode("cp949", errors="replace")
            txt = txt.replace("&cr;", " ")
            out.append(re.sub(r"\s*\n\s*", " ", re.sub(r"[ \t\xa0]+", " ", re.sub(r"<[^>]+>", " ", txt))))
    return out


def _unit_factor(text: str, pos: int) -> float:
    """표 앞쪽에서 가장 가까운 단위 표기를 찾는다."""
    head = text[max(0, pos - 1200):pos]
    units = re.findall(r"\(단위\s*:?\s*(천원|백만원)\)", head)
    return EOK_PER_UNIT[units[-1]] if units else EOK_PER_UNIT["백만원"]


def _is_consolidated(text: str, pos: int) -> bool:
    head, tail = text[max(0, pos - 1500):pos], text[pos:pos + 4000]
    if "비용의 성격별 분류 (연결)" in head or "성격별 분류(연결)" in head:
        return True
    # 구양식은 표 아래 각주로 연결 여부를 밝힌다. 표가 길어 최대 4,000자까지 본다.
    return "연결포괄손익계산서상의" in tail or "연결실체" in tail


def expense_note(texts: list[str], line: str = "연료유류비") -> tuple[float, str] | None:
    """(3개월 금액[억원], 근거 문자열). 연결 표를 찾지 못하면 None."""
    for text in texts:
        for m in re.finditer(line + r"\s*([\d,]+)", text):
            if not _is_consolidated(text, m.start()):
                continue
            factor = _unit_factor(text, m.start())
            value = int(m.group(1).replace(",", "")) / factor
            return value, m.group(0)[:40]
    return None


def build_quarterly_fuel(key: str, reports: dict[str, str], annual_fuel: dict[int, float]) -> tuple[dict, list[str]]:
    """reports: {"2019Q1": rcept_no, ...} (분기·반기 보고서만). 반환: ({분기: 유류비[억원]}, 검증 메모).

    4분기는 연간 값에서 1~3분기를 빼서 만든다.
    """
    quarterly, notes = {}, []
    for label, rcept in sorted(reports.items()):
        found = expense_note(report_text(key, rcept))
        if found is None:
            notes.append(f"{label}: 연결 유류비 표를 찾지 못함 (rcept {rcept})")
            continue
        quarterly[label] = found[0]

    for year, total in annual_fuel.items():
        q13 = [quarterly.get(f"{year}Q{q}") for q in (1, 2, 3)]
        if all(v is not None for v in q13):
            quarterly[f"{year}Q4"] = total - sum(q13)
            notes.append(f"{year}: 연간 {total:,.0f}억 − 1~3분기 {sum(q13):,.0f}억 = 4분기 {total - sum(q13):,.0f}억")
        else:
            notes.append(f"{year}: 1~3분기 중 결측이 있어 4분기를 계산하지 못함")
    return quarterly, notes
