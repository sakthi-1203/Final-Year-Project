"""
Cleans output/qna_raw.jsonl -> output/biomaterials_qna_clean.parquet (+ output/rejected.csv for review).

Usage (from the project folder, env active):
    python clean_dataset.py
"""
import json, re
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

IN = Path("output/qna_raw.jsonl")
OUT_PARQUET = Path("output/biomaterials_qna_clean.parquet")
OUT_JSONL = Path("output/biomaterials_qna_clean.jsonl")
OUT_REJECTED = Path("output/rejected.csv")

NON_ANSWER = re.compile(
    r"i can'?t|i cannot|not (explicitly )?(specified|stated|mentioned|provided|directly stated)|"
    r"no (specific|quantitative)|does not (mention|specify|state)|no mention|provided text|"
    r"the text (does|do) not|^here is|^sure|^certainly", re.I)

CITATION_Q = re.compile(
    r"\btitle of\b|\bauthors?\b|affiliation|\bjournal\b|published in|publication|listed (research )?(papers|publications)|"
    r"cited papers|common (thread|denominator|theme)|supplied the|company that|\bdoi\b|et al", re.I)

METHODS_Q = re.compile(
    r"\bsoftware\b|microscope|spectrophotometer|what equipment|\bresolution of the\b|significance level|p-value|"
    r"statistical (test|analysis)|scale bar|magnification|centrifug|passag|antibod|supplier|sputter|"
    r"what instrument|which instrument|what type of (microscope|instrument)|rheometer used|how many (rats|mice|samples)|"
    r"number of (rats|mice|samples)", re.I)

EQUATION_Q = re.compile(r"\bequation\b|\bformula\b|\beq\.|\beqs\.", re.I)
STUDY_REF_Q = re.compile(r"\b(in|of|from|by) (this|the) (study|work|paper|article|research)\b|\bthis study\b|\bthe study\b", re.I)

# Known wrong acronym expansions made by the small model
KNOWN_BAD = re.compile(
    r"poly\(lactic-co-glycolic acid\)\s*\(PCL\)|poly\(lactic acid\)\s*\(PCL\)|synthetic fiber\s*\(SF\)|"
    r"Digital Inkjet Writing|Laser-Printed Bioprinted|nanobenzoylglutamide|oxalic acid glycerol", re.I)
GARBLED = re.compile(r"[\u0002\u00f0\u00fe\u00bc]")

MAX_Q_LEN, MIN_A_LEN, MAX_A_LEN = 220, 60, 1200
NEAR_DUP = 0.85


def reason(r):
    q, a = r["question"].strip(), r["answer"].strip()
    if NON_ANSWER.search(a):            return "non-answer"
    if CITATION_Q.search(q):            return "citation/reference question"
    if EQUATION_Q.search(q) or GARBLED.search(q + a): return "equation/garbled"
    if KNOWN_BAD.search(q + " " + a):   return "known wrong acronym"
    if METHODS_Q.search(q):             return "methods trivia"
    if STUDY_REF_Q.search(q):           return "refers to 'the study'"
    if len(q) > MAX_Q_LEN:              return "question too long"
    if len(a) < MIN_A_LEN or len(a) > MAX_A_LEN: return "answer length"
    return None


def main():
    rows = [json.loads(l) for l in IN.read_text(encoding="utf-8").splitlines() if l.strip()]
    kept, rejected, seen_q = [], [], {}
    for r in rows:
        why = reason(r)
        if why is None:
            qn = re.sub(r"\W+", " ", r["question"].lower()).strip()
            for prev in seen_q.get(r["source"], []):
                if qn == prev or SequenceMatcher(None, qn, prev).ratio() > NEAR_DUP:
                    why = "duplicate/near-duplicate"
                    break
            if why is None:
                seen_q.setdefault(r["source"], []).append(qn)
        (rejected if why else kept).append({**r, "reason": why} if why else r)
    pd.DataFrame({"text": [r["text"] for r in kept]}).sample(frac=1, random_state=42) \
    .reset_index(drop=True).to_parquet(OUT_PARQUET, index=False)
    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for r in kept: 
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    pd.DataFrame(rejected)[["reason", "source", "question", "answer"]].to_csv(OUT_REJECTED, index=False, encoding="utf-8-sig")

    print(f"Total {len(rows)} | kept {len(kept)} | rejected {len(rejected)}")
    print(pd.Series([r["reason"] for r in rejected]).value_counts().to_string())
    print(f"\nSaved {OUT_JSONL} and {OUT_REJECTED}")


if __name__ == "__main__":
    main()
