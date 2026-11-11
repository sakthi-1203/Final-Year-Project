"""
PDF -> 5 QnA pairs per paper -> llama2 style parquet.

"""
import argparse, json, re, time
from pathlib import Path

import fitz  
import ollama
import pandas as pd
from tqdm import tqdm

MODEL = "llama3.2:3b"
N_QNA = 5
CHUNK_WORDS = 500
OPTIONS = {"temperature": 0.3, "num_ctx": 2048, "num_predict": 200}

Q_PROMPT = (
    'Give me a short question starting with What, How or Why, to which the answer is "{txt}". '
    "Don't include the answer in the question. Answer as a question, one sentence. "
    "Output only the question."
)
A_PROMPT = (
    'Using ONLY the text below, answer the question. Write a succinct summary of key concepts of how '
    '"{txt}" answers "{q}". The summary must stand on its own, so never refer to \'the text\', '
    "'the paper' or 'the authors'. Do not include equations or citation markers. "
    "Keep key material names and quantitative values if they are stated. "
    "Use 3 sentences or fewer. Output only the summary."
)

# Chunks containing 2 or more of these are skipped (licence text, funding, acknowledgements...)
BOILERPLATE = ("creative commons", "licen", "copyright", "acknowledg", "conflict of interest",
               "funding", "author contributions", "supporting information", "data availability",
               "orcid", "received:", "accepted:", "publisher")
# Pairs whose QUESTION mentions any of these are rejected
BAD_QUESTION_WORDS = ("licen", "copyright", "funding", "creative commons")


def llm(prompt: str) -> str:
    r = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}],
                    options=OPTIONS, keep_alive="30m")
    return r["message"]["content"].strip()


def is_boilerplate(chunk: str) -> bool:
    low = chunk.lower()
    return sum(k in low for k in BOILERPLATE) >= 2


def pdf_to_text(path: Path) -> str:
    doc = fitz.open(path)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    # drop reference section (last occurrence of a References heading)
    m = list(re.finditer(r"\n\s*(references|bibliography)\s*\n", text, re.I))
    if m and m[-1].start() > len(text) * 0.5:
        text = text[: m[-1].start()]
    text = re.sub(r"-\n(\w)", r"\1", text)
    text = re.sub(r"\[\d+(?:[,\u2013\-]\s*\d+)*\]", "", text)
    text = re.sub(r"\b(Figure|Fig\.|Table|Chapter|Schematic)\s*\d*\.?", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def pick_chunks(text: str, n: int = N_QNA):
    words = text.split()
    chunks = [" ".join(words[i:i + CHUNK_WORDS]) for i in range(0, len(words), CHUNK_WORDS)]
    chunks = [c for c in chunks if len(c.split()) > 200 and not is_boilerplate(c)]
    if len(chunks) <= n:
        return chunks
    # evenly spaced across the paper (intro -> methods -> results -> discussion)
    step = (len(chunks) - 1) / (n - 1)
    return [chunks[round(i * step)] for i in range(n)]


def formate(q: str, a: str) -> str:
    # exact format used by mlabonne/guanaco-llama2
    return f"<s>[INST] {q} [/INST] {a} </s>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf_dir", default="papers")
    ap.add_argument("--out_dir", default="output")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(exist_ok=True)
    jsonl, donefile = out / "qna_raw.jsonl", out / "processed.txt"

    # papers already handled (including ones that produced 0 pairs) are skipped on resume
    done = set(donefile.read_text(encoding="utf-8").splitlines()) if donefile.exists() else set()

    pdfs = sorted(Path(args.pdf_dir).glob("*.pdf"))
    if args.limit:
        pdfs = pdfs[: args.limit]

    for pdf in tqdm(pdfs, desc="papers"):
        if pdf.name in done:
            continue
        t0, kept, tried = time.time(), 0, 0
        try:
            chunks = pick_chunks(pdf_to_text(pdf))
            with open(jsonl, "a", encoding="utf-8") as f:
                for c in chunks:
                    tried += 1
                    q = llm(Q_PROMPT.format(txt=c)).strip('" ')
                    if any(k in q.lower() for k in BAD_QUESTION_WORDS):
                        continue
                    a = llm(A_PROMPT.format(txt=c, q=q))
                    if len(q) < 10 or len(a) < 30:
                        continue
                    f.write(json.dumps({"source": pdf.name, "question": q, "answer": a,
                                        "text": formate(q, a)}, ensure_ascii=False) + "\n")
                    f.flush()
                    kept += 1
            with open(donefile, "a", encoding="utf-8") as f:
                f.write(pdf.name + "\n")
        except Exception as e:
            tqdm.write(f"[skip] {pdf.name}: {e}")
            continue
        tqdm.write(f"{pdf.name[:50]}: kept {kept}/{tried} chunks in {time.time() - t0:.0f}s")

    if not jsonl.exists():
        print("No pairs generated."); return
    # build ONE parquet with a single "text" column, like guanaco-llama2
    rows = [json.loads(l) for l in jsonl.read_text(encoding="utf-8").splitlines() if l]
    df = pd.DataFrame({"text": [r["text"] for r in rows]})
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_parquet(out / "biomaterials_qna.parquet", index=False)
    print(f"{len(df)} QnA pairs -> {out / 'biomaterials_qna.parquet'}")


if __name__ == "__main__":
    main()