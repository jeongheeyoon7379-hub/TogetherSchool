from pathlib import Path
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DATA_DIR = Path(__file__).parent.parent / "data"
INDEX_PATH = Path(__file__).parent.parent / "vectorstore"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def load_texts() -> list[dict]:
    """data/ 폴더의 모든 .txt 파일을 읽어 {source, text} 목록으로 반환"""
    docs = []
    for txt_file in DATA_DIR.glob("*.txt"):
        text = txt_file.read_text(encoding="utf-8").strip()
        if text:
            docs.append({"source": txt_file.name, "text": text})
    return docs


def chunk_texts(docs: list[dict]) -> tuple[list[str], list[dict]]:
    """텍스트를 청크로 자르고 (청크 내용, 메타데이터) 쌍으로 반환"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks, metas = [], []
    for doc in docs:
        parts = splitter.split_text(doc["text"])
        for i, part in enumerate(parts):
            chunks.append(part)
            metas.append({"source": doc["source"], "chunk": i})
    return chunks, metas


def build_faiss(chunks: list[str], metas: list[dict]) -> FAISS:
    """청크를 OpenAI 임베딩으로 변환해 FAISS 인덱스를 생성"""
    print(f"  OpenAI 임베딩 생성 중... (총 {len(chunks)}개 청크)")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_texts(chunks, embeddings, metadatas=metas)
    return vectorstore


def main():
    print("1단계: txt 파일 로드")
    docs = load_texts()
    if not docs:
        print("  data/ 폴더에 .txt 파일이 없습니다. 먼저 parse_pdfs.py를 실행하세요.")
        return
    for d in docs:
        print(f"  로드됨: {d['source']} ({len(d['text'])}자)")

    print("\n2단계: 청킹")
    chunks, metas = chunk_texts(docs)
    print(f"  청크 수: {len(chunks)}개 (청크 크기: {CHUNK_SIZE}자, 겹침: {CHUNK_OVERLAP}자)")

    print("\n3단계: 임베딩 + FAISS 인덱스 저장")
    vectorstore = build_faiss(chunks, metas)
    vectorstore.save_local(str(INDEX_PATH))
    print(f"  저장 완료: {INDEX_PATH}/")

    print("\n완료! FAISS 인덱스가 vectorstore/ 에 저장되었습니다.")


if __name__ == "__main__":
    main()
