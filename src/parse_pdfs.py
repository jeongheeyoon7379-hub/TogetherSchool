from pathlib import Path
import pypdf

PDF_DIR = Path(__file__).parent.parent / "pdf"
OUTPUT_DIR = Path(__file__).parent.parent / "data"

OUTPUT_DIR.mkdir(exist_ok=True)


def extract_pdf(pdf_path: Path) -> str:
    reader = pypdf.PdfReader(str(pdf_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def main():
    pdf_files = list(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"pdf/ 폴더에 PDF 파일이 없습니다: {PDF_DIR}")
        return

    for pdf_path in pdf_files:
        print(f"처리 중: {pdf_path.name}")
        text = extract_pdf(pdf_path)

        out_path = OUTPUT_DIR / (pdf_path.stem + ".txt")
        out_path.write_text(text, encoding="utf-8")

        word_count = len(text)
        print(f"  저장 완료: {out_path.name}  ({word_count}자)\n")

    print(f"완료! data/ 폴더에 {len(pdf_files)}개 .txt 파일이 저장되었습니다.")


if __name__ == "__main__":
    main()
