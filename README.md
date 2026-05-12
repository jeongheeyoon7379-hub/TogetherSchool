# 단대소고 함께학교 챗봇

다문화 가정 학생 및 학부모를 위한 **다국어 학교생활 안내 RAG 챗봇**입니다.
학교에서 배포한 가정통신문 PDF를 기반으로 질문에 답변하며, 한국어·영어·베트남어·중국어를 지원합니다.

---

## 주요 기능

| 기능 | 설명 |
|---|---|
| 다국어 답변 | 사용자 언어 설정에 따라 GPT가 해당 언어로 답변 |
| RAG 검색 | FAISS 벡터 검색으로 관련 문서 청크 추출 |
| 출처 표시 | 답변 아래 근거 문서 파일명·청크·관련도% 노출 |
| 회원 인증 | 학번 기반 회원가입/로그인, 비밀번호 SHA-256 해시 저장 |
| 대화 기록 | 세션 내 이전 대화 유지 |

---

## 기술 스택

```
UI          Streamlit
LLM         OpenAI GPT-4o-mini
임베딩       OpenAI text-embedding-3-small
벡터 DB      FAISS
RAG 파이프라인 LangChain
문서 파싱    pypdf
```

---

## 프로젝트 구조

```
TogetherSchool/
├── app.py                  # Streamlit 메인 앱 (랜딩·로그인·챗봇 페이지)
├── src/
│   ├── parse_pdfs.py       # PDF → txt 텍스트 추출
│   └── build_index.py      # txt → 청킹 → 임베딩 → FAISS 인덱스 저장
├── data/                   # 추출된 가정통신문 .txt 파일
├── vectorstore/            # FAISS 인덱스 (index.faiss, index.pkl)
├── pdf/                    # 원본 PDF 파일 (git 제외)
├── users/                  # 회원 정보 JSON (git 제외)
├── requirements.txt
└── .env                    # API 키 (git 제외)
```

---

## 시작하기

### 1. 저장소 클론

```bash
git clone https://github.com/jeongheeyoon7379-hub/TogetherSchool.git
cd TogetherSchool
```

### 2. 가상환경 생성 및 패키지 설치

```bash
python -m venv venv

# Windows
.\venv\Scripts\Activate.ps1

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. 환경변수 설정

프로젝트 루트에 `.env` 파일을 만들고 아래 내용을 입력합니다.

```
OPENAI_API_KEY=sk-...
LLAMA_CLOUD_API_KEY=llx-...  # LlamaParse 사용 시
```

> OpenAI API 키 발급: https://platform.openai.com/api-keys

### 4. PDF 파싱 및 인덱스 빌드

```bash
# pdf/ 폴더에 가정통신문 PDF를 넣은 후 실행
python src/parse_pdfs.py    # PDF → data/*.txt
python src/build_index.py   # txt → vectorstore/
```

### 5. 앱 실행

```bash
streamlit run app.py
```

브라우저에서 http://localhost:8501 접속

---

## 화면 흐름

```
랜딩 페이지 (로그인/회원가입)
    ├── 회원가입 → 학번·이름·언어·비밀번호 입력 → 완료 후 랜딩으로
    └── 로그인  → 학번·비밀번호 확인 → 챗봇 페이지
                        │
                        ▼
              질문 입력 → FAISS 검색 → GPT 답변
                        │
                        ▼
              📎 출처 카드 (파일명 · 청크 · 관련도%)
                  └── expander 열면 원문 청크 표시
```

---

## 개발 기간

2026.04.23 ~ 2026.05.17

## 개발자

단국대학교부속소프트웨어고등학교
