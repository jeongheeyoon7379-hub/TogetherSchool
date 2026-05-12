import hashlib
import json
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

load_dotenv()

VECTORSTORE_PATH = Path(__file__).parent / "vectorstore"
USERS_DIR = Path(__file__).parent / "users"
USERS_DIR.mkdir(exist_ok=True)

LANGUAGES = ["한국어", "English", "Tiếng Việt", "中文"]


# ── 유틸 ──────────────────────────────────────────────────────────────────────

def hash_pw(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def user_path(student_id: str) -> Path:
    return USERS_DIR / f"{student_id}.json"


def save_user(student_id: str, name: str, language: str, password: str):
    data = {
        "student_id": student_id,
        "name": name,
        "language": language,
        "password_hash": hash_pw(password),
    }
    user_path(student_id).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def load_user(student_id: str) -> dict | None:
    p = user_path(student_id)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def go(page: str):
    st.session_state.page = page
    st.rerun()


# ── RAG 챗봇 ─────────────────────────────────────────────────────────────────

@st.cache_resource
def load_vectorstore():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.load_local(
        str(VECTORSTORE_PATH),
        embeddings,
        allow_dangerous_deserialization=True,
    )


@st.cache_resource
def get_openai_client():
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def retrieve(question: str, k: int = 3) -> str:
    db = load_vectorstore()
    docs = db.similarity_search(question, k=k)
    return "\n\n".join(doc.page_content for doc in docs)


def ask_gpt(question: str, context: str, language: str) -> str:
    client = get_openai_client()
    lang_instruction = {
        "한국어": "한국어로 답변하세요.",
        "English": "Please answer in English.",
        "Tiếng Việt": "Vui lòng trả lời bằng tiếng Việt.",
        "中文": "请用中文回答。",
    }.get(language, "한국어로 답변하세요.")

    system_prompt = (
        f"당신은 학교 안내 챗봇입니다. {lang_instruction} "
        "아래 [참고 자료]만 사용해서 질문에 답변하세요. "
        "참고 자료에 없는 내용은 '해당 정보를 찾을 수 없습니다'라고 답하세요.\n\n"
        f"[참고 자료]\n{context}"
    )
    response = get_openai_client().chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content


# ── 페이지 ────────────────────────────────────────────────────────────────────

def page_landing():
    st.title("단대소고 함께학교")
    st.markdown(
        """
        **함께학교**는 단국대학교부속소프트웨어고등학교의
        학생·학부모를 위한 **다국어 학교생활 안내 챗봇**입니다.

        학교 행사, 일정, 공지사항 등을 쉽게 검색하고
        본인의 언어로 답변을 받아보세요.
        """
    )
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("회원가입", use_container_width=True):
            go("signup")
    with col2:
        if st.button("로그인", use_container_width=True, type="primary"):
            go("login")


def page_signup():
    st.title("회원가입")

    student_id = st.text_input("학번")
    name = st.text_input("이름")
    language = st.selectbox("사용 언어", LANGUAGES)
    pw = st.text_input("비밀번호 (4자 이상)", type="password")
    pw_confirm = st.text_input("비밀번호 확인", type="password")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("뒤로가기", use_container_width=True):
            go("landing")
    with col2:
        if st.button("회원가입", use_container_width=True, type="primary"):
            if not student_id or not name:
                st.error("학번과 이름을 입력해주세요.")
            elif len(pw) < 4:
                st.error("비밀번호는 4자 이상이어야 합니다.")
            elif pw != pw_confirm:
                st.error("비밀번호가 일치하지 않습니다.")
            elif user_path(student_id).exists():
                st.error("이미 가입된 학번입니다.")
            else:
                save_user(student_id, name, language, pw)
                st.success("회원가입이 완료되었습니다!")
                st.session_state.signup_done = True
                st.rerun()

    if st.session_state.get("signup_done"):
        st.session_state.pop("signup_done")
        go("landing")


def page_login():
    st.title("로그인")

    student_id = st.text_input("학번")
    pw = st.text_input("비밀번호", type="password")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("뒤로가기", use_container_width=True):
            go("landing")
    with col2:
        if st.button("로그인", use_container_width=True, type="primary"):
            user = load_user(student_id)
            if user is None:
                st.error("존재하지 않는 학번입니다.")
            elif user["password_hash"] != hash_pw(pw):
                st.error("비밀번호가 올바르지 않습니다.")
            else:
                st.session_state.user = user
                st.session_state.messages = []
                go("chatbot")


def page_chatbot():
    user = st.session_state.user
    st.title("함께학교 챗봇")
    st.caption(f"{user['name']}님 환영합니다 · 사용 언어: {user['language']}")

    if st.button("로그아웃"):
        st.session_state.clear()
        go("landing")

    st.divider()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if question := st.chat_input("학교 관련 질문을 입력하세요..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("답변 생성 중..."):
                try:
                    context = retrieve(question)
                    answer = ask_gpt(question, context, user["language"])
                except FileNotFoundError:
                    answer = "인덱스 파일이 없습니다. 먼저 `python src/build_index.py`를 실행해주세요."
                except Exception as e:
                    answer = f"오류가 발생했습니다: {e}"
            st.write(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})


# ── 라우터 ────────────────────────────────────────────────────────────────────

if "page" not in st.session_state:
    st.session_state.page = "landing"

match st.session_state.page:
    case "landing":
        page_landing()
    case "signup":
        page_signup()
    case "login":
        page_login()
    case "chatbot":
        if "user" not in st.session_state:
            go("landing")
        else:
            page_chatbot()
