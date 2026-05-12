import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

load_dotenv()

VECTORSTORE_PATH = Path(__file__).parent / "vectorstore"

st.title("함께학교 챗봇")
st.caption("학교 안내문을 기반으로 질문에 답변합니다.")


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


def ask_gpt(question: str, context: str) -> str:
    client = get_openai_client()
    system_prompt = (
        "당신은 학교 안내 챗봇입니다. "
        "아래 [참고 자료]만 사용해서 질문에 한국어로 답변하세요. "
        "참고 자료에 없는 내용은 '해당 정보를 찾을 수 없습니다'라고 답하세요.\n\n"
        f"[참고 자료]\n{context}"
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content


# 대화 기록 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 이전 대화 출력
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 사용자 입력
if question := st.chat_input("학교 관련 질문을 입력하세요..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("답변 생성 중..."):
            try:
                context = retrieve(question)
                answer = ask_gpt(question, context)
            except FileNotFoundError:
                answer = "인덱스 파일이 없습니다. 먼저 `python src/build_index.py`를 실행해주세요."
            except Exception as e:
                answer = f"오류가 발생했습니다: {e}"
        st.write(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
