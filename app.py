import hashlib
import json
import math
import os
from datetime import datetime
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

# 학년: 인덱스 순서 = canonical 순서 (0=1학년, 1=2학년, ...)
GRADES_KO   = ["1학년", "2학년", "3학년", "학부모", "교사"]
GRADES_I18N = {
    "한국어":     ["1학년",   "2학년",   "3학년",   "학부모",    "교사"],
    "English":    ["Grade 1", "Grade 2", "Grade 3", "Parent",    "Teacher"],
    "Tiếng Việt": ["Lớp 10",  "Lớp 11",  "Lớp 12",  "Phụ huynh","Giáo viên"],
    "中文":       ["高一",    "高二",    "高三",    "家长",      "教师"],
}

# UI 번역 사전
T: dict[str, dict[str, str]] = {
    "한국어": {
        "app_name": "단대소고 함께학교",
        "app_subtitle": "단국대학교부속소프트웨어고등학교 학교생활 안내 챗봇",
        "app_desc": (
            "학교 공지, 행사, 일정 등을 **본인의 언어**로 쉽게 확인하세요.\n\n"
            "다문화 가정 학생·학부모님을 위해 한국어·영어·베트남어·중국어를 지원합니다."
        ),
        "btn_signup": "회원가입",
        "btn_login": "로그인",
        "btn_back": "← 뒤로가기",
        "btn_logout": "로그아웃",
        "signup_title": "📝 회원가입",
        "login_title": "🔐 로그인",
        "grade_setup_title": "학년을 선택해 주세요",
        "grade_setup_desc": "더 정확한 안내를 드리기 위해 학년 정보가 필요합니다.",
        "btn_confirm": "확인",
        "label_student_id": "학번",
        "label_name": "이름",
        "label_grade": "학년",
        "label_language": "사용 언어",
        "label_pw": "비밀번호 (4자 이상)",
        "label_pw_confirm": "비밀번호 확인",
        "label_pw_login": "비밀번호",
        "chat_placeholder": "학교 관련 질문을 입력하세요...",
        "chat_welcome": "{name}님 환영합니다 👋",
        "spinner": "답변을 생성하고 있습니다...",
        "source_label": "📎 출처",
        "source_chunk": "원문 청크",
        "err_fill_fields": "학번과 이름을 모두 입력해 주세요.",
        "err_pw_short": "비밀번호는 4자 이상이어야 합니다.",
        "err_pw_mismatch": "비밀번호가 일치하지 않습니다.",
        "err_duplicate_id": "이미 가입된 학번입니다.",
        "err_no_user": "존재하지 않는 학번입니다.",
        "err_wrong_pw": "비밀번호가 올바르지 않습니다.",
        "err_no_index": "인덱스 파일이 없습니다. `python src/build_index.py`를 먼저 실행해 주세요.",
        "success_signup": "🎉 회원가입 완료! 로그인해 주세요.",
        "ui_lang_label": "🌐 화면 언어",
    },
    "English": {
        "app_name": "Together School — Dankook SW High",
        "app_subtitle": "School Life Guide Chatbot",
        "app_desc": (
            "Get easy answers about school **notices, events, and schedules** in your language.\n\n"
            "Supports Korean · English · Vietnamese · Chinese."
        ),
        "btn_signup": "Sign Up",
        "btn_login": "Log In",
        "btn_back": "← Back",
        "btn_logout": "Log Out",
        "signup_title": "📝 Sign Up",
        "login_title": "🔐 Log In",
        "grade_setup_title": "Please select your grade",
        "grade_setup_desc": "We need your grade to give you more accurate information.",
        "btn_confirm": "Confirm",
        "label_student_id": "Student ID",
        "label_name": "Name",
        "label_grade": "Grade",
        "label_language": "Preferred Language",
        "label_pw": "Password (min. 4 characters)",
        "label_pw_confirm": "Confirm Password",
        "label_pw_login": "Password",
        "chat_placeholder": "Ask anything about school...",
        "chat_welcome": "Welcome, {name} 👋",
        "spinner": "Generating an answer...",
        "source_label": "📎 Source",
        "source_chunk": "Original Chunk",
        "err_fill_fields": "Please enter your Student ID and Name.",
        "err_pw_short": "Password must be at least 4 characters.",
        "err_pw_mismatch": "Passwords do not match.",
        "err_duplicate_id": "This Student ID is already registered.",
        "err_no_user": "Student ID not found.",
        "err_wrong_pw": "Incorrect password.",
        "err_no_index": "Index not found. Please run `python src/build_index.py` first.",
        "success_signup": "🎉 Registration complete! Please log in.",
        "ui_lang_label": "🌐 Display Language",
    },
    "Tiếng Việt": {
        "app_name": "Trường Cùng Nhau — Dankook SW",
        "app_subtitle": "Chatbot hướng dẫn cuộc sống học đường",
        "app_desc": (
            "Tra cứu **thông báo, sự kiện và lịch học** bằng ngôn ngữ của bạn.\n\n"
            "Hỗ trợ Tiếng Hàn · Tiếng Anh · Tiếng Việt · Tiếng Trung."
        ),
        "btn_signup": "Đăng ký",
        "btn_login": "Đăng nhập",
        "btn_back": "← Quay lại",
        "btn_logout": "Đăng xuất",
        "signup_title": "📝 Đăng ký",
        "login_title": "🔐 Đăng nhập",
        "grade_setup_title": "Vui lòng chọn lớp của bạn",
        "grade_setup_desc": "Chúng tôi cần thông tin lớp để hỗ trợ bạn chính xác hơn.",
        "btn_confirm": "Xác nhận",
        "label_student_id": "Mã học sinh",
        "label_name": "Tên",
        "label_grade": "Lớp",
        "label_language": "Ngôn ngữ sử dụng",
        "label_pw": "Mật khẩu (ít nhất 4 ký tự)",
        "label_pw_confirm": "Xác nhận mật khẩu",
        "label_pw_login": "Mật khẩu",
        "chat_placeholder": "Đặt câu hỏi về trường học...",
        "chat_welcome": "Chào mừng, {name} 👋",
        "spinner": "Đang tạo câu trả lời...",
        "source_label": "📎 Nguồn",
        "source_chunk": "Đoạn văn gốc",
        "err_fill_fields": "Vui lòng nhập mã học sinh và tên.",
        "err_pw_short": "Mật khẩu phải có ít nhất 4 ký tự.",
        "err_pw_mismatch": "Mật khẩu không khớp.",
        "err_duplicate_id": "Mã học sinh đã được đăng ký.",
        "err_no_user": "Không tìm thấy mã học sinh.",
        "err_wrong_pw": "Mật khẩu không đúng.",
        "err_no_index": "Không tìm thấy index. Vui lòng chạy `python src/build_index.py` trước.",
        "success_signup": "🎉 Đăng ký thành công! Vui lòng đăng nhập.",
        "ui_lang_label": "🌐 Ngôn ngữ hiển thị",
    },
    "中文": {
        "app_name": "一起学校 — 单国SW高中",
        "app_subtitle": "学校生活指南聊天机器人",
        "app_desc": (
            "用您的语言轻松查询学校**通知、活动和日程**。\n\n"
            "支持韩语 · 英语 · 越南语 · 中文。"
        ),
        "btn_signup": "注册",
        "btn_login": "登录",
        "btn_back": "← 返回",
        "btn_logout": "退出登录",
        "signup_title": "📝 注册",
        "login_title": "🔐 登录",
        "grade_setup_title": "请选择您的年级",
        "grade_setup_desc": "我们需要您的年级信息以提供更准确的帮助。",
        "btn_confirm": "确认",
        "label_student_id": "学号",
        "label_name": "姓名",
        "label_grade": "年级",
        "label_language": "使用语言",
        "label_pw": "密码（至少4个字符）",
        "label_pw_confirm": "确认密码",
        "label_pw_login": "密码",
        "chat_placeholder": "请输入关于学校的问题...",
        "chat_welcome": "欢迎，{name} 👋",
        "spinner": "正在生成回答...",
        "source_label": "📎 来源",
        "source_chunk": "原文段落",
        "err_fill_fields": "请输入学号和姓名。",
        "err_pw_short": "密码至少需要4个字符。",
        "err_pw_mismatch": "密码不匹配。",
        "err_duplicate_id": "该学号已被注册。",
        "err_no_user": "学号不存在。",
        "err_wrong_pw": "密码不正确。",
        "err_no_index": "未找到索引文件。请先运行 `python src/build_index.py`。",
        "success_signup": "🎉 注册成功！请登录。",
        "ui_lang_label": "🌐 界面语言",
    },
}

LANG_PROMPT = {
    "한국어":     "한국어로 답변해 주세요.",
    "English":    "Please respond in English.",
    "Tiếng Việt": "Vui lòng trả lời bằng tiếng Việt.",
    "中文":       "请用中文回答。",
}


# ── 유틸 ──────────────────────────────────────────────────────────────────────

def t(key: str) -> str:
    """현재 UI 언어 기준으로 번역 문자열 반환"""
    lang = st.session_state.get("ui_lang", "한국어")
    return T.get(lang, T["한국어"]).get(key, key)


def hash_pw(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def user_path(student_id: str) -> Path:
    return USERS_DIR / f"{student_id}.json"


def save_user(student_id: str, name: str, grade: str, language: str, password: str):
    data = {
        "student_id": student_id,
        "name": name,
        "grade": grade,
        "language": language,
        "password_hash": hash_pw(password),
    }
    user_path(student_id).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def update_user_grade(student_id: str, grade: str):
    p = user_path(student_id)
    data = json.loads(p.read_text(encoding="utf-8"))
    data["grade"] = grade
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def load_user(student_id: str) -> dict | None:
    p = user_path(student_id)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def grade_to_canonical(display: str, ui_lang: str) -> str:
    """표시용 학년 → 한국어 정규형 변환"""
    grades = GRADES_I18N.get(ui_lang, GRADES_KO)
    try:
        return GRADES_KO[grades.index(display)]
    except ValueError:
        return display


def go(page: str):
    st.session_state.page = page
    st.rerun()


def render_lang_selector():
    """모든 페이지 상단 우측에 공통으로 들어가는 화면 언어 선택기"""
    _, col_lang = st.columns([4, 1])
    with col_lang:
        current = st.session_state.get("ui_lang", "한국어")
        chosen = st.selectbox(
            t("ui_lang_label"),
            LANGUAGES,
            index=LANGUAGES.index(current),
            label_visibility="collapsed",
            key=f"lang_sel_{st.session_state.get('page', 'landing')}",
        )
        if chosen != current:
            st.session_state.ui_lang = chosen
            st.rerun()


# ── RAG ───────────────────────────────────────────────────────────────────────

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


MIN_RELEVANCE = 0.15  # 이 값 이하의 청크는 컨텍스트에서 제외


def expand_query(question: str, grade: str, history: list) -> str:
    """GPT mini로 질문을 문서 검색에 유리한 키워드로 변환.
    대화 히스토리를 포함해 후속 질문도 올바르게 해석."""
    recent_ctx = "\n".join(
        m["content"] for m in history[-4:] if m["role"] == "user"
    )
    user_msg = f"사용자 학년: {grade}\n이전 대화:\n{recent_ctx}\n현재 질문: {question}"
    response = get_openai_client().chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "학교 공지 문서를 검색하기 위한 핵심 키워드를 추출하세요. "
                    "사용자 학년과 대화 맥락을 고려해 검색에 필요한 학년·주제·날짜 키워드를 포함하세요. "
                    "한국어 키워드만 20단어 이내로 출력하세요. 설명 없이 키워드만."
                ),
            },
            {"role": "user", "content": user_msg},
        ],
        max_tokens=60,
    )
    return response.choices[0].message.content.strip()


def retrieve(question: str, user: dict, k: int = 10) -> tuple[str, list]:
    db = load_vectorstore()
    grade = user.get("grade", "")
    history = st.session_state.get("messages", [])

    # GPT mini가 대화 맥락을 반영한 검색 키워드 생성
    expanded = expand_query(question, grade, history)

    raw = db.similarity_search_with_score(expanded, k=k)
    # exp(-dist)로 관련도 변환 후 임계값 이하 청크 제거
    results = [
        (doc, math.exp(-dist))
        for doc, dist in raw
        if math.exp(-dist) >= MIN_RELEVANCE
    ]

    if not results:  # 모두 임계값 미만이면 상위 2개만 유지
        results = [(doc, math.exp(-dist)) for doc, dist in raw[:2]]

    context = "\n\n".join(
        f"[출처: {doc.metadata.get('source','').removesuffix('.txt')}]\n{doc.page_content}"
        for doc, _ in results
    )
    return context, results


def build_system_prompt(user: dict, context: str) -> str:
    name     = user.get("name", "학생")
    grade    = user.get("grade", "미설정")
    language = user.get("language", "한국어")
    lang_inst = LANG_PROMPT.get(language, LANG_PROMPT["한국어"])

    now = datetime.now().strftime("%Y년 %m월 %d일 %H:%M")
    return (
        f"당신은 단국대학교부속소프트웨어고등학교의 친절한 학교 안내 도우미입니다.\n"
        f"학생과 학부모님이 학교 생활에 잘 적응할 수 있도록 따뜻하고 명확하게 안내해 주세요.\n\n"
        f"[현재 날짜/시간]\n{now}\n\n"
        f"[사용자 정보]\n"
        f"- 이름: {name}\n"
        f"- 학년: {grade}\n\n"
        f"[답변 원칙]\n"
        f"1. {lang_inst} 사용자가 다른 언어로 질문하더라도 반드시 {language}로만 답변하세요.\n"
        f"2. 반드시 아래 [참고 자료]에 있는 내용만 근거로 답변하세요.\n"
        f"3. 참고 자료에 없는 내용은 절대 지어내지 말고, 솔직하게 '해당 정보를 찾을 수 없습니다'라고 안내하세요.\n"
        f"4. 사용자의 학년은 **{grade}** 입니다. 참고 자료에 해당 학년 내용이 있으면 반드시 언급하세요. "
        f"다른 학년 자료만 있다면 '현재 {grade} 자료는 없으나 [해당 학년] 관련 내용이 있습니다' 형태로 안내하세요.\n"
        f"5. {name}님의 이름을 첫 문장에 불러 주며 인사하세요.\n"
        f"6. 날짜·장소·준비물 등 핵심 정보는 글머리 기호(•)로 정리해 주세요.\n"
        f"7. 답변 마지막에 마무리 인사를 붙여 주세요.\n\n"
        f"[참고 자료]\n{context}"
    )


def ask_gpt(user: dict, context: str) -> str:
    system_prompt = build_system_prompt(user, context)
    response = get_openai_client().chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            *[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.get("messages", [])
            ],
        ],
    )
    return response.choices[0].message.content


def render_sidebar_sources(sources: list):
    """사이드바에 출처 카드 표시"""
    if not sources:
        return
    with st.sidebar:
        st.markdown(f"### {t('source_label')}")
        st.caption("가장 최근 답변 기준")
        for i, (doc, score) in enumerate(sources):
            pct   = round(score * 100, 1)
            fname = doc.metadata.get("source", "알 수 없음").removesuffix(".txt")
            st.markdown(f"**{i + 1}. {fname}**")
            st.progress(min(score, 1.0), text=f"관련도 {pct}%")
            with st.expander(t("source_chunk"), expanded=False):
                st.text(doc.page_content)
            if i < len(sources) - 1:
                st.divider()


# ── CSS ───────────────────────────────────────────────────────────────────────

def inject_css():
    st.markdown("""
    <style>
    /* 헤더 카드 */
    .hero {
        background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 60%, #3b82f6 100%);
        padding: 2.2rem 2rem 1.8rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 1.6rem;
    }
    .hero h1 { font-size: 2rem; margin: 0 0 .3rem; }
    .hero p  { font-size: 1rem; opacity: .85; margin: 0; }

    /* 학년 뱃지 — 다크/라이트 모드 모두 대응 */
    .badge {
        display: inline-block;
        background: #2563eb;
        color: #ffffff;
        font-size: .78rem;
        padding: .2rem .65rem;
        border-radius: 999px;
        margin-right: .4rem;
        font-weight: 600;
    }

    /* 구분선 */
    hr { border-color: #e2e8f0 !important; }
    </style>
    """, unsafe_allow_html=True)


# ── 페이지 ────────────────────────────────────────────────────────────────────

def page_landing():
    inject_css()
    render_lang_selector()

    st.markdown(f"""
    <div class="hero">
        <h1>🏫 {t('app_name')}</h1>
        <p>{t('app_subtitle')}</p>
    </div>
    """, unsafe_allow_html=True)

    st.info(t("app_desc"))

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        if st.button(t("btn_signup"), use_container_width=True):
            go("signup")
    with col2:
        if st.button(t("btn_login"), use_container_width=True, type="primary"):
            go("login")


def page_signup():
    inject_css()
    render_lang_selector()
    ui_lang = st.session_state.get("ui_lang", "한국어")

    if st.button(t("btn_back")):
        go("landing")

    st.markdown(f"## {t('signup_title')}")

    with st.container(border=True):
        student_id = st.text_input(t("label_student_id"))
        name       = st.text_input(t("label_name"))
        grade_opts = GRADES_I18N[ui_lang]
        grade_disp = st.selectbox(t("label_grade"), grade_opts)
        language   = st.selectbox(t("label_language"), LANGUAGES,
                                  index=LANGUAGES.index(ui_lang))
        pw         = st.text_input(t("label_pw"), type="password")
        pw_confirm = st.text_input(t("label_pw_confirm"), type="password")

    if st.button(t("btn_signup"), type="primary", use_container_width=True):
        grade_canonical = grade_to_canonical(grade_disp, ui_lang)
        if not student_id or not name:
            st.error(t("err_fill_fields"))
        elif len(pw) < 4:
            st.error(t("err_pw_short"))
        elif pw != pw_confirm:
            st.error(t("err_pw_mismatch"))
        elif user_path(student_id).exists():
            st.error(t("err_duplicate_id"))
        else:
            save_user(student_id, name, grade_canonical, language, pw)
            st.success(t("success_signup"))
            st.session_state.signup_done = True
            st.rerun()

    if st.session_state.pop("signup_done", False):
        go("landing")


def page_login():
    inject_css()
    render_lang_selector()

    if st.button(t("btn_back")):
        go("landing")

    st.markdown(f"## {t('login_title')}")

    with st.container(border=True):
        student_id = st.text_input(t("label_student_id"))
        pw         = st.text_input(t("label_pw_login"), type="password")

    if st.button(t("btn_login"), type="primary", use_container_width=True):
        user = load_user(student_id)
        if user is None:
            st.error(t("err_no_user"))
        elif user["password_hash"] != hash_pw(pw):
            st.error(t("err_wrong_pw"))
        else:
            st.session_state.user     = user
            st.session_state.messages = []
            # 학년 미설정 기존 회원 처리
            if not user.get("grade"):
                go("grade_setup")
            else:
                go("chatbot")


def page_grade_setup():
    """로그인 후 학년이 없는 계정에게 한 번만 물어보는 페이지"""
    inject_css()
    render_lang_selector()
    ui_lang = st.session_state.get("ui_lang", "한국어")

    st.markdown(f"## {t('grade_setup_title')}")
    st.info(t("grade_setup_desc"))

    grade_opts = GRADES_I18N[ui_lang]
    grade_disp = st.selectbox(t("label_grade"), grade_opts)

    if st.button(t("btn_confirm"), type="primary", use_container_width=True):
        grade_canonical = grade_to_canonical(grade_disp, ui_lang)
        update_user_grade(st.session_state.user["student_id"], grade_canonical)
        st.session_state.user["grade"] = grade_canonical
        go("chatbot")


def page_chatbot():
    inject_css()
    render_lang_selector()
    user    = st.session_state.user
    ui_lang = st.session_state.get("ui_lang", "한국어")

    # 상단 헤더
    col_title, col_logout = st.columns([5, 1])
    with col_title:
        st.markdown(f"### 🏫 {t('app_name')}")
    with col_logout:
        if st.button(t("btn_logout"), use_container_width=True):
            st.session_state.clear()
            go("landing")

    # 사용자 뱃지
    grade_display = GRADES_I18N.get(ui_lang, GRADES_KO)
    try:
        grade_label = grade_display[GRADES_KO.index(user.get("grade", ""))]
    except (ValueError, IndexError):
        grade_label = user.get("grade", "")

    welcome = t("chat_welcome").format(name=user["name"])
    st.markdown(
        f'{welcome} &nbsp; <span class="badge">{grade_label}</span>'
        f'<span class="badge">{user["language"]}</span>',
        unsafe_allow_html=True,
    )
    st.divider()

    # 사이드바: 가장 최근 출처 표시
    last = next(
        (m["sources"] for m in reversed(st.session_state.messages)
         if m["role"] == "assistant" and m.get("sources")),
        [],
    )
    render_sidebar_sources(last)

    # 이전 대화 렌더링 (출처 없이)
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # 새 질문 처리
    if question := st.chat_input(t("chat_placeholder")):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        sources = []
        with st.chat_message("assistant"):
            with st.spinner(t("spinner")):
                try:
                    context, sources = retrieve(question, user)
                    answer = ask_gpt(user, context)
                except FileNotFoundError:
                    answer = t("err_no_index")
                except Exception as e:
                    answer = f"오류가 발생했습니다: {e}"
            st.write(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
        render_sidebar_sources(sources)


# ── 라우터 ────────────────────────────────────────────────────────────────────

if "page" not in st.session_state:
    st.session_state.page = "landing"
if "ui_lang" not in st.session_state:
    st.session_state.ui_lang = "한국어"

match st.session_state.page:
    case "landing":
        page_landing()
    case "signup":
        page_signup()
    case "login":
        page_login()
    case "grade_setup":
        page_grade_setup()
    case "chatbot":
        if "user" not in st.session_state:
            go("landing")
        else:
            page_chatbot()
