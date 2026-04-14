import os
from typing import List
from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from services.rag_pipeline import get_rag_app

rag_app = get_rag_app()


# -----------------------------
# Session helpers
# -----------------------------
def init_session():
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "last_documents" not in st.session_state:
        st.session_state.last_documents = []
    if "thread_turns" not in st.session_state:
        st.session_state.thread_turns = 0
    if "last_debug" not in st.session_state:
        st.session_state.last_debug = {}
    if "last_node_logs" not in st.session_state:
        st.session_state.last_node_logs = []


def render_message(msg: BaseMessage):
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)


def render_sources(docs: List[Document]):
    st.subheader("참조 문서")

    if not docs:
        st.caption("이번 턴에서 사용된 문서가 없습니다.")
        return

    for idx, doc in enumerate(docs, start=1):
        title = doc.metadata.get("title", f"문서 {idx}")
        source = doc.metadata.get("source", "-")
        if source == 'web':
            source = doc.metadata.get("url", "-")
        score = doc.metadata.get("score")

        with st.container(border=True):
            st.markdown(f"**[{idx}] {title}**")
            meta_line = f"출처: `{source}`"
            if score is not None:
                try:
                    meta_line += f" · score: `{float(score):.3f}`"
                except Exception:
                    meta_line += f" · score: `{score}`"
            st.caption(meta_line)

            snippet = doc.page_content[:300]
            st.write(snippet + ("..." if len(doc.page_content) > 300 else ""))

            if len(doc.page_content) > 300:
                with st.expander("전체 내용 보기"):
                    st.write(doc.page_content)


def render_debug(debug: dict):
    st.subheader("실행 정보")

    if not debug:
        st.caption("아직 실행 정보가 없습니다.")
        return

    st.json(debug, expanded=False)


def render_node_logs(logs: list):
    st.subheader("노드 로그")

    if not logs:
        st.caption("이번 턴의 노드 로그가 없습니다.")
        return

    for i, log in enumerate(logs, start=1):
        with st.container(border=True):
            st.markdown(f"**[{i}] {log.get('node', '-')}**")
            st.caption(log.get("status", ""))
            if log.get("detail"):
                st.write(log["detail"])


def get_last_ai_message(messages: list[BaseMessage]):
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return msg
    return None


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="IRIS RAG Chat",
    layout="wide",
    initial_sidebar_state="collapsed",
)

init_session()

st.title("IRIS RAG Chat")
st.caption("아이리스 제품 문서를 기반으로 답변합니다.")

with st.sidebar:
    st.markdown("### graph")
    st.image("/Users/jisoo0130/proj/toy_proj2/graph.png")

    st.markdown(
        "### [LangSmith](https://smith.langchain.com/o/1a71cb02-30ff-4a53-8d2a-ab1454c113a7/projects/p/da40a522-055e-4831-bcc9-30fdb0ed0073?timeModel=%7B%22duration%22%3A%221d%22%7D)",
        unsafe_allow_html=True,
    )

left, right = st.columns([3, 2], gap="large")

with left:
    # 기존 대화만 렌더
    for msg in st.session_state.chat_messages:
        render_message(msg)

    # 항상 맨 아래에 입력창
    user_input = st.chat_input("아이리스에 대해 질문해보세요")

if user_input:
    # 1. 사용자 메시지는 세션에만 저장
    human = HumanMessage(content=user_input)
    st.session_state.chat_messages.append(human)

    initial_state = {
        "messages": st.session_state.chat_messages,
        "documents": [],
        "filtered_documents": [],
    }

    node_logs = [
        {"node": "grade_question", "status": "완료", "detail": "질문을 분석했습니다."},
    ]

    with st.spinner("답변을 생성하는 중입니다..."):
        result = rag_app.invoke(initial_state)

    ai_msg = get_last_ai_message(result["messages"])

    route = result.get("decision", "")
    if route == "useful":
        route = "rag_pipeline"

    docs = result.get("filtered_documents", [])
    rewritten_query = result.get("rewritten_question", "") or result.get("rewrited_question", "")

    if route:
        if route == "rag_pipeline":
            node_logs.append(
                {"node": "route_question", "status": "완료", "detail": "문서 검색 경로로 라우팅되었습니다."}
            )
            node_logs.append(
                {"node": "search", "status": "완료", "detail": f"검색 문서 수: {len(docs)}"}
            )
            if rewritten_query:
                node_logs.append(
                    {"node": "transform_query", "status": "완료", "detail": rewritten_query}
                )
            node_logs.append(
                {"node": "generate", "status": "완료", "detail": "검색 문서를 바탕으로 답변을 생성했습니다."}
            )
        elif route == "generate_direct":
            node_logs.append(
                {"node": "route_question", "status": "완료", "detail": "직접 답변 경로로 라우팅되었습니다."}
            )
            node_logs.append(
                {"node": "generate_direct", "status": "완료", "detail": "문서 검색 없이 답변을 생성했습니다."}
            )
        else:
            node_logs.append(
                {"node": "route_question", "status": "완료", "detail": f"route={route}"}
            )

    debug = {
        "route": route,
        "rewritten_query": rewritten_query,
        "num_documents": len(docs),
        "message_count": len(result.get("messages", [])),
        "filtered_documents_count": len(result.get("filtered_documents", [])),
    }

    if ai_msg is not None:
        st.session_state.chat_messages.append(ai_msg)
    else:
        st.session_state.chat_messages.append(
            AIMessage(content="AI 응답을 찾지 못했습니다. messages 반환값을 확인해보세요.")
        )

    st.session_state.last_debug = debug
    st.session_state.last_node_logs = node_logs
    st.session_state.last_documents = docs
    st.session_state.thread_turns += 1

    # 새 메시지까지 포함해서 다시 렌더
    st.rerun()

with right:
    tab1, tab2, tab3 = st.tabs(["참조 문서", "실행 정보", "노드 로그"])

    with tab1:
        render_sources(st.session_state.last_documents)

    with tab2:
        render_debug(st.session_state.last_debug)

    with tab3:
        render_node_logs(st.session_state.last_node_logs)