"""RAG pipeline"""
from textwrap import dedent
import sys, os
from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.getenv("PROJECT_PATH"))  # test

from langgraph.graph import StateGraph, START, END 
from langchain_core.messages import HumanMessage, AIMessage
import streamlit as st

from llm_toolkit.retrieval_orchestrator import RetrievalOrchestrator
from llm_toolkit.answer_orchestrator import AnswerOrchestrator
from common.ragstate import RAGState 

@st.cache_resource
def get_rag_app():

    retrieval_orch = RetrievalOrchestrator()
    retriever = retrieval_orch.retriever
    answer_orch = AnswerOrchestrator()

    # 그래프 정의 
    rag_builder = StateGraph(RAGState)
    search_builder = StateGraph(RAGState)

    # 1. 문서 검색
    search_builder.add_edge(START, "ann_search")
    search_builder.add_edge(START, "web_search")

    search_builder.add_node("ann_search", retriever.ann_search)
    search_builder.add_node("web_search", retriever.web_search)

    # 2. 검색된 문서 평가 
    search_builder.add_node("filter_docs", retrieval_orch.filter_documents)
    search_builder.add_edge("ann_search", "filter_docs")
    search_builder.add_edge("web_search", "filter_docs")
    
    search_graph = search_builder.compile()

    # 라우팅 (테스트)
    rag_builder.add_edge(START, "grade_question")
    rag_builder.add_node("grade_question", retrieval_orch.grade_question)
    rag_builder.add_node("search_graph", search_graph)
    rag_builder.add_conditional_edges(
        "grade_question",
        retrieval_orch.route_question,
        {
            "retrieve": "search_graph",
            "generate_direct": "generate_direct",
        },
    )

    rag_builder.add_node("generate_direct", answer_orch.generate_answer)
    rag_builder.add_edge("generate_direct", END)

    # 조건부 엣지 추가: 문서 평가 후 결정
    rag_builder.add_conditional_edges(
        "search_graph",
        answer_orch.decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
    # 쿼리 재생성 
    rag_builder.add_node("transform_query", retrieval_orch.rewrite_query)
    rag_builder.add_edge("transform_query", "search_graph")  # 수정된 쿼리로 검색 재실행

    # 답변 생성 
    rag_builder.add_node("generate", answer_orch.generate_answer)

    # 답변 평가
    rag_builder.add_edge("generate", "grade_generation")
    rag_builder.add_node("grade_generation", answer_orch.grade_generation)

    # 조건부 엣지 추가: 답변 생성 후 평가
    rag_builder.add_conditional_edges(
        "grade_generation",
        answer_orch.route_generation,
        {
            "not supported": "generate",          # 환각이 발생한 경우 -> 답변을 다시 생성 
            "useful": END, 
            "end": END,
        },
    )

    app = rag_builder.compile()

    app.get_graph().draw_png("graph.png")

    return app

if __name__ == "__main__":

    app = get_rag_app()

    q = [
        "아이리스에 대해 설명해줘",
        "아이리스로 차트 그리기",
        "안녕",
    ]
    
    question = q[2]
    
    initial_state = {
        "messages": [HumanMessage(content=question)],
        "documents": [],
        "filtered_documents": []
    }
    res = app.invoke(
        initial_state
    )

    print(res)
  
   