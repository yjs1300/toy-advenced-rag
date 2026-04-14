"""Provide RetrievalOrchestrator for RAG service"""

import sys, os
from pydantic import BaseModel, Field
from typing import List, Literal
from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.getenv("PROJECT_PATH"))  # test

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.tools import TavilySearchResults
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser

from vectorstore.corpus import CorpusLoader
from common.ragstate import RAGState
from common.common_prompt import COMMON_IDENTITY
from common.llm import LLMFactory
from common.singleton import Singleton

# 검색된 문서의 관련성 평가 결과를 위한 데이터 모델
class BinaryGradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

# 질문을 라우팅하기 위한 데이터 모델 
class RouteQuery(BaseModel):
    route: Literal["retrieve", "generate_direct"] = Field(
        description="문서 검색이 필요하면 retrieve, 아니면 generate_direct"
    )

class Retriever:

    def __init__(self):
        # load corpus 
        self.vector_store = CorpusLoader().get_vector_store()  
        
        self.web_search_k = 2
        self.similarity_search_k = 5

        self.llm = LLMFactory.create()

        # 참고한 문서 많을수록 좋음 -> 웹, 가이드 검색 
        
    def __web_search(self, query: str) -> List[Document]:
        tavily_search = TavilySearchResults(max_results=self.web_search_k)

        results = tavily_search.invoke(query)

        return [
            Document(
                page_content=r["content"],
                metadata={
                    "title": r["title"],
                    "url": r["url"],
                    "source": "web",
                    "score": r["score"]
                }
            )
            for r in results
        ]

    def __ann_search(self, query: str) -> List[Document]:
        """벡터 저장소에서 유사한 문서를 검색하여 반환한다."""
        similar_docs = self.vector_store.similarity_search_with_score(query, k=self.similarity_search_k)
        documents = [
            Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "score": score}
            )
            for doc, score in similar_docs
        ]
        return documents
        
        
    def web_search(self, state:RAGState) -> dict:
        """웹 검색을 수행하여 결과를 상태에 저장한다."""
        query = state["messages"][-1].content  

        retrieved_docs = self.__web_search(query)

        # print(retrieved_docs)
        return {"documents": retrieved_docs}       
              
    def ann_search(self, state:RAGState) -> dict:
        """벡터 저장소에서 유사한 문서를 검색하여 결과를 상태에 저장한다."""
        query = state["messages"][-1].content  

        retrieved_docs = self.__ann_search(query)

        # print(retrieved_docs)
        return {"documents": retrieved_docs}


class RetrivalChainFactory:

    def __init__(self):
        self.llm = LLMFactory.create()

        # LLM Chain
    def create_question_router(self) -> RunnableSequence:
        """아이리스에 관한 질문인지 아닌지 판별하는 체인을 반환한다."""

        structured_llm_grader = self.llm.with_structured_output(RouteQuery)
        
        system_prompt = """너는 사용자 질문이 문서 검색이 필요한지 판단하는 라우터다.
        다음 기준으로 분류해라.

        [retrieve]
        - 사용자가 자사 아이리스 제품에 대한 정보를 요구함
        
        [generate_direct]
        - 아이리스 제품과 관련 되지 않은 질문 (인사, 일반적인 설명, 창의적 작성, 요약)

        반드시 route 값만 결정해라.
        """

        prompt = ChatPromptTemplate([
            ("system", f"{COMMON_IDENTITY}\n\n{system_prompt}"),
            ("human", "{question}")
        ])

        question_router = prompt | structured_llm_grader 

        return question_router
    
    def create_binary_relevance_grader(self) -> RunnableSequence:
        """검색된 문서와 쿼리의 관련 여부를 평가하는 체인을 반환한다."""

        # LLM 모델 초기화 및 구조화된 출력 설정
        structured_llm_grader = self.llm.with_structured_output(BinaryGradeDocuments)

        # 문서 관련성 평가를 위한 시스템 프롬프트 정의
        system_prompt = """You are an expert in evaluating the relevance of search results to user queries.

        [Evaluation criteria]
        1. 키워드 관련성: 문서가 질문의 주요 단어나 유사어를 포함하는지 확인
        2. 의미적 관련성: 문서의 전반적인 주제가 질문의 의도와 일치하는지 평가
        3. 부분 관련성: 질문의 일부를 다루거나 맥락 정보를 제공하는 문서도 고려
        4. 답변 가능성: 직접적인 답이 아니더라도 답변 형성에 도움될 정보 포함 여부 평가

        [Scoring]
        - Rate 'yes' if relevant, 'no' if not
        - Default to 'no' when uncertain

        [Key points]
        - Consider the full context of the query, not just word matching
        - Rate as relevant if useful information is present, even if not a complete answer

        Your evaluation is crucial for improving information retrieval systems. Provide balanced assessments."""

        # 채점 프롬프트 템플릿 생성
        grade_prompt = ChatPromptTemplate.from_messages([
            ("system", f"{COMMON_IDENTITY}\n\n{system_prompt}"),
            ("human", "[Retrieved document]\n{document}\n\n[User question]\n{question}"),
        ])

        binary_relevance_grader = grade_prompt | structured_llm_grader

        return binary_relevance_grader

    def create_question_rewriter(self) -> RunnableSequence:
        """주어진 질문을 검색에 최적화된 형태로 다시 작성하는 체인을 반환"""

        # 시스템 프롬프트 정의
        system_prompt = """
        You are an expert question re-writer. Your task is to convert input questions into optimized versions 
        for vectorstore retrieval. Analyze the input carefully and focus on capturing the underlying semantic 
        intent and meaning. Your goal is to create a question that will lead to more effective and relevant 
        document retrieval.

        [Guidelines]
            1. 질문에서 핵심 개념과 주요 대상을 식별하고 강조합니다.
            2. 약어나 모호한 용어를 풀어서 사용합니다.
            3. 관련 문서에 등장할 수 있는 동의어나 연관된 용어를 포함합니다.
            4. 질문의 원래 의도와 범위를 유지합니다.
            5. 복잡한 질문은 간단하고 집중된 하위 질문으로 나눕니다.

        Remember, the goal is to improve retrieval effectiveness, not to change the fundamental meaning of the question.
        """

        # 질문 다시 쓰기 프롬프트 템플릿 생성
        re_write_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", f"{COMMON_IDENTITY}\n\n{system_prompt}"),
                (
                    "human",
                    "[Initial question]\n{question}\n\n[Improved question]\n",
                ),
            ]
        )

        question_rewriter = re_write_prompt | self.llm | StrOutputParser()

        return question_rewriter


class RetrievalOrchestrator(metaclass=Singleton):

    def __init__(self):
        self.retriever = Retriever()

        chain_factory = RetrivalChainFactory()

        self.router = chain_factory.create_question_router()
        self.grader = chain_factory.create_binary_relevance_grader()
        self.question_rewriter = chain_factory.create_question_rewriter()
        
    def grade_question(self, state:RAGState) -> str:
        question = state["messages"][-1].content  

        decision = self.router.invoke({"question": question}).route 

        return {
            "decision" : decision
        } 
    
    def route_question(self, state: RAGState) -> str:
        return state["decision"]

    def rewrite_query(self, state: RAGState) -> dict:
        """검색을 위해 질문을 개선하는 함수"""
        print("--- 질문 개선 ---")

        if state.get("rewrited_question", None):
            question = state["rewrited_question"]
        else:
            question = state["messages"][-1].content  
        
        # 질문 재작성
        rewritten_question = self.question_rewriter.invoke({"question": question})

        # 생성 횟수 업데이트
        num_generations = state.get("num_generations", 0)
        num_generations += 1
        return {"rewrited_question": rewritten_question, "num_generations": num_generations}      # 재작성한 질문을 저장, 생성횟수 업데이트 
        
        # 관련성 평가 후에 필터링 문서만을 저장 
    
    def filter_documents(self, state: RAGState):
        """검색된 문서의 관련성을 평가하고 필터링하는 함수"""

        print("--- 문서 관련성 평가 ---")
        question = state["messages"][-1].content
        documents = state["documents"]
        
        # 각 문서 평가
        filtered_docs = []
        for d in documents:
            score = self.grader.invoke({"question": question, "document": d})
            grade = score.binary_score
            if grade == "yes":
                print("---문서 관련성: 있음---")
                filtered_docs.append(d)
            else:
                print("---문서 관련성: 없음---")
                
        return {"filtered_documents": filtered_docs}   # 관련성 평가에 합격한 문서들만 저장 (override)
    


if __name__ == "__main__":
    
    question = """
                아이리스로 차트 그리기
            """
 
    retriever = Retriever()

    res = retriever.ann_search.invoke(question)
    print(res)

        