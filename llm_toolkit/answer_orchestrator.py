"""Provide AnswerOrchestrator for RAG service"""
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv
load_dotenv()
import sys, os
import logging
sys.path.append(os.getenv("PROJECT_PATH"))  # test

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.base import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

from common.singleton import Singleton
from common.ragstate import RAGState
from common.common_prompt import COMMON_IDENTITY
from common.llm import LLMFactory

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


class AnswerChainFactory():
        
    def __init__(self):
        self.llm = LLMFactory.create()

    def create_rag_answerer(self) -> RunnableSequence:
        """RAG를 이용하여 답변을 생성하는 체인을 반환한다."""

        system_prompt = """
        Answer the question based solely on the given context. Do not use any external information or knowledge.

        [Instructions]
            1. 질문과 관련된 정보를 제공된 문서에서 신중하게 확인합니다.
            2. 답변에 질문과 직접 관련된 정보만 사용합니다.
            3. 문서에 명시되지 않은 내용에 대해 추측하지 않습니다.
            4. 불필요한 정보를 피하고, 답변을 간결하고 명확하게 작성합니다.
            5. 문서에서 답을 찾을 수 없으면 "주어진 정보만으로는 답할 수 없습니다."라고 답변합니다.
            6. 적절한 경우 문서에서 직접 인용하며, 따옴표를 사용합니다.

        [Documents]
        {filtered_documents}

        [Answer]
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", f"{COMMON_IDENTITY}\n\n{system_prompt}"),
            MessagesPlaceholder("messages"),
        ])

        answer_generator = prompt | self.llm | StrOutputParser()

        return answer_generator

    def create_gerneral_answerer(self) -> RunnableSequence:
        """답변을 생성하는 체인을 반환한다."""

        system_prompt = """
        너는 전문적인 AI 어시스턴트다.

        사용자의 질문에 대해 다음 원칙을 지켜 답변하라:

        1. 정확하고 사실 기반으로 답변할 것
        2. 불확실한 내용은 추측하지 말고 모른다고 말할 것
        3. 필요한 경우 예시를 들어 설명할 것
        4. 불필요한 설명은 줄이고 핵심 위주로 작성할 것
        5. 친절하지만 과하게 감정적이지 않게 답변할 것
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("messages"),
        ])

        answer_generator = prompt | self.llm | StrOutputParser()

        return answer_generator
    
    def create_hallucination_grader(self) -> RunnableSequence:
        """환각 평가를 위한 체인을 반환한다."""

        structured_llm_grader = self.llm.with_structured_output(GradeHallucinations)

        # 환각 평가를 위한 시스템 프롬프트 정의
        system_prompt = """
        You are an expert evaluator assessing whether an LLM-generated answer is grounded in and supported by a given set of facts.

        [Your task]
            - Review the LLM-generated answer.
            - Determine if the answer is fully supported by the given facts.

        [Evaluation criteria]
            - 답변에 주어진 사실이나 명확히 추론할 수 있는 정보 외의 내용이 없어야 합니다.
            - 답변의 모든 핵심 내용이 주어진 사실에서 비롯되어야 합니다.
            - 사실적 정확성에 집중하고, 글쓰기 스타일이나 완전성은 평가하지 않습니다.

        [Scoring]
            - 'yes': The answer is factually grounded and fully supported.
            - 'no': The answer includes information or claims not based on the given facts.

        Your evaluation is crucial in ensuring the reliability and factual accuracy of AI-generated responses. Be thorough and critical in your assessment.
        """

        # 환각 평가 프롬프트 템플릿 생성
        hallucination_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", f"{COMMON_IDENTITY}\n\n{system_prompt}"),
                ("human", "[Set of facts]\n{filtered_documents}\n\n[LLM generation]\n{generation}"),
            ]
        )

        # Hallucination Grader 파이프라인 구성
        hallucination_grader = hallucination_prompt | structured_llm_grader

        return hallucination_grader

class AnswerOrchestrator(metaclass=Singleton):

    def __init__(self):
        chain_factory = AnswerChainFactory()
        self.grader = chain_factory.create_hallucination_grader()
        self.answer_generator_rag = chain_factory.create_rag_answerer()
        self.answer_generator_direct = chain_factory.create_gerneral_answerer()

    def decide_to_generate(self, state:RAGState) -> str:
        """답변 생성 여부를 결정하는 함수"""

        num_generations = state.get("num_generations", 0)
        if num_generations >= 3: # 전체(쿼리, 답변) 재생성 횟수 제한
            logger.info("--- 결정: 생성 횟수 초과, 답변 생성 (-> generate)---")
            return "generate"

        logger.info("--- 평가된 문서 분석 ---")
        filtered_documents = state.get("filtered_documents", None)
        
        if not filtered_documents:
            logger.info("--- 결정: 모든 문서가 질문과 관련이 없음, 질문 개선 필요 (-> transform_query)---")
            return "transform_query"
        else:
            logger.info("--- 결정: 답변 생성 (-> generate)---")
            return "generate"     

    def generate_answer(self, state:RAGState) -> dict:
        """RAG 체인을 이용하여 답변을 생성하는 함수"""

        logger.info("--- 답변 생성 ---")
        
        if state.get("decision", None) == "generate_direct":
            logger.info("-> 일반 생성 체인 사용")
            answer = self.answer_generator_direct.invoke({
                "messages": state["messages"]
            })

            return {"messages": AIMessage(content=answer)}
        else:
            # RAG 답변 체인 실행
            answer = self.answer_generator_rag.invoke({
                "messages": state["messages"],
                "filtered_documents": state["filtered_documents"]
            })

            # 생성 횟수 업데이트
            num_generations = state.get("num_generations", 0)
            num_generations += 1

            return {"generation": answer, "num_generations": num_generations}    
        
    def grade_generation(self, state: RAGState) -> str:
        """생성된 답변의 품질을 평가하는 함수"""

        num_generations = state.get("num_generations", 0)
        if num_generations >= 3:
            print("--- 결정: 생성 횟수 초과, 종료 (-> END)---")
            return {
                "decision": "end",
                "messages": AIMessage(content=state["generation"])
            }
        
        # 1단계: 환각 여부 확인
        print("--- 환각 여부 확인 ---")
        generation, filtered_documents = state["generation"], state["filtered_documents"]
        
        hallucination_grade = self.grader.invoke(
            {"filtered_documents": filtered_documents, "generation": generation}
        )

        if "yes" in hallucination_grade.binary_score.lower():
            print("--- 결정: 환각이 없음 (답변이 컨텍스트에 근거함) ---")
            return {
                "decision": "useful",
                "messages": AIMessage(content=state["generation"])
            }
        else:
            print("--- 결정: 생성된 답변이 문서에 근거하지 않음, 재시도 필요 (-> generate) ---")
            for doc in filtered_documents:
                print(f"관련 문서: {doc.page_content[:200]}...")
                print(f"생성된 답변: {generation[:200]}...")
                print(f"환각 평가 결과: {hallucination_grade.binary_score}\n")
            return {
                "decision": "not supported"
            }
        
    def route_generation(self, state: RAGState) -> str:
        return state["decision"]
    
    
        


if __name__ == "__main__":
    
    question = """
                아이리스로 차트 그리기
            """

    qp = QueryProcessor()
    res = qp.generator_rag_answer()

    raise
    print(f"Original question: {question}\nRewritten question: {res}")
    
    if False:
        # 검색된 문서의 관련성 평가
        binary_relevance_grader = QueryProcessor()

        for doc, score in similar_docs:

            res = binary_relevance_grader.grader.invoke({
                "document": doc.page_content,
                "question": question
            })
            print(f"\n--- 평가 결과 ---\nDocument: {doc.page_content[:100]}...\nRelevance: {res.binary_score}\nScore: {score}")



        