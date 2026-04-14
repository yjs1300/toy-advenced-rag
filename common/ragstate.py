from typing import List, Tuple, Optional, Annotated
from operator import add

from langgraph.graph import MessagesState
from langchain_core.documents import Document

class RAGState(MessagesState):
    """
    메시지 기반 그래프의 상태를 나타내는 클래스
    
    Attributes:
        generation: AI 모델의 답변
        documents: 검색된 문서 리스트
        filtered_documents: 검색된 것 중 관련있는 문서 리스트
        rewrited_question: 재검색을 위해 생성한 쿼리

        num_generations: 생성 횟수 (무한 루프 방지에 활용)
        decision: 분기 
    """
    generation: str
    documents: Annotated[List[Document], add] 
    filtered_documents: List[Document]   
    rewrited_question: str

    num_generations: int 
    decision: Optional[str]