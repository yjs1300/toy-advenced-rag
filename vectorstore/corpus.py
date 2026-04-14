"""
    CorpusLoder: hnsw 인덱스와 document corpus를 제공한다. 
    CorpusBuilder: PDF를 읽어들여 임베딩 후 document, faiss index를 저장하는 클래스
"""

import os
from dotenv import load_dotenv
load_dotenv()
from copy import deepcopy
import json
from time import perf_counter
import logging
import re

from openai import OpenAI
import faiss
import pandas as pd
import numpy as np
# from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy

from common.singleton import Singleton


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

FAISS_INDEX = os.getenv('FAISS_INDEX_PATH') 
DOCUMENT = os.getenv('DOCUMENT_PATH') 


class CorpusBuilder:

    def __init__(self):
        self.file_path = os.getenv('IRIS_VDAP_PDF_PATH')
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.openai_client = None

    def _clean_pdf_text(self, text: str) -> str:
        text = text.replace("堺", " ")
        text = text.replace("\xa0", " ")
        text = text.replace("\u200b", "")
        text = text.replace("\ufeff", "")
        text = text.replace("\n9999", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)
        return text.strip()

    def _embed(self, text):
        if self.openai_client is None:
            self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

        response = self.openai_client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )

        return response.data[0].embedding
    
    def _save_docs(self, save_path:str, docs:list[Document]):

        with open(save_path, "a", encoding="utf-8") as f:
            for doc in docs:
                row = {
                    "page_content": doc.page_content,
                    **doc.metadata
                }

                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def save_iris_manual_embedded(self):

        save_path = f"/Users/jisoo0130/proj/toy_proj2/data/embeded_docs_{perf_counter()}.jsonl"

        loader = PyMuPDFLoader(self.file_path)

        num_docs = 0
        processed_docs = []
        
        for doc in loader.lazy_load():
            num_docs += 1

            new_doc = deepcopy(doc)
            
            # text cleaning
            cleaned_content = self._clean_pdf_text(doc.page_content)
            
            # text splitting and embedding 
            splited_content = self.splitter.split_text(cleaned_content)

            for i, chunked_content in enumerate(splited_content):
                new_doc.id = num_docs + i # 저장할 떄 같이 안 됨
                new_doc.page_content = chunked_content
                new_doc.metadata['embedding'] = self._embed(chunked_content)
                processed_docs.append(new_doc)

            # save docs
            if num_docs % 100 == 0:
                self._save_docs(save_path, processed_docs)
                
                logger.info("Processing docs: %s docs, %s chunks", num_docs, len(splited_content))
                processed_docs = []

    def save_faiss_index(self, embedded_file:str):
        
        embedded_docs = []
        
        with open(embedded_file, "r", encoding="utf-8") as file:
            for i, line in enumerate(file):
                doc = json.loads(line)
                embedded_docs.append({"id": i, "embedding": doc['embedding']})
                
        logger.info("Completed loading embedded doc: %s", len(embedded_docs))        

        df = pd.DataFrame.from_dict(embedded_docs) # openai embedding 이미 L2 norm 되어 있음.

        vectors = np.array(df["embedding"].tolist(), dtype="float32")
        
        # Create the FAISS index
        dim = vectors.shape[1]  # 임베딩 벡터의 차원 수
        neighbors = 16  # 노드의 이웃 수

        base_index = faiss.IndexHNSWFlat(dim, neighbors, faiss.METRIC_INNER_PRODUCT) # 코사인 유사도 사용 
        base_index.hnsw.efConstruction = 200 # 그래프 구축시 후보 이웃
        base_index.hnsw.efSearch = 64 # 탐색시 큐 크기 

        index = faiss.IndexIDMap(base_index)

        logger.info("Embedding dimension: %s, neighbors: %s, " \
                    "efConstruction=%s, efSearch=%s", 
                    dim, neighbors, base_index.hnsw.efConstruction, base_index.hnsw.efSearch)
        
        index.add_with_ids(vectors, df["id"].values.astype("int64"))

        # 검색 테스트
        # xq = vectors[:5]
        # v, id = index.search(xq, 5)
        # print(id[:10])

        faiss.write_index(index, FAISS_INDEX)
        
    def load_faiss_index(self) -> faiss.IndexIDMap:
        index = faiss.read_index(FAISS_INDEX)

        logger.info("Loaded FAISS index: %s, total vectors: %s", FAISS_INDEX, index.ntotal)

        return index
    
    def load_docs(self, embedded_file:str) -> list[Document]|InMemoryDocstore:

        documents = []
        with open(embedded_file, "r") as f:
            for line in f:
                data = json.loads(line)
                documents.append(data)

        # 4. docstore 생성
        docstore = InMemoryDocstore({
            str(i): Document(
                page_content=doc["page_content"],  # jsonl의 텍스트 필드명에 맞게 수정
                metadata={k: v for k, v in doc.items() if k not in ('page_content', 'embedding')}
            )
            for i, doc in enumerate(documents)
        })

        return docstore
    
    def search_test(self, query:str, top_k:int=5) -> tuple[np.ndarray, np.ndarray]:
        index = self.load_faiss_index()
        query_embedding = np.array(self._embed(query), dtype="float32").reshape(1, -1)
        distances, indices = index.search(query_embedding, top_k)
        return indices[0], distances[0]

class CorpusLoader(CorpusBuilder, metaclass=Singleton):
    """CorpusBuilder에서 저장한 document corpus와 faiss index를 불러오는 역할을 한다."""
    def __init__(self):
        super().__init__()

    def get_vector_store(self) -> FAISS:
        index = self.load_faiss_index()

        docstore = self.load_docs(DOCUMENT)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small") #
        index_to_docstore_id = {i: str(i) for i in range(len(docstore._dict))}

        vector_store = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
            distance_strategy=DistanceStrategy.DOT_PRODUCT,
            normalize_L2=False
        )

        return vector_store

    
if __name__ == '__main__':
    sv = CorpusBuilder()
    sv.save_faiss_index("/Users/jisoo0130/proj/toy_proj2/data/embeded_docs_48306.860223916.jsonl")
    res1, res2 = sv.search_test("선택된 객체의 외부 그림자에 대해 위치, 색상, 투명도, 크기, 흐림, 객체와의 거리를 설정 할 수 있습니다.", top_k=3)
    print(res1, res2)

    assert False

    try:
        sv = CorpusLoader()
        vector_store = sv.create_vector_store()
    finally:
        if sv.openai_client is not None:
            sv.openai_client.close()