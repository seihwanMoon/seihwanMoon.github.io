#langchain 
지은이: 서지영
출판사: [길벗·이지톡](https://www.gilbut.co.kr/)

- 책이 코드가 2023년 으로 최근(2024.5)에  LangChain v0.2 로 업그레이드 되며 변경사항이 많이 발생하여 코드를 수정하여 적용하였습니다.
- 오픈소스 환경에서 적용가능 하도록 LLM은 Groq or ollama 를 이용
- 임베딩 도 오픈소스 허깅페이스의 것을 활용 하도록 수정
## 주요정리
- 랭체인 주요모듈
	- 모델IO 
		- 프롬프트: 프로프트생성
		- 언어모델: 언어모델호출
		- 출력파서: 응답을 출력 (원하는 형식으로 출력)
	- 데이터연결: 문서가져오기, 문서변환, 문서임베딩, 벡터저장소, 검색기
	- 체인
	- 메모리
	- 에이전트, 툴

## 목차
1장 LLM 훑어보기
__1.1 LLM 개념
____1.1.1 언어 모델
____1.1.2 거대 언어 모델
__1.2 LLM 특징과 종류
____1.2.1 LLM의 특징
____1.2.2 LLM의 종류
____1.2.3 LLM과 GAI, SLM
__1.3 LLM 생성 과정
__1.4 LLM 생성 후 추가 고려 사항

2장 LLM 활용하기
__2.1 LLM 활용 방법
____2.1.1 파인튜닝
____2.1.2 RAG
____2.1.3 퓨샷 러닝
__2.2 LLM 활용 시 주의 사항
__2.3 LLM의 한계

3장 RAG 훑어보기
__3.1 RAG 개념
__3.2 RAG 구현 과정
____3.2.1 정보 검색
____3.2.2 심화 정보 검색
____3.2.3 텍스트 생성
__3.3 RAG 구현 시 필요한 것
____3.3.1 데이터
____3.3.2 벡터 데이터베이스
____3.3.3 프레임워크(랭체인)

4장 랭체인 익숙해지기
__4.1 랭체인 훑어보기
__4.2 랭체인을 사용하기 위한 환경 구성
____4.2.1 아나콘다 환경 구성
____4.2.2 필요한 라이브러리 설치
____4.2.3 키 발급
__4.3 랭체인 주요 모듈
____4.3.1 모델 I/O
____4.3.2 데이터 연결
____4.3.3 체인
____4.3.4 메모리
____4.3.5 에이전트/툴

5장 랭체인으로 RAG 구현하기
__5.1 간단한 챗봇 만들기
__5.2 RAG 기반의 챗봇 만들기
__5.3 PDF 요약 웹사이트 만들기
__5.4 독립형 질문 챗봇 만들기
__5.5 대화형 챗봇 만들기
__5.6 번역 서비스 만들기
__5.7 메일 작성기 만들기
__5.8 CSV 파일 분석하기

6장 LLM을 이용한 서비스 알아보기
__6.1 콜센터
__6.2 상품 추천
__6.3 보험 언더라이팅
__6.4 코드 생성 및 리뷰
__6.5 문장 생성, M365 코파일럿

부록 코랩 사용법
__A.1 코랩 사용 방법
__A.2 코랩에 파일 업로드하기

## 4장 

### 모델_I_O_(Model_I_O)

- PromptTemplate 을 활용하여 프롬프트를 생성: LLM에게 어떤 문장을 만들지 알는는 역할할
```python fold title:"예제코드"
from langchain_core.prompts import PromptTemplate
template = "{topic}를 홍보하기 위한 좋은 문구를 추천해줘?"
prompt = PromptTemplate(
    input_variables=["topic"],
    template=template,
)
prompt.invoke({"topic": "카메라"})
```

- Groq 의 mixtral 모델 적용
```python fold title:"예제코드"
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os 
load_dotenv("D:\\CODE\\LANG\\.env")
llm1 = ChatGroq(
    temperature=0.1,  # 창의성 (0.0 ~ 2.0)
    max_tokens=8192,  # 최대 토큰수
    model_name="mixtral-8x7b-32768",  #모델명:llama3-8b-8192,llama3-70b-8192,gemma2-9b-it,gemma-7b-it,mixtral-8x7b-32768,whisper-large-v3
)
```

- 모델 llm1 으로 질의 생성
```python fold title:"예제코드"
from langchain_core.messages import HumanMessage, SystemMessage
messages = [
    SystemMessage(content=""),
    HumanMessage(content="진희는 강아지를 키우고 있습니다. 진희가 키우고 있는 동물은?"),
]
llm1.invoke(messages).content
```

- Groq 의 gemma2 모델 적용
```python fold title:"예제코드"
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os 
load_dotenv("D:\\CODE\\LANG\\.env")
llm2 = ChatGroq(
    temperature=0.1,  # 창의성 (0.0 ~ 2.0)
    max_tokens=8192,  # 최대 토큰수
    model_name="gemma2-9b-it",  #모델명:llama3-8b-8192,llama3-70b-8192,gemma2-9b-it,gemma-7b-it,mixtral-8x7b-32768,whisper-large-v3
)
```

- 모델 llm2 으로 질의 생성
```python fold title:"예제코드"
from langchain_core.messages import HumanMessage, SystemMessage
messages = [
    SystemMessage(content=""),
    HumanMessage(content="진희는 강아지를 키우고 있습니다. 진희가 키우고 있는 동물은?"),
]
llm2.invoke(messages).content

```

- ModelLaboratory 이용 모델의 성능 비교 가능
	- llm 2개에 질의
```python fold title:"예제코드"
from langchain.model_laboratory import ModelLaboratory
model_lab = ModelLaboratory.from_llms([llm1, llm2])
model_lab.compare("대한민국의 가을은 몇 월부터 몇 월까지야?")
```

- 출력형식을 콤마로 분리하여 적용하는 예제
```python fold title:"예제코드"
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.prompts import PromptTemplate

output_parser = CommaSeparatedListOutputParser() #파서 초기화
format_instructions = output_parser.get_format_instructions() #출력 형식 지정
prompt = PromptTemplate(
    # 주제에 대한 7가지를 나열하라는 템플릿
    template="7개의 팀을 보여줘 {subject}.\n{format_instructions}",
    input_variables=["subject"],  # 입력 변수로 'subject' 사용
    # 부분 변수로 형식 지침 사용
    partial_variables={"format_instructions": format_instructions},
)

# 출력 결과 생성
chain= prompt| llm2 | output_parser
chain.invoke({"subject": "한국의 야구팀은?"})
```

### 데이터_연결(Data_Connection) 

- 데이터의 ETL 과정: 데이터 추출 -> 변환 -> 적재
- document loaders(데이터읽기) -> document transformers(청크분할) -> embedding model(벡터화) -> vector stores(저장) -> retrievers(검색)

- PyPDFLoader 이용해 pdf 문서 불러오기
```python fold title:"예제코드"
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("../data/The_Adventures_of_Tom_Sawyer.pdf")
document = loader.load()
document[5].page_content[:5000]  # 6페이지의 5,000 글자를 읽어오기
```

Groq 서비스의 gemma2-9b 모델을 설정
```python fold title:"예제코드"
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os 
load_dotenv("D:\\CODE\\LANG\\.env") # 사용자별로 .env에 API키를 넣은 파일 위치
llm = ChatGroq(
    temperature=0.1,  # 창의성 (0.0 ~ 2.0)
    max_tokens=8192,  # 최대 토큰수
    model_name="gemma2-9b-it",  #모델명칭:llama3-8b-8192,llama3-70b-8192,gemma2-9b-it,gemma-7b-it,mixtral-8x7b-32768,whisper-large-v3
)
```

임베딩은 오픈소스인 허깅페이스의 BAAI/bge-m3 모델을 적용할때
```python fold title:"예제코드"
from langchain_huggingface import HuggingFaceEmbeddings
model_name = "BAAI/bge-m3"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)
```

벡터저장소는 FAISS 를 이용
```python fold title:"예제코드"
from langchain.vectorstores import FAISS
db = FAISS.from_documents(document, embeddings)
```

텍스트를 임베딩한 벡터 값을 확인
```python fold title:"예제코드"
text = "진희는 강아지를 키우고 있습니다. 진희가 키우고 있는 동물은?"
text_embedding = embeddings.embed_query(text)
print(text_embedding)
```

chain.invoke 를 활용하여 쿼리 확인
```python fold title:"예제코드"
from langchain.chains import RetrievalQA
retriever = db.as_retriever()
chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
chain.invoke("마을 무덤에 있던 남자를 죽인 사람은 누구니?")
```

### 체인 (chain)
- LLMChain 적용 예제


- SequentialChain 을 이용해 2개의 체인을 연결하고, output_key 로 각각으 결과를 확인
	- 영어-> 한글 번역후 -> 한문장으로 요약 하는 chian



### 메모리(Memory) 
- 대화 과정의 데이터를 저장하는 방법
	- 모든 대화 유지
	- 최근 K 개 유지
	- 대화를 요약해서 유지

- Groq 서비스의 gemma2-9b 모델을 설정
```python fold title:"예제코드"
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os 
load_dotenv("D:\\CODE\\LANG\\.env") # 사용자별로 .env에 API키를 넣은 파일 위치
llm = ChatGroq(
    temperature=0.1,  # 창의성 (0.0 ~ 2.0)
    max_tokens=8192,  # 최대 토큰수
    model_name="gemma2-9b-it",  #모델명칭:llama3-8b-8192,llama3-70b-8192,gemma2-9b-it,gemma-7b-it,mixtral-8x7b-32768,whisper-large-v3
)
```

- ConversationChain 적용예제 -> LangChain v0.2 기준으로 변경 적용
```python fold title:"예제코드"
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a assistnat. Answer the following questions as best you can.Answer in Korean"),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ]
)

history = InMemoryChatMessageHistory()
def get_history():
    return history

chain = prompt | llm | StrOutputParser()

wrapped_chain = RunnableWithMessageHistory(
    chain,
    get_history,
    history_messages_key="chat_history",
)
wrapped_chain.invoke({"input":"진희는 강아지를 한마리 키우고 있습니다."})
wrapped_chain.invoke({"input":"영수는 고양이를 두마리 키우고 있습니다."})
wrapped_chain.invoke({"input":"진희와 영수가 키우는 동물은 총 몇마리?"})
```

### 에이전트/툴
- 에이전트: LLM을 이용해 어쩐 순서로 작업을 할지 결정
- 툴: 특정작업을 수행하기 위한 도구

- wikipedia 라이브러리 이용한 기사검색 tool, numexpr 이라는 연산용 tool
- initialize_agent 는 다양한 에이전트를 정의할수 있다 (랭체인 문서 참조)
	- AgentType.Zero_SHOT_REACT_DESCRIPTION: 툴의 용도와 사용시기를 결정하는 에이전트
		- 툴 마다 설명(description)을 제공 해야 함

## 5장