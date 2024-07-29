#langchain 
지은이: 서지영
출판사: [길벗·이지톡](https://www.gilbut.co.kr/)

- 책이 코드가 2023년 으로 최근(2024.5)에  LangChain v0.2 로 업그레이드 되며 변경사항이 많이 발생하여 코드를 수정하여 적용하였습니다.
- 오픈소스 환경에서 적용가능 하도록 LLM은 Groq or ollama 를 이용
- 임베딩 도 오픈소스 허깅페이스의 것을 활용 하도록 수정
## 4장 예제
### 데이터_연결(Data_Connection) 예제코드

```python fold title:"예제코드"
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("../data/The_Adventures_of_Tom_Sawyer.pdf")
document = loader.load()
document[5].page_content[:5000]
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

임베딩은 허깅페이스의 BAAI/bge-m3 모델을 적용
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

### 메모리(Memory) 예제

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

LangChain v0.2 기준으로 변경 적용
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

### 모델_I_O_(Model_I_O)

프롬프트템플릿을 활용한 프롬프트 생
```python fold title:"예제코드"
from langchain_core.prompts import PromptTemplate
template = "{topic}를 홍보하기 위한 좋은 문구를 추천해줘?"
prompt = PromptTemplate(
    input_variables=["topic"],
    template=template,
)
prompt.invoke({"topic": "카메라"})
```

Groq 의 mixtral 모델 적용
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

모델 llm1 으로 질의 생성
```python fold title:"예제코드"
from langchain_core.messages import HumanMessage, SystemMessage
messages = [
    SystemMessage(content=""),
    HumanMessage(content="진희는 강아지를 키우고 있습니다. 진희가 키우고 있는 동물은?"),
]
llm1.invoke(messages).content
```

Groq 의 gemma2 모델 적용
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

모델 llm2 으로 질의 생성
```python fold title:"예제코드"
from langchain_core.messages import HumanMessage, SystemMessage
messages = [
    SystemMessage(content=""),
    HumanMessage(content="진희는 강아지를 키우고 있습니다. 진희가 키우고 있는 동물은?"),
]
llm2.invoke(messages).content

```

llm 2개에 질의
```python fold title:"예제코드"
from langchain.model_laboratory import ModelLaboratory
model_lab = ModelLaboratory.from_llms([llm1, llm2])
model_lab.compare("대한민국의 가을은 몇 월부터 몇 월까지야?")
```

출력형식을 콤마로 분리하여 적용하는 예제
```python fold title:"예제코드"
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.prompts import PromptTemplate

output_parser = CommaSeparatedListOutputParser() #파서 초기화
format_instructions = output_parser.get_format_instructions() #출력 형식 지정
prompt = PromptTemplate(
    # 주제에 대한 다섯 가지를 나열하라는 템플릿
    template="7개의 팀을 보여줘 {subject}.\n{format_instructions}",
    input_variables=["subject"],  # 입력 변수로 'subject' 사용
    # 부분 변수로 형식 지침 사용
    partial_variables={"format_instructions": format_instructions},
)

# 출력 결과 생성
chain= prompt| llm2 | output_parser
chain.invoke({"subject": "한국의 야구팀은?"})
```
