


### 데이터 읽기
#### 랭체인 DirectoryLoader() : 폴더의 다중 문서를 읽어오기
- DirectoryLoader()는 파일을 순서대로 로드하지 않는다
```python fold title:예제
text_loader_kwargs={'autodetect_encoding': True} # 인코딩에러를 해결
loader = DirectoryLoader("./data/", glob="./*.txt", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
documents = loader.load()
print('문서의 개수 :', len(documents))
print('1번 문서 :', documents[1])
print('-' * 20)
print('21번 문서 :', documents[21])
print('-' * 20)
```

#### 문서분할 RecursiveCharacterTextSplitte()
```python fold title:예제
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

print('분할된 텍스트의 개수 :', len(texts))
texts[0]
```

#### 1개의 문서가 여러문서로 분할된 것 확인
```python fold title:예제
source_lst = []
for i in range(0, len(texts)):
  source_lst.append(texts[i].metadata['source'])

element_counts = Counter(source_lst)
filtered_counts = {key: value for key, value in element_counts.items() if value >= 2}
print('2개 이상으로 분할된 문서 :', filtered_counts)
print('분할된 텍스트의 개수 :', len(documents) + len(filtered_counts))
```

### Chroma DB를 이용한 검색기 사용
- Chroma DB을 이용해 임베딩,벡터화, 코사인 유사도 검색
```python fold title:예제
from langchain_openai import OpenAIEmbeddings
embedding = OpenAIEmbeddings()
vectordb = Chroma.from_documents(
    documents=texts, # 벡터로 전환할 문서
    embedding=embedding) # 임베딩 모델
```

#### retriever (검색기) 만들기
- as_retriever() 로 검색기 선언 후, get_relevant_documents() 유사문서 찾기
```python fold title:예제
retriever = vectordb.as_retriever()
docs = retriever.get_relevant_documents("신혼 부부를 위한 정책이 있어?")
print('유사 문서 개수 :', len(docs))
print('--' * 20)
print('첫번째 유사 문서 :', docs[0])
print('--' * 20)
print('각 유사 문서의 문서 출처 :')
for doc in docs:
    print(doc.metadata["source"])
```
- 유사문서를 k 개만 가져올땐  search_kwargs() 이요
```python fold title:예제
retriever = vectordb.as_retriever(search_kwargs={"k": 2})  # 2개의 문서만 찾도록 설정
docs = retriever.get_relevant_documents("신혼 부부를 위한 정책이 있어?")
for doc in docs:
    print(doc.metadata["source"])
```

### GPT 에서 답변얻기
#### 체인 만들기
```python fold title:예제
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-4o-mini", temperature=0),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True)
```
- chain_type="stuff" 으로 설정하면 내부적으론 아래의 프롬프트를 사용하여 구현됨
> [!note]+ prompt
> Use the following pieces of context to answer the users question.
> If you don't know the answer, just say that you don't know, don't try to make up an answer.
> ----------------
> {텍스트}
> 
> {질문}
- {텍스트} 에는 유사높은 텍스트의 본문이 삽입됨.
-  {질문} 에는 입력한 텍스트가 사용됨
```python fold title:질행
input_text = "대출과 관련된 정책이 궁금합니다"
chatbot_response = qa_chain(input_text)
print(chatbot_response)
```

> [!info]+ query과
> 'query': 사용자질문
> 'result':
> 'source_documents': 분할된 텍스트문서
> - Document( metadata={'source': 'data\\39.txt'}, page_content='ㅇㅇㅇㅇㅇ' )
> 

- 챗봇답변인 result 와 출처를 확인하는 netadata 의 source 의 값만  출력해봄
```python fold title:예제
def get_chatbot_response(chatbot_response):
    print(chatbot_response['result'].strip())
    print('\n문서 출처:')
    for source in chatbot_response["source_documents"]:
        print(source.metadata['source'])
        
input_text = "신혼 부부의 신혼집 마련을 위한 정책이 있을까?"
chatbot_response = qa_chain(input_text)
get_chatbot_response(chatbot_response)
print(chatbot_response)
```
### Gradio 챗봇 UI 사용하기
```python fold title:예제
import gradio as gr

# 인터페이스를 생성.
with gr.Blocks() as demo:
    chatbot = gr.Chatbot(label="청년정책챗봇") # 청년정책챗봇 레이블을 좌측 상단에 구성
    msg = gr.Textbox(label="질문해주세요!")  # 하단의 채팅창의 레이블
    clear = gr.Button("대화 초기화")  # 대화 초기화 버튼

    # 챗봇의 답변을 처리하는 함수
    def respond(message, chat_history):
      result = qa_chain(message)
      bot_message = result['result']
      bot_message += ' # sources :'

      # 답변의 출처를 표기
      for i, doc in enumerate(result['source_documents']):
          bot_message += '[' + str(i+1) + '] ' + doc.metadata['source'] + ' '

      # 채팅 기록에 사용자의 메시지와 봇의 응답을 추가.
      chat_history.append((message, bot_message))
      return "", chat_history

    # 사용자의 입력을 제출(submit)하면 respond 함수가 호출.
    msg.submit(respond, [msg, chatbot], [msg, chatbot])

    # '초기화' 버튼을 클릭하면 채팅 기록을 초기화.
    clear.click(lambda: None, None, chatbot, queue=False)

# 인터페이스 실행.
demo.launch(debug=True)

```
()