---
title: <% tp.file.title %>
tags:
---
 <% tp.file.title %>


### chatgpt 이용한 데이터 전처리
#### 방법1: prompt 이용한 전처리
- 다양한 데이터를 복사하여 chatgpt 에 붙여넣고 프롬프트를 수행
> [!note]+ prompt
> 1. 위 내용 전부를 읽기 쉽도록 평문으로 재작성 해주세요. 이 요구 조건은 매우 중요하며 꼭 지켜져 야 합니다. 임의로 정리하거나 소제목을 붙이지 마세요. 끊어서 작성하지 말고 반드시 하나의 평문 이어야 합니다. 
> 2. 웹 사이트 주소나 URL 주소가 있다면 해당 주소를 단축하거나 애매모호하게 줄이지 마세요. URL 주소는 반드시 원문 그대로 작성해야 합니다.

#### 방법2: APi 를 이용한 전처리
```python fold title:예제
!pip install openai
import openai openai.api_key = "OepnAI API Key"

content = '''청년예술지원 사업 개요 정책 유형 문화/예술 정책 소개 활동 경력이 부족한 청년예술인 대상 중간생략 ....'''

prompt = f'''아래 내용은 정리되지 않은 표 형태의 데이터입니다. 당신은 아래 내용을 평문으로 작성해야만 합니다.

<내용> 
{content}
<내용끝>

1. 위 내용 전부를 읽기 쉽도록 평문으로 재작성 해주세요. 이 요구 조건은 매우 중요하며 지켜져야 합니다. 임의로 정리하거나 소제목을 붙이지 마세요. 끊어서 작성하지 말고 반드 하나의 평문이어야 합니다. 
2. 웹 사이트 주소나 URL 주소가 있다면 해당 주소를 단축하거나 애매모호하제 줄이지 마세드 URL 주소는 반드시 원문 그대로 작성해야 합니다.'''

def get_completion(prompt, model="gpt-4o-mini"): 
	messages=[("role": "user", "content": prompt]] 
	response = openai.ChatCompletion.create( 
		model=model, 
		messages=messages,
	return response.choices[0].message["content"]
	
result = get_completion(prompt) 
print(result)
```

### 벡터 유사도

#### 텍스트 임베딩: 텍스트 -> 벡터화
- 단어/문장/문서 임베딩
- 코사인 유사도
	- -1~1 범위로 유사할 수록 1에 가까움
- openAPI 의 임베딩 중 가장 저렴한 text-embedding-3-small 모델을 적용
```python fold title:예제
from openai import OpenAI
client = OpenAI()

response = client.embeddings.create(
    input="Your text string goes here",
    model="text-embedding-3-small"
)
print(response.data[0].embedding)
```

```python fold title:예제
import os
import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
import ast
import openai
import streamlit as st
from streamlit_chat import message

client = openai.OpenAI(api_key = "사용자의 OpenAI API Key 값")

def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model='text-embedding-3-small'
    )
    return response.data[0].embedding

# folder_path와 file_name을 결합하여 file_path = './data/embedding.csv'
folder_path = './data'
file_name = 'embedding.csv'
file_path = os.path.join(folder_path, file_name)

# if: embedding.csv가 이미 존재한다면 데이터프레임 df로 로드한다.
if os.path.isfile(file_path):
    print(f"{file_name} 파일이 존재합니다.")
    df = pd.read_csv(file_path)
    df['embedding'] = df['embedding'].apply(ast.literal_eval)

# 그렇지 않다면 text열과 embedding열이 존재하는 df를 신규 생성해야한다.
else:
    # 57개의 서울 청년 정책 txt 파일명을 txt_files에 저장한다.
    txt_files = [file for file in os.listdir(folder_path) if file.endswith('.txt')]

    data = []
    # txt_files로부터 57개의 청년 정책 데이터를 로드하여 df를 신규 생성한다.
    for file in txt_files:
        file_path = os.path.join(folder_path, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            data.append(text)

    df = pd.DataFrame(data, columns=['text'])

    # 데이터프레임의 text 열로부터 embedding열을 생성한다.
    df['embedding'] = df.apply(lambda row: get_embedding(
        row.text,
    ), axis=1)

    # 추후 사용을 위해 df를 'embedding.csv' 파일로 저장한다.
    # 이렇게 저장되면 추후 실행에서는 df를 새로 만드는 과정을 생략한다.
    df.to_csv(file_path, index=False, encoding='utf-8-sig')

# 주어진 질의로부터 유사한 문서 개를 반환하는 검색 시스템.
# 함수 return_answer_candidate내부에서 유사도 계산을 위해 cos_sim을 호출.
def cos_sim(A, B):
    return dot(A, B)/(norm(A)*norm(B))

def return_answer_candidate(df, query):
    query_embedding = get_embedding(query)
    df["similarity"] = df.embedding.apply(lambda x: cos_sim(np.array(x), np.array(query_embedding)))
    top_three_doc = df.sort_values("similarity", ascending=False).head(3)
    return top_three_doc

# 챗봇의 답변을 만들기 위해 사용될 프롬프트를 만드는 함수.
def create_prompt(df, query):
    result = return_answer_candidate(df, query)
    system_role = f"""You are an artificial intelligence language model named "정채기" that specializes in summarizing \
    and answering documents about Seoul's youth policy, developed by developers 유원준 and 안상준.
    You need to take a given document and return a very detailed summary of the document in the query language.
    Here are the document: 
            doc 1 :{str(result.iloc[0]['text'])}
            doc 2 :{str(result.iloc[1]['text'])}
            doc 3 :{str(result.iloc[2]['text'])}
    You must return in Korean. Return a accurate answer based on the document.
    """
    user_content = f"""User question: "{str(query)}". """

    messages = [
        {"role": "system", "content": system_role},
        {"role": "user", "content": user_content}
    ] 
    return messages

# 위의 create_prompt 함수가 생성한 프롬프트로부터 챗봇의 답변을 만드는 함수.
def generate_response(messages):
    result = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.4,
        max_tokens=500)
    return result.choices[0].message.content

st.image('images/ask_me_chatbot.png')

# 화면에 보여주기 위해 챗봇의 답변을 저장할 공간 할당
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

# 화면에 보여주기 위해 사용자의 답변을 저장할 공간 할당
if 'past' not in st.session_state:
    st.session_state['past'] = []

# 사용자의 입력이 들어오면 user_input에 저장하고 Send 버튼을 클릭하면
# submitted의 값이 True로 변환.
with st.form('form', clear_on_submit=True):
    user_input = st.text_input('정책을 물어보세요!', '', key='input')
    submitted = st.form_submit_button('Send')

# submitted의 값이 True면 챗봇이 답변을 하기 시작
if submitted and user_input:
    # 프롬프트 생성
    prompt = create_prompt(df, user_input)
    # 생성한 프롬프트를 기반으로 챗봇 답변을 생성
    chatbot_response = generate_response(prompt)
    # 화면에 보여주기 위해 사용자의 질문과 챗봇의 답변을 각각 저장
    st.session_state['past'].append(user_input)
    st.session_state['generated'].append(chatbot_response)

# 사용자의 질문과 챗봇의 답변을 순차적으로 화면에 출력
if st.session_state['generated']:
    for i in reversed(range(len(st.session_state['generated']))):
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
        message(st.session_state['generated'][i], key=str(i))
        
```