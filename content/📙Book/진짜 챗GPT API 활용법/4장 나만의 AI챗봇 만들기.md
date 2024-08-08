---
title: <% tp.file.title %>
tags:
---
 <% tp.file.title %>

#### DALL-E-2 를 이용한 이미지 생성
```python fold title:예제
import openai
import urllib
from dotenv import load_dotenv
import os 
load_dotenv("D:\\CODE\\LANG\\.env")
from openai import OpenAI
client = OpenAI( api_key = os.environ.get("OPENAI_API_KEY") )

response = client.images.generate(
  model="dall-e-2", # 모델설정: dall-e-3, dall-e-2
  prompt="A futuristic city at day", # 프롬프트
  size="512x512", # 크기 512x512, 1024x1024
  quality="standard", # 해상도: standard, HD
  n=1 # 생성할 이미지수
  )

image_url = response.data[0].url
urllib.request.urlretrieve(image_url, "test.jpg")
```

#### urllib3.PoolManager() 로 HTTP 통신 test
```python fold title:get_test
import urllib3
http = urllib3.PoolManager()
url = "https://jsonplaceholder.typicode.com/posts/1"
response = http.request('GET', url)
print(response.data)
```
> [!info]+ urllib3 test
> 호출한 사이트 https://jsonplaceholder.typicode.com/posts/1
> 에 접속해 보면 출력과 동일한 값을 확인

```python fold title:post_test
import urllib3
http = urllib3.PoolManager()
url = 'https://jsonplaceholder.typicode.com/posts'
data = {"title": "Created Post", "body": "Lorem ipsum", "userId": 15}
response = http.request('POST', url, fields=data)
print(response.data)
```

#### FastAPI 예제
- 비동기 서버 uvicorn 설치 해서 test 
```python fold title:fastapi 
# pip install uvicorn[standard]
# poetry add uvicorn@0.25.0 설체해서 에러 해결
from fastapi import FastAPI
app = FastAPI()
@app.get("/")
async def root():
    return {"message": "This is my house"}

@app.get("/room1")
async def room1():
    return {"message": "Welcome to room1"}

@app.get("/room2")
async def room2():
    return {"message": "Welcome to room2"}
```

### 카카오 챗봇 구현하기
![[20240808_094518.jpg|400]]
![[20240808_094904.jpg|400]]
#### 1)카카오톡 챗봇 구현용 uvicorn 창을 실행
```python fold title:예제
###### 기본 정보 설정 단계 #######
from fastapi import Request, FastAPI
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "kakaoTest"}

@app.post("/chat/")
async def chat(request: Request):
    kakaorequest = await request.json()
    print(kakaorequest)
    return {"message":"kakaoTest"}
```

> [!info]+ 주의사항
> - 실행창에서 ` uvicorn kakaobot:app --reload ` 를 입력해야 함.
> .py를 붙이면 에러발생
> - 브라우저에서 http://127.0.0.1:8000/  접속하면 {"message":"kakaoTest"}  확인
> - 이 창을 유지함

#### 2) Ngrok  구성
- https://ngrok.com/  접속해서 실행파일을 다운로드 받음
- ngrok 의 왼쪽 "Your Authtoken" 선택해서 상단의 토큰번호를 copy 함
- 다운로드한 파일 ngrok.exe 을 실행하면 ngrok 전용창이 생성됨
- 전용창에서  `ngrok authtoken 본인토큰 ` 명령어 실행
- 이어서 ` ngrok http 8000 `  실행하면 외부에서 접속가능한 서버주소가 생성됨
	- 주소중 Forwarding  항목 `https://a8a6-XXXXX.ngrok-free.app/ ` 이 외부 접근 주소

#### 3) 카카오톡 챗봇관리자센터
##### 스킬설정 
- 카카오톡 챗봇관리자센터 https://chatbot.kakao.com/  에 접속
- 왼쪽의 스킬목록 에서 생성 버튼 선택
	- 설명: kakaobot 등 임의로 작성, 기본 스킬로 설정 선택
	- URL: 위의 nrgok 주소에 chat/추가해서 입력 `https://a8a6-XXXXX.ngrok-free.app/chat/ 
	- 오른쪽 하단 "스킬서버로전송" 클릭
- uvicorn 창에 JSON 파일이 전송되어 표시됨
- 챗봇관리자 화면의 응답미리보기엔 "message": "kakaoTest" 가 전송되어 표시됨
- 상단의 "저장" 하여 스킬설정 마침

##### 시나리오 생성 
- 왼쪽 시나리오 선택 +시나리오 폴백블록 선택해 신규시나리오 생성
- 오른쪽 스킬검색/선택에서  위의 스킬에서 만든것을 선택
- 하단의 "스킬데이터 <>" 선택 후 저장
##### 배포
- 왼쪽 "설정" 카카오톡 채널 연결  운영체널선택하기에서 개설한 채널 선택후 저장
- 왼쪽 "배포" 우상단 배포선택

##### 대화창생성 
- 왼쪽 "친구모으기" "채널홍보"  메뉴에서 채널URL 을 복사하여 브라우저에서 열어봄
- 우측 로봇 아이콘 선택하면 대화창이 생성되고 카톡창이 생성됨

#### 4) 챗봇 구성시 고려사항
- 카카오톡서버는 응답시간이 5초 이상되면 답변을 차단함.  따라서 별도 로직 구현 필요
	- 이미지등의 생성시 사용자 요청후 3.5초 이내에 답변이 없을땐 '아직 생각이 끝나지 않았어요' 메세지를 전송후 '생각이 다 끝났나요' 라는버튼을 생성하여 사용자가 버튼 클릭시 다시 전송 하는 식으로 구현해야 함.
	- 카카오톡 챗봇은 사용자 입력에 대한 답변만 가능하고 사용자 입력없이 챗봇이 독단적으로 메세지 보내는 것을 금지해 놨기 때문.

```python fold title:전체코드
###### 기본 정보 설정 단계 #######
from fastapi import Request, FastAPI
import openai
import threading
import time
import queue as q
import os

from dotenv import load_dotenv
import os 
load_dotenv("D:\\CODE\\LANG\\.env")
from openai import OpenAI
client = OpenAI( api_key = os.environ.get("OPENAI_API_KEY") )

# # OpenAI API KEY
# API_KEY = "OpenAI API Key"
# client = openai.OpenAI(api_key = API_KEY)

###### 기능 구현 단계 #######

# 메세지 전송: llm 답변을 카톡서버로 전달하는 json형태 구조
def textResponseFormat(bot_response):
    response = {'version': '2.0', 'template': {
    'outputs': [{"simpleText": {"text": bot_response}}], 'quickReplies': []}}
    return response

# 사진 전송:  생성한 이미지를 카톡서버로 전달하는 json형태 구조
def imageResponseFormat(bot_response,prompt):
    output_text = prompt+"내용에 관한 이미지 입니다"
    response = {'version': '2.0', 'template': {
    'outputs': [{"simpleImage": {"imageUrl": bot_response,"altText":output_text}}], 'quickReplies': []}}
    return response

# 응답 초과시 답변: 응답지연시 메세지보내고ㅡ 답변다시 요청 버튼생성
def timeover():
    response = {"version":"2.0","template":{
      "outputs":[
         {
            "simpleText":{
               "text":"아직 제가 생각이 끝나지 않았어요🙏🙏\n잠시후 아래 말풍선을 눌러주세요👆"
            }
         }
      ],
      "quickReplies":[
         {
            "action":"message",
            "label":"생각 다 끝났나요?🙋",
            "messageText":"생각 다 끝났나요?"
         }]}}
    return response

# ChatGPT에게 질문/답변 받기
def getTextFromGPT(messages):
    messages_prompt = [{"role": "system", "content": 'You are a thoughtful assistant. Respond to all input in 25 words and answer in korea'}]
    messages_prompt += [{"role": "user", "content": messages}]
    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages_prompt)
    message = response.choices[0].message.content
    return message

# DALLE 에게 질문/그림 URL 받기
def getImageURLFromDALLE(messages):   
    response = client.images.generate(
    model="dall-e-2",
    prompt=messages,
    size="512x512",
    quality="standard",
    n=1)
    image_url = response.data[0].url
    return image_url


# 텍스트파일 초기화
def dbReset(filename):
    with open(filename, 'w') as f:
        f.write("")

###### 서버 생성 단계 #######
app = FastAPI()

# get 메서드로 root 접속시 실행할 함수
@app.get("/")
async def root():
    return {"message": "kakaoTest"}

# put 메서드로 /chat/ 접속시 실행할 함수
@app.post("/chat/")
async def chat(request: Request):
    kakaorequest = await request.json()
    return mainChat(kakaorequest)

###### 메인 함수 단계 #######

# 메인 함수
def mainChat(kakaorequest):

    run_flag = False
    start_time = time.time()

    # 응답 결과를 저장하기 위한 텍스트 파일 생성
    cwd = os.getcwd()
    filename = cwd + '/botlog.txt'
    if not os.path.exists(filename):
        with open(filename, "w") as f:
            f.write("")
    else:
        print("File Exists")    

    # 답변 생성 함수 실행: 큐 자료구조를 사용하여 저장용 put(),꺼내기 get() 메서드로 응답을 받는다.
    # threading 을 이용헤 gpt 와 달리의 응답요청을 동시 처리
    response_queue = q.Queue()
    request_respond = threading.Thread(target=responseOpenAI,
                                        args=(kakaorequest, response_queue,filename))
    request_respond.start()

    # 답변 생성 시간 체크
    while (time.time() - start_time < 3.5):
        if not response_queue.empty():
            # 3.5초 안에 답변이 완성되면 바로 값 리턴
            response = response_queue.get()
            run_flag= True
            break
        # 안정적인 구동을 위한 딜레이 타임 설정
        time.sleep(0.01)

    # 3.5초 내 답변이 생성되지 않을 경우
    if run_flag== False:     
        response = timeover()

    return response

# 답변/사진 요청 및 응답 확인 함수
def responseOpenAI(request,response_queue,filename):
    # 사용자가 버튼을 클릭하여 답변 완성 여부를 다시 봤을 시
    if '생각 다 끝났나요?' in request["userRequest"]["utterance"]:
        # 텍스트 파일 열기
        with open(filename) as f:
            last_update = f.read()
        # 텍스트 파일 내 저장된 정보가 있을 경우
        if len(last_update.split())>1:
            kind = last_update.split()[0]  
            if kind == "img":
                bot_res, prompt = last_update.split()[1],last_update.split()[2]
                response_queue.put(imageResponseFormat(bot_res,prompt))
            else:
                bot_res = last_update[4:]
                response_queue.put(textResponseFormat(bot_res))
            dbReset(filename)

    # 이미지 생성을 요청한 경우
    elif '/img' in request["userRequest"]["utterance"]:
        dbReset(filename)
        prompt = request["userRequest"]["utterance"].replace("/img", "")
        bot_res = getImageURLFromDALLE(prompt)
        response_queue.put(imageResponseFormat(bot_res,prompt))
        save_log = "img"+ " " + str(bot_res) + " " + str(prompt)
        with open(filename, 'w') as f:
            f.write(save_log)

    # ChatGPT 답변을 요청한 경우
    elif '/ask' in request["userRequest"]["utterance"]:
        dbReset(filename)
        prompt = request["userRequest"]["utterance"].replace("/ask", "")
        bot_res = getTextFromGPT(prompt)
        response_queue.put(textResponseFormat(bot_res))

        save_log = "ask"+ " " + str(bot_res)
        with open(filename, 'w') as f:
            f.write(save_log)
            
    #아무 답변 요청이 없는 채팅일 경우
    else:
        # 기본 response 값
        base_response = {'version': '2.0', 'template': {'outputs': [], 'quickReplies': []}}
        response_queue.put(base_response)
```
