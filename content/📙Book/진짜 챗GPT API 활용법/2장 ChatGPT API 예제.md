

#### 환경설정
```python fold title:환경설정
# API 인증서 파일 설정
from dotenv import load_dotenv
import os 
load_dotenv("D:\\CODE\\LANG\\.env")
# OPEN API키 설정
from openai import OpenAI
client = OpenAI( api_key = os.environ.get("OPENAI_API_KEY") )
```

#### 질문하기 `.chat.completions.create`
```python fold title:질문하기
response = client.chat.completions.create(
  model="gpt-4o-mini",
  messages=[{"role": "user", "content": "Tell me how to make a pizza"}])
print(response) # 응답 전체 출력
# 응답에서 답변만 출력
print(response.choices[0].message.content)
# 소모토큰 확인
print(response.usage)
```

> [!note]+ 응답분석
> - choices: 완료 개체 목록. 질문 시응답 개수(n)를 1로 설정하면 한개, 2로 설정하면 2개의 완료 개 체가 리스트 형태로 저장. 
> - index: 완료 개체의 인덱스.
> - message: 모델에서 생성된 메시지 내용.
> - content:  답변 내용 
> - role:  질문 시 지정한 역할
> - created: 요청한 시점의 타임스탬프
> - 0bject:  반환된 객체의 유형
> - usage: 질문할 때 사용한 토큰 수, 응답할 때 사용한 토큰 수, 총 사용한 토큰 수를 각각 제공

> [!note]+ 파라미터 옵션
> 모델(model): 사용할 언어 모델을 지정
> - 메시지(message): 사용자가 입력할 프롬프트가 포함된 리스트.기본적인 질문 방법은  role: "user", "content: "프롬프트 입력" 
> - 온도 조절(temperature): 텍스트의 랜덤성(randomness). 설정값 범위는 0.0~2.0이며(기본값은 1)
>  	- 온도를 높게 설정하 면( 최댓값은 2.0)  창의적으로 응답. 
>  	- 온도를 낮게 설정하면 좀 더 전형적이 응답.
> - 핵 샘플링(top 0): 다음 단어 또는 토큰이 샘플링되는 범위를 제어합니다. 응답을 생성할 때 모델은 다음 토큰의 어휘에 대한 확률 분포를 계산합니다. 예를 들어, top _p를 0.5로 설정하면 모델이 샘플링할 때.누 적 확률이 0.5보다 큰 상위 토큰 중에서만 다음 토큰을 샘플링합니다. 1op_p를 1.0으로 설정하면 모든 토 큰(전체 분포)에서 샘플링하고, top _p를 0.0으로 설정하면 항상 가장 확률이 높은 단일 토큰을 선택합니 다. 설정값의 범위는 0.0~1.0이|며, 값을 설정하지 않으면 기본값인 1로 설정됩니다.
> - 존재 페널티(presence penalty): 단어가 이미 생성된 텍스트에 나타난 경우 해당 단어가 등장할 가능성 을 줄입니다.  과거 예측에서 단어가 나타나는 빈도수에 따라 달라지 지는 않습니다. 이 파라미터 값을 크게 설정할수록 모델이 새로운 주제에 대해 이야기할 가능성이 높아진다. 설정값의 범위는 0.0~2.00|며, 값을 설정하지 않으면 기본값인0
> - 빈도수 페널티(trequency_ penall): 모델이 동일한 단어를 반복적으로 생성하지 않도록 설정하는 값.좀 더 다양하고 중복되지 않은 텍스트 를 생성하게 유도할 수 있습니다. 따라서 모델이 특정 단어를 반복하는 경향을 보인다면 빈도수 페널티 값 을 높게 설정합니다. 설정값의 범위는 0.0~2.0이며, 값을 설정하지 않으면 기본값인 0으로 설정됩니다. 
> - 응답 개수(n): 입력 메시지에 대해 생성할 답변의 수를 설정합니다. (기본값 1)
> - 최대 토큰(max_tokens): 최대 토큰 수를 제한합니다. 값을 설정하지 않으면 모델의 최대 토큰 수 에 맞춰 설정됩니다.
> - 중지 문자(stop): 토큰 생성을 중지하는 문자입니다. stop= ['|n', 'end of text']처럼 문자열 목록으로 값을 설정합니다. None으로 설정하면 따로 중지 문자 설정들 하지 않고 답변을 끝까지 생성합니다.

#### 역할 부여하기
```python fold title:"예제코드"
response = client.chat.completions.create(
 model="gpt-4o-mini",
 messages=[
         {"role": "system", "content": "너는 친절하게 답변해주는 비서야"},
        {"role": "user", "content": "2020년 월드시리즈에서는 누가 우승했어?"}
 ]
)
print(response.choices[0].message.content)
```
#### 이전 대화를 포함한 답변
```python fold title:예제
response = client.chat.completions.create(
 model="gpt-4o-mini",
 messages=[
 {"role": "user", "content": "2002년 월드컵에서 가장 화제가 되었던 나라는 어디야?"},
 {"role": "assistant", "content": "바로 예상을 뚫고 4강 진출 신화를 일으킨 한국입니다."},
 {"role": "user", "content": "그 나라가 화제가 되었던 이유를 자세하게 설명해줘"}
 ]
)
print(response.choices[0].message.content)
```
