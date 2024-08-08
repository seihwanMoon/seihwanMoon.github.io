---
title: <% tp.file.title %>
tags:
---
 <% tp.file.title %>

#### 파인튜닝 데이터셋 준비
- 추가 학습 데이터는 각 줄에 하나의 데이터로 구성하며, 하나의 데이 터는 프롬프트-컴플리션의 쌍(prompt-completion pair)으로 이뤄져 있습니다. 
	- 프롬프트(prompt):  사용자가 모델에게 요청할 입력이나 명령
	- 컴플리션(completion):  그에 대해서 모델이 적절하게 해야 하는 답변입니다
- jsonl 형식으로 구성한다
	- ex) 심리상담 챗봇만드는 데이터셋
> [!example]+ 데이터예
> ["prompt": "너무 마음이 안 좋아요", "completion": "마음이 안 좋을 때는 산책은 어떠세요?" ]
> ["prompt": "남자친구랑 헤어졌어요", "completion": "더 좋은 사람을 만날 거예요!"] 

### 금융뉴스를 감성 분류하는 AI 챗봇 만들기
#### 데이터 입수
-  Finance Phrase Bank라고 불리는 이 데이터셋은. 16명의 금융 전문가가 직접 만든 데이터로 총 4,840개의 금융 문장에 대해서 긍정, 부정, 중립 세 가지 카테고리가 지 정된 데이터입니다. 원본 데이터는 영어로 돼 있지만,  이를 한국어로 번역 한 데이터를 사용
- 해당자료를 깃헙에서 다운로드 
```python fold title:예제
# Finance Phrase Bank 의 금융문
!wget https://raw.githubusercontent.com/ukairia777/finance_sentiment_corpus/main/finance_data.csv
data = pd.read_csv('finance_data.csv')
data = data.drop_duplicates().reset_index(drop=True)
data.head()
```
- labels 열은 neutral, negative, positive 3가지 감정

####  데이터 전처리
- labels 과 kor_sentence 2개의 컬럼만 추출하고
- 열 이름을 변경 ( labels ->completion ,  kor_sentence -> prompt )
- jsonl 파일로 변환
```python fold title:예제
data['prompt'] = data['kor_sentence']
data['completion'] = data['labels']
data = data[['prompt', 'completion']]
data.head()
# jsonl 파일로 변환
data.to_json('finance_data.jsonl', orient='records', force_ascii=False, lines=True)
```

#### openAI 데이터 준비 도구
- openai 의 데이터형식 검사와 수정하는 도구를 적용
- ` !openai tools fine_tunes.prepare_data -f "변환한 jsonl 파일명" `
- 추가 전처리 진행에 맞춰서 Y
	- 중복데이터를 제거 할것인지 (Y)
	- 영어 데이터는 소문자로 하는 것이 좋다고 함 (Y)
	- prompt 와 completion 사이 구분을 위ㅔ 공통으로 데이터 끝에 `-> `  기호를 넣도록 (Y)
	- 모든 completion 열의 시작부분에 공백을 추가 (Y)
	- 훈련과 검증 데이터 셋으로 분라할 것인지(Y)
	- 새로운 JSONL 파일로 저장할 것인지 (Y) 
- 최종실행하면 전처리가 되어 훈련용과 검증용 2개의 파일로 분리되어 생성

####  파인튜닝 실행
- 제시한 명령어를 수행하여 파인 튜닝 실시
- 학습용 파일, 검증용 파일, 분류하고자 하는 카테고리 숫자, 모델 을 설정한다
	- `!openai api fine_tunes.create -t "finance_data_prepared_train.jsonl" -v "finance_data_prepared_valid.jsonl" --compute_classification_metrics --classification_n_classes 3 -m ada`
- 중간에 학습이 끊기는 경우 다시 진행하기위에 학습중에 학습키값을 넣어서 이어서 학습수행한다
	- `!openai api fine_tunes.follow -i ft-파인튜닝모델key값`
- 최종 작업이 완료되면 최종 학습된 파일을 csv 로 저장한다
	- `!openai api fine_tunes.results -i ft-파인튜닝모델key값 > result.csv`
- 최종 csv 파일을 확인
```python fold title:예제
result = pd.read_csv('result.csv')
result
# 검증데이터에 의한 최종평가결과를 확인
result[result['classification/accuracy'].notnull()].tail(1)
# 그래프로 확인
result[result['classification/accuracy'].notnull()]['classification/accuracy'].plot()
# 학습한 데이터를 확인
train = pd.read_json('finance_data_prepared_train.jsonl', lines=True)
train.head()
# 부정인 것만 확인
train[train['completion']==' negative']
# 분류한 갯수 확인
train['completion'].value_counts()
# 검증데이터 확인
test = pd.read_json('finance_data_prepared_valid.jsonl', lines=True)
test.head()
test.loc[0]['prompt']
```

#### 모델 호출하기
- openai 에서 임의의 입력에 대해 모델호출하는 형식은 
	- `openai.Completion.create(model=학습한모델, prompt=프롬프트)`
- max_tokem 을 1로 하여 예측한 텍스트값만   생성하도록 조정
```python fold title:예제
ft_model = 'ada:ft-personal-2023-06-18-11-05-21'
res = openai.Completion.create(model=ft_model, prompt=test['prompt'][0], max_tokens=1, temperature=0)
print(res['choices'][0]['text'])
```

####  쿼리 test
```python fold title:예제
def get_result(input_text):
  input_text = input_text + ' ->'
  ft_model = 'ada:ft-personal-2023-06-18-11-05-21'
  res = openai.Completion.create(model=ft_model, prompt=input_text, max_tokens=1, temperature=0)
  return res['choices'][0]['text'].strip()

test = '바이톤의 순매출이 45% 감소함에 따라서 주가도 지속적으로 하락하고 있다.'
print(get_result(test))
```
#### 그라디오 이용한 UI 구성
```python fold title:예제
iface = gr.Interface(fn=get_result,
                     inputs=gr.inputs.Textbox(lines=5, placeholder='감성 분석할 뉴스를 입력해주세요.'),
                     outputs='text',
                     title="금융 뉴스 감성 분석",
                     description="금융 뉴스를 감성 분석하여 긍정(positive), 부정(negative), 중립(neutral)인지를 알려줍니다.")

iface.launch(share=True)

```