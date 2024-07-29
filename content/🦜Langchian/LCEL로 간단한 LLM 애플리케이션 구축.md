출처: (Build a Simple LLM Application with LCEL) https://python.langchain.com/v0.2/docs/introduction/

LCEL로 간단한 LLM 애플리케이션 구축
- [언어 모델](https://python.langchain.com/v0.2/docs/concepts/#chat-models) 사용
- [프롬프트](https://python.langchain.com/v0.2/docs/concepts/#prompt-templates) 템플릿 및 출력 [구문](https://python.langchain.com/v0.2/docs/concepts/#output-parsers) 분석기 사용
- [LCEL(LangChain Expression Language](https://python.langchain.com/v0.2/docs/concepts/#langchain-expression-language-lcel))을 사용하여 구성 요소를 함께 체인화합니다.
- [LangSmith](https://python.langchain.com/v0.2/docs/concepts/#langsmith)를 사용하여 응용 프로그램 디버깅 및 추적
- [LangServe](https://python.langchain.com/v0.2/docs/concepts/#langserve)를 사용하여 애플리케이션 배포

### 설정
```
pip install langchain
```
자세한 내용은 [설치 가이드](https://python.langchain.com/v0.2/docs/how_to/installation/)를 참조하십시오.

### 랭스미스 LangSmith

LangChain으로 구축하는 많은 응용 프로그램에는 LLM 호출이 여러 번 호출되는 여러 단계가 포함됩니다. 이러한 애플리케이션이 점점 더 복잡해짐에 따라 체인 또는 에이전트 내부에서 정확히 무슨 일이 일어나고 있는지 검사할 수 있는 것이 중요합니다. 이것을 하는 가장 좋은 방법은 [LangSmith](https://smith.langchain.com/)와 함께 하는 것입니다.

위 링크에서 가입한 후에는 추적 기록을 시작하도록 환경 변수를 설정해야 합니다.
```
export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_API_KEY="..."
```

또는 노트북에 있는 경우 다음과 같이 설정할 수 있습니다.
```
import getpass
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()
```

### 언어 모델 사용
- OpenAI
- Anthropic
- Azure
- Google
- Cohere
- FireworksAI
- Groq
- MistralAI
- TogetherAI

### Groq 관련
```
pip install -qU langchain-groq
```

```
import getpass
import os
os.environ["GROQ_API_KEY"] = getpass.getpass()
from langchain_groq import ChatGroq
model = ChatGroq(model="llama3-8b-8192")
```

 ChatModel은 LangChain "Runnables"의 인스턴스이며, 이는 상호 작용을 위한 표준 인터페이스를 노출한다는 의미입니다. 모델을 간단히 호출하려면 .invoke 메서드에 메시지 목록을 전달하면 됩니다.

```
from langchain_core.messages import HumanMessage, SystemMessage
messages = [
    SystemMessage(content="Translate the following from English into Italian"),
    HumanMessage(content="hi!"),
]
model.invoke(messages)
```

**API 참조:**[Human](https://api.python.langchain.com/en/latest/messages/langchain_core.messages.human.HumanMessage.html) Message | [시스템 메시지](https://api.python.langchain.com/en/latest/messages/langchain_core.messages.system.SystemMessage.html)

LangSmith를 활성화하면 이 실행이 LangSmith에 기록되고 [LangSmith 추적](https://smith.langchain.com/public/88baa0b2-7c1a-4d09-ba30-a47985dde2ea/r)을 볼 수 있습니다.

### OutputParsers 출력 구문 분석기

모델의 응답은 `AIMessage`입니다. 여기에는 응답에 대한 다른 메타데이터와 함께 문자열 응답이 포함됩니다.

```
from langchain_core.output_parsers import StrOutputParser
parser = StrOutputParser()
```

**API 참조:**[StrOutputParser](https://api.python.langchain.com/en/latest/output_parsers/langchain_core.output_parsers.string.StrOutputParser.html)

예를 들어, 언어 모델 호출 결과를 저장한 다음 파서에게 전달할 수 있습니다.
```
# 기존방법
result = model.invoke(messages)
parser.invoke(result)
```

더 일반적으로는 이 출력 구문 분석기로 모델을 '체인'할 수 있습니다. 즉, 이 출력 구문 분석기는 이 체인에서 매번 호출됩니다. 이 체인은 언어 모델의 입력 유형(문자열 또는 메시지 목록)을 취하고 출력 구문 분석기의 출력 유형(문자열)을 반환합니다. `|` 연산자를 사용하여 체인을 쉽게 만들 수 있습니다. 연산자는 LangChain에서 두 요소를 결합하는 데 사용됩니다.

```
# chain 방법
chain = model | parser
chain.invoke(messages)
```

LangSmith를 보면 체인에는 두 가지 단계가 있습니다. 먼저 언어 모델이 호출되고 그 결과가 출력 파서로 전달됩니다. 우리는 [LangSmith 흔적](https://smith.langchain.com/public/f1bdf656-2739-42f7-ac7f-0f1dd712322f/r)을 볼 수 있습니다.

### 프롬프트 템플릿

지금 우리는 언어 모델에 직접 메시지 목록을 전달하고 있습니다. 이 메시지 목록은 어디서 온 것입니까? 일반적으로 사용자 입력과 응용 프로그램 로직의 조합으로 구성됩니다. 이 응용 프로그램 로직은 일반적으로 원시 사용자 입력을 받아 언어 모델에 전달할 준비가 된 메시지 목록으로 변환합니다. 일반적인 변환에는 시스템 메시지를 추가하거나 사용자 입력으로 템플릿을 포맷하는 것이 포함됩니다.

PromptTemplates는 원시 사용자 입력을 받고 언어 모델에 전달할 준비가 된 데이터(프롬프트)를 반환합니다.

```
from langchain_core.prompts import ChatPromptTemplate
```

**API 참조:**[채팅 프롬프트 템플릿](https://api.python.langchain.com/en/latest/prompts/langchain_core.prompts.chat.ChatPromptTemplate.html)

여기에 프롬프트 템플릿을 만들어 보겠습니다. 다음 두 가지 사용자 변수가 필요합니다.
- `language`: 텍스트를 번역할 언어
- `text`: 번역할 텍스트

먼저 시스템 메시지로 포맷할 문자열을 만듭니다.
```
system_template = "Translate the following into {language}:"
```

다음으로 프롬프트 템플릿을 만들 수 있습니다. 이것은 다음의 조합이 될 것입니다. `system_template` 또한 번역할 텍스트를 어디에 둘지에 대한 간단한 템플릿도 제공합니다.
```
prompt_template = ChatPromptTemplate.from_messages(    
	[("system", system_template), ("user", "{text}")])
```

이 프롬프트 템플릿에 대한 입력은 dictionary입니다. 우리는 이 프롬프트 템플릿을 가지고 놀 수 있어 그 자체로 무엇을 하는지 확인할 수 있습니다.
```
result = prompt_template.invoke({"language": "italian", "text": "hi"})
```

반환값은 `ChatPromptValue` 두 개의 메시지로 구성되어 있습니다. 메시지에 직접 액세스하려면 다음을 수행합니다.
```
result.to_messages()
```

### LCEL과 함께 구성요소 연결

이제 파이프 `|` 를 사용하여 이것을 위에서 모델 및 출력 파서와 결합할 수 있습니다.
```
chain = prompt_template | model | parser
chain.invoke({"language": "italian", "text": "hi"})
```

이것은 [LCEL(LangChain Expression Language](https://python.langchain.com/v0.2/docs/concepts/#langchain-expression-language-lcel))을 사용하여 LangChain 모듈을 서로 연결하는 간단한 예입니다. 이 접근 방식에는 최적화된 스트리밍 및 추적 지원을 포함하여 여러 가지 이점이 있습니다.

LangSmith 트레이스를 보면 [LangSmith 트레이스](https://smith.langchain.com/public/bc49bec0-6b13-4726-967f-dbd3448b786d/r)에 세 가지 구성요소가 모두 나타나는 것을 볼 수 있습니다.

### LangServe와 함께 서빙하기

 LangServe는 개발자들이 LangChain 체인을 REST API로 배포할 수 있도록 도와줍니다. LangChain을 사용하기 위해 LangServe를 사용할 필요는 없지만 이 가이드에서는 LangServe로 앱을 배포하는 방법을 보여드리겠습니다.

설치 :
```
pip install "langserve[all]"
```
#### 서버
응용 프로그램을 위한 서버를 만들기 위해 우리는 다음을 만들 것입니다. `serve.py` 파일. 이것은 우리의 응용 프로그램을 제공하기 위한 우리의 논리를 포함할 것입니다. 다음 세 가지로 구성되어 있습니다.

1. 우리가 방금 위에 구축한 체인의 정의
2. FastAPI 앱
3. 체인에 서비스를 제공할 경로에 대한 정의로, 다음과 같이 수행됩니다. `langserve.add_routes`

```python fold title:"예제코드"
#!/usr/bin/env python
from typing import List
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langserve import add_routes

# 1. Create prompt template
system_template = "Translate the following into {language}:"
prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', '{text}')
])

# 2. Create model
model = ChatOpenAI()

# 3. Create parser
parser = StrOutputParser()

# 4. Create chain
chain = prompt_template | model | parser

# 4. App definition
app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple API server using LangChain's Runnable interfaces",
)

# 5. Adding chain route

add_routes(
    app,
    chain,
    path="/chain",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)


```
**API 참조:**채팅 [프롬프트](https://api.python.langchain.com/en/latest/prompts/langchain_core.prompts.chat.ChatPromptTemplate.html) 템플릿 | [StrOutputParser](https://api.python.langchain.com/en/latest/output_parsers/langchain_core.output_parsers.string.StrOutputParser.html) | [채팅오픈AI](https://api.python.langchain.com/en/latest/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html)

이 파일을 실행하는 경우:
```
python serve.py
```
우리 체인은 `http://localhost:8000` 에서 제공되는 것을 봐야 합니다.

### Playground

모든 LangServe 서비스에는 스트리밍 출력과 중간 단계로의 가시성을 갖춘 애플리케이션을 구성하고 호출할 수 있는 간단한 [내장](https://github.com/langchain-ai/langserve/blob/main/README.md#playground) UI가 제공됩니다. 로 이동하여 사용해 보십시오! 이전과 동일한 입력을 전달합니다. `{"language": "italian", "text": "hi"}` - 그리고 이전과 동일하게 응답해야 합니다.

#### Client

이제 저희 서비스와 프로그래밍 방식으로 상호 작용할 수 있는 클라이언트를 설정해 보겠습니다. 우리는  [langserve.RemoteRunnable](https://python.langchain.com/v0.2/docs/langserve/#client)서비스로 쉽게 이것을 할 수 있습니다. 이를 사용하면 클라이언트 측에서 실행되는 것처럼 서비스 체인과 상호 작용할 수 있습니다.
```
from langserve import RemoteRunnableremote_chain = RemoteRunnable("http://localhost:8000/chain/")
remote_chain.invoke({"language": "italian", "text": "hi"})
```
LangServe의 다른 많은 기능에 대해 자세히 알아보려면 [여기](https://python.langchain.com/v0.2/docs/langserve/)를 참조하십시오.

### 결론
첫 번째 간단한 LLM 애플리케이션을 만드는 방법에 대해 배웠습니다. 언어 모델을 사용하는 방법, 출력을 구문 분석하는 방법, 프롬프트 템플릿을 만드는 방법, LCEL로 연결하는 방법, LangSmith로 생성한 체인에 우수한 관찰 가능성을 얻는 방법 및 LangServe로 배포하는 방법에 대해 배웠습니다.

LangChain의 핵심 개념에 대한 자세한 내용을 보려면 자세한 [개념 가이드](https://python.langchain.com/v0.2/docs/concepts/)가 있습니다.

이러한 개념에 대해 더 구체적인 질문이 있는 경우 사용 방법 안내서의 다음 섹션을 확인하십시오.
- [LCEL(Lang Chain Expression Language)](https://python.langchain.com/v0.2/docs/how_to/#langchain-expression-language-lcel)
- [프롬프트 템플릿](https://python.langchain.com/v0.2/docs/how_to/#prompt-templates)
- [채팅 모델](https://python.langchain.com/v0.2/docs/how_to/#chat-models)
- [출력 파서](https://python.langchain.com/v0.2/docs/how_to/#output-parsers)
- [랭서브](https://python.langchain.com/v0.2/docs/langserve/)

