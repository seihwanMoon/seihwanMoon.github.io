

출처: https://python.langchain.com/v0.2/docs/

### Overview[​](https://python.langchain.com/v0.2/docs/tutorials/llm_chain/#overview "Direct link to Overview")

LLM 기반 챗봇을 설계하고 구현하는 방법에 대한 예시를 살펴보겠습니다. 이 챗봇은 대화를 할 수 있고 이전 상호작용을 기억할 수 있습니다.

우리가 구축하는 이 챗봇은 언어 모델만 사용하여 대화를 할 수 있다는 점에 유의하세요. 이와 관련된 몇 가지 다른 개념이 있습니다:

- [Conversational RAG](https://python.langchain.com/v0.2/docs/tutorials/qa_chat_history/): 외부 데이터 소스를 통한 챗봇 환경 지원
- [Agents](https://python.langchain.com/v0.2/docs/tutorials/agents/): 조치를 취할 수 있는 챗봇 구축

### Setup

```
pip install -qU langchain-groq
```

```
import getpassimport osos.environ["GROQ_API_KEY"] = getpass.getpass()from langchain_groq import ChatGroqmodel = ChatGroq(model="llama3-8b-8192")
```

- ChatModel은 LangChain "Runnables"의 인스턴스이며, 이는 상호 작용을 위한 표준 인터페이스를 노출한다는 의미입니다. 모델을 간단히 호출하려면 '.invoke' 메서드에 메시지 목록을 전달하면 됩니다.

```
from langchain_core.messages import HumanMessagemodel.invoke([HumanMessage(content="Hi! I'm Bob")])
```

**API Reference:**[HumanMessage](https://api.python.langchain.com/en/latest/messages/langchain_core.messages.human.HumanMessage.html)

- 모델 자체에는 상태라는 개념이 없습니다. 예를 들어 후속 질문을 하는 경우입니다:

```
model.invoke([HumanMessage(content="What's my name?")])# 이전내용을 기억못함
```

- LangSmith 추적 예시를 살펴보겠습니다. 이전 대화 내용을 문맥으로 가져가지 않아 질문에 답할 수 없음을 알 수 있습니다. 이 문제를 해결하려면 전체 대화 기록을 모델에 전달해야 합니다.

Folded Code

**API Reference:**[AIMessage](https://api.python.langchain.com/en/latest/messages/langchain_core.messages.ai.AIMessage.html)

이것이 챗봇의 대화형 상호작용 기능을 뒷받침하는 기본 아이디어입니다. 그렇다면 이를 가장 잘 구현하려면 어떻게 해야 할까요?

### Message History 메시지 기록[​](https://python.langchain.com/v0.2/docs/tutorials/llm_chain/#message-history "Direct link to Message History")

- 메시지 히스토리 클래스를 사용하여 모델을 래핑하고 상태 저장소로 만들 수 있습니다. 이렇게 하면 모델의 입력과 출력을 추적하고 일부 데이터 저장소에 저장합니다. 그러면 향후 상호 작용에서 해당 메시지를 로드하여 입력의 일부로 체인에 전달합니다. 먼저, 메시지 기록을 저장하기 위해 통합을 사용할 것이므로 `langchain-community`를 설치해 보겠습니다.

```
# ! pip install langchain_community
```

그런 다음 관련 클래스를 임포트하고 모델을 래핑하고 이 메시지 히스토리를 추가하는 체인을 설정할 수 있습니다. 여기서 중요한 부분은 `get_session_history`로 전달하는 함수입니다. 이 함수는 `session_id`를 받아 메시지 히스토리 객체를 반환해야 합니다. 이 `session_id`는 개별 대화를 구분하는 데 사용되며, 새 체인을 호출할 때 구성의 일부로 전달해야 합니다

예제코드

- **API Reference:**[BaseChatMessageHistory](https://api.python.langchain.com/en/latest/chat_history/langchain_core.chat_history.BaseChatMessageHistory.html) | [InMemoryChatMessageHistory](https://api.python.langchain.com/en/latest/chat_history/langchain_core.chat_history.InMemoryChatMessageHistory.html) | [RunnableWithMessageHistory](https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.history.RunnableWithMessageHistory.html)
    
- 이제 매번 런어블에 전달할 `config`를 만들어야 합니다. 이 구성에는 입력에 직접 포함되지는 않지만 여전히 유용한 정보가 포함됩니다. 이 경우 `session_id`를 포함하려고 합니다. 다음과 같이 표시되어야 합니다:
    

```
config = {"configurable": {"session_id": "abc2"}}response = with_message_history.invoke(    [HumanMessage(content="Hi! I'm Bob")],    config=config,)response.content
```

```
response = with_message_history.invoke(    [HumanMessage(content="What's my name?")],    config=config,)response.content
```

잘됐네요! 이제 챗봇이 사용자에 대한 정보를 기억합니다. 다른 `session_id`를 참조하도록 구성을 변경하면 대화가 새로 시작되는 것을 볼 수 있습니다.

```
config = {"configurable": {"session_id": "abc3"}}response = with_message_history.invoke(    [HumanMessage(content="What's my name?")],    config=config,)response.content
```

그러나 언제든지 원래 대화로 돌아갈 수 있습니다(데이터베이스에 유지되므로).

```
config = {"configurable": {"session_id": "abc2"}}response = with_message_history.invoke(    [HumanMessage(content="What's my name?")],    config=config,)response.content
```

이것이 바로 많은 사용자와 대화하는 챗봇을 지원하는 방법입니다! 지금은 모델 주위에 간단한 지속성 레이어를 추가한 것뿐입니다. 프롬프트 템플릿을 추가하여 더 복잡하고 개인화된 챗봇을 만들 수 있습니다.

### Prompt templates[​](https://python.langchain.com/v0.2/docs/tutorials/llm_chain/#prompt-templates "Direct link to Prompt templates")

프롬프트 템플릿은 원시 사용자 정보를 LLM이 작업할 수 있는 형식으로 변환하는 데 도움이 됩니다. 이 경우 원시 사용자 입력은 LLM에 전달되는 메시지일 뿐입니다. 이제 이를 좀 더 복잡하게 만들어 보겠습니다. 먼저 사용자 지정 지침이 포함된 시스템 메시지를 추가해 보겠습니다(하지만 여전히 메시지를 입력으로 받습니다). 다음으로 메시지 외에 더 많은 입력을 추가하겠습니다.

먼저 시스템 메시지를 추가해 보겠습니다. 이를 위해 채팅 프롬프트 템플릿을 생성합니다.  `MessagesPlaceholder`를 활용하여 모든 메시지를 전달하겠습니다.

예제코드

**API Reference:**[ChatPromptTemplate](https://api.python.langchain.com/en/latest/prompts/langchain_core.prompts.chat.ChatPromptTemplate.html) | [MessagesPlaceholder](https://api.python.langchain.com/en/latest/prompts/langchain_core.prompts.chat.MessagesPlaceholder.html)

이렇게 하면 입력 유형이 약간 변경됩니다. 이제 메시지 목록을 전달하는 대신 메시지 목록이 포함된 `messages` key가 있는 dictionary을 전달하게 됩니다.

```
response = chain.invoke({"messages": [HumanMessage(content="hi! I'm bob")]})response.content
```

이제 이전과 동일한 메시지 기록 개체로 래핑할 수 있습니다.

```
with_message_history = RunnableWithMessageHistory(chain, get_session_history)config = {"configurable": {"session_id": "abc5"}}response = with_message_history.invoke(    [HumanMessage(content="Hi! I'm Jim")],    config=config,)response.content
```

```
response = with_message_history.invoke(    [HumanMessage(content="What's my name?")],    config=config,)response.content
```

멋지네요! 이제 프롬프트를 조금 더 복잡하게 만들어 보겠습니다. 이제 프롬프트 템플릿이 다음과 같이 표시된다고 가정해 보겠습니다:

예제코드

프롬프트에 새로운 language입력이 추가되었음을 참고하세요. 이제 체인을 호출하고 원하는 언어로 전달할 수 있습니다.

```
response = chain.invoke(    {"messages": [HumanMessage(content="hi! I'm bob")], "language": "Spanish"})response.content
```

이제 이 더 복잡한 체인을 메시지 히스토리 클래스로 감싸 보겠습니다. 이번에는 입력에 여러 개의 키가 있으므로 채팅 기록을 저장하는 데 사용할 올바른 키를 지정해야 합니다.

예제코드

```
response = with_message_history.invoke(    {"messages": [HumanMessage(content="whats my name?")], "language": "Spanish"},    config=config,)response.content
```

### Managing Conversation History 대화 기록 관리[​](https://python.langchain.com/v0.2/docs/tutorials/llm_chain/#managing-conversation-history "Direct link to Managing Conversation History")

챗봇을 구축할 때 이해해야 할 중요한 개념 중 하나는 대화 기록을 관리하는 방법입니다. 관리하지 않고 방치하면 메시지 목록이 무제한으로 늘어나 LLM의 컨텍스트 창을 넘길 수 있습니다. 따라서 전달되는 메시지의 크기를 제한하는 단계를 추가하는 것이 중요합니다.

**중요한 것은 메시지 템플릿을 사용하기 전에 메시지 기록에서 이전 메시지를 로드한 후에 이 작업을 수행해야 한다는 점입니다.

프롬프트 앞에 `messages` 키를 적절히 수정하는 간단한 단계를 추가한 다음 메시지 기록 클래스에서 새 체인을 래핑하면 됩니다.

LangChain에는 [메시지 목록 관리](https://python.langchain.com/v0.2/docs/how_to/#messages)를 위한 몇 가지 기본 제공 헬퍼가 있습니다. 여기서는 [trim_messages](https://python.langchain.com/v0.2/docs/how_to/trim_messages/) 헬퍼를 사용하여 모델에 보내는 메시지 수를 줄이겠습니다. 트리머를 사용하면 시스템 메시지를 항상 유지할지, 부분 메시지를 허용할지 등의 다른 매개변수와 함께 유지하려는 토큰 수를 지정할 수 있습니다:

예제코드

`          `

**API Reference:**[SystemMessage](https://api.python.langchain.com/en/latest/messages/langchain_core.messages.system.SystemMessage.html) | [trim_messages](https://api.python.langchain.com/en/latest/messages/langchain_core.messages.utils.trim_messages.html)

체인에서 이를 사용하려면 트리머를 실행한 후 messages 입력을 프롬프트에 전달하기만 하면 됩니다. 이제 모델에게 이름을 물어보려고 해도 채팅 기록에서 해당 부분을 잘라냈기 때문에 이름을 알 수 없습니다:

예제코드

**API Reference:**[RunnablePassthrough](https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.passthrough.RunnablePassthrough.html)

하지만 최근 몇 개의 메시지 내에 있는 정보에 대해 물어보면 기억합니다:

```
response = chain.invoke(    {        "messages": messages + [HumanMessage(content="what math problem did i ask")],        "language": "English",    })response.content
```

이제 이 내용을 메시지 기록에 포함시켜 보겠습니다.

예제코드

`      `

예상대로 이름이 언급된 첫 번째 메시지가 잘려나갔습니다. 또한 이제 대화 내역에 두 개의 새로운 메시지(최근 질문과 최근 답변)가 추가되었습니다. 즉, 대화 기록에서 볼 수 있었던 더 많은 정보를 더 이상 볼 수 없게 되었습니다! 이 경우 첫 번째 수학 문제도 기록에서 삭제되었으므로 모델은 더 이상 이 문제를 알 수 없습니다:

```
response = with_message_history.invoke(    {        "messages": [HumanMessage(content="what math problem did i ask?")],        "language": "English",    },    config=config,)response.content
```

### Streaming[​](https://python.langchain.com/v0.2/docs/tutorials/llm_chain/#streaming "Direct link to Streaming")

이제 기능 챗봇이 생겼습니다. 하지만 챗봇 애플리케이션에 있어 정말 중요한 UX 고려 사항 중 하나는 스트리밍입니다. LLM은 때때로 응답하는 데 시간이 걸릴 수 있으므로 사용자 경험을 개선하기 위해 대부분의 애플리케이션은 각 토큰이 생성될 때 이를 스트리밍합니다. 이를 통해 사용자는 진행 상황을 확인할 수 있습니다.

모든 체인은 '.stream' 메서드를 노출하며, 메시지 기록을 사용하는 체인도 다르지 않습니다. 이 메서드를 사용하면 간단히 스트리밍 응답을 반환할 수 있습니다.

Folded Code

### Next Steps[​](https://python.langchain.com/v0.2/docs/tutorials/llm_chain/#next-steps "Direct link to Next Steps")

이제 LangChain에서 챗봇을 만드는 방법의 기본을 이해하셨으니, 좀 더 고급 튜토리얼에 관심을 가질 수 있습니다:

- [Conversational RAG](https://python.langchain.com/v0.2/docs/tutorials/qa_chat_history/): 외부 데이터 소스를 통한 챗봇 환경 지원
- [Agents](https://python.langchain.com/v0.2/docs/tutorials/agents/): 조치를 취할 수 있는 챗봇 구축

구체적인 내용을 더 자세히 살펴보고 싶다면 다음과 같은 몇 가지 사항을 확인해 보세요:

- [Streaming](https://python.langchain.com/v0.2/docs/how_to/streaming/): 스트리밍은 채팅 애플리케이션에 _중요한_ 요소입니다.
- [How to add message history](https://python.langchain.com/v0.2/docs/how_to/message_history/): 를 참조하여 메시지 기록과 관련된 모든 것을 자세히 알아보세요.
- [How to manage large message history](https://python.langchain.com/v0.2/docs/how_to/trim_messages/): 대규모 채팅 기록 관리를 위한 더 많은 기술
 