#langchain 
출처: [Introduction | 🦜️🔗 LangChain](https://python.langchain.com/v0.2/docs/introduction/)

LangChain은 대규모 언어 모델(LLM)로 구동되는 애플리케이션을 개발하기 위한 프레임워크입니다
![langchain](https://python.langchain.com/v0.2/svg/langchain_stack_062024_dark.svg)
- **`langchain-core`**: 기본 추상화 및 LangChain Expression Language(LCEL).
- **`langchain-community`**: 서드파티 통합 모
    - 파트너 패키지(예: **`langchain-openai`**, **`langchain-anthropic`** 등): 
    - 일부 통합은 **`langchain-core`**에만 의존하는 자체 경량 패키지로 더욱 분할되었습니다.
- **`langchain`**: 애플리케이션의 코그너티브 아키텍처를 구성하는 체인, 에이전트 및 검색 전략.
- **[LangGraph](https://langchain-ai.github.io/langgraph)**: 그래프의 에지와 노드로 단계를 모델링하여 LLM으로 강력하고 상태가 저장된 멀티 액터 애플리케이션을 구축합니다. LangChain과 원활하게 통합되지만 LangChain 없이도 사용할 수 있습니다.
- **[LangServe](https://python.langchain.com/v0.2/docs/langserve/)**: LangChain 체인을 REST API로 배포합니다.
- **[LangSmith](https://docs.smith.langchain.com/)**: LLM 애플리케이션을 디버그, 테스트, 평가 및 모니터링할 수 있는 개발자 플랫폼입니다.