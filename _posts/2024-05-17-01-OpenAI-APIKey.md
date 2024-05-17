---
layout: single
title:  "jupyter notebook 변환하기!"
categories: coding
tag: [python, blog, jekyll]
toc: true
author_profile: false
---

<head>
  <style>
    table.dataframe {
      white-space: normal;
      width: 100%;
      height: 240px;
      display: block;
      overflow: auto;
      font-family: Arial, sans-serif;
      font-size: 0.9rem;
      line-height: 20px;
      text-align: center;
      border: 0px !important;
    }

    table.dataframe th {
      text-align: center;
      font-weight: bold;
      padding: 8px;
    }

    table.dataframe td {
      text-align: center;
      padding: 8px;
    }

    table.dataframe tr:hover {
      background: #b8d1f3; 
    }

    .output_prompt {
      overflow: auto;
      font-size: 0.9rem;
      line-height: 1.45;
      border-radius: 0.3rem;
      -webkit-overflow-scrolling: touch;
      padding: 0.8rem;
      margin-top: 0;
      margin-bottom: 15px;
      font: 1rem Consolas, "Liberation Mono", Menlo, Courier, monospace;
      color: $code-text-color;
      border: solid 1px $border-color;
      border-radius: 0.3rem;
      word-break: normal;
      white-space: pre;
    }

  .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead th {
      text-align: center !important;
      padding: 8px;
  }

  .page__content p {
      margin: 0 0 0px !important;
  }

  .page__content p > strong {
    font-size: 0.8rem !important;
  }

  </style>
</head>


## OpenAI API 키 발급 및 설정



1. OpenAI API 키 발급



- [OpenAI API 키 발급방법](https://teddylee777.github.io/openai/openai-api-key/) 글을 참고해 주세요.



2. `.env` 파일 설정



- 프로젝트 루트 디렉토리에 `.env` 파일을 생성합니다.

- 파일에 API 키를 다음 형식으로 저장합니다:

  `OPENAI_API_KEY` 에 발급받은 API KEY 를 입력합니다.



- `.env` 파일에 발급한 API KEY 를 입력합니다.




```python
# LangChain 업데이트
# !pip install -U langchain langchain-community langchain-experimental langchain-core langchain-openai langsmith
```


```python
# API KEY를 환경변수로 관리하기 위한 설정 파일
# 설치: pip install python-dotenv
from dotenv import load_dotenv

# API KEY 정보로드
load_dotenv()
```

<pre>
True
</pre>
API Key 가 잘 설정되었는지 확인합니다.




```python
import os

print(f"[API KEY]\n{os.environ['OPENAI_API_KEY']}")
```
