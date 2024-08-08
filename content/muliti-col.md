
>[!multi-column]
>> [!info]- 제목
>> gh
>
>> [!abstract]+ 제목
>> jkj
>
>> [!abstract]+ 제목
>> jkj
>

## 영상모음1
- ### 섹션1 #mcl/list-grid
	- 노트1
	- 노트2
- ### 섹션2
	- 노트1
	- 노트2
- ### 섹션3
	- 노트1
	- 노트2

## 영상모음2
- ### 섹션1 #mcl/list-card
	- 노트1
	- 노트2
- ### 섹션2
	- 노트1
	- 노트2
- ### 섹션3
	- 노트1
	- 노트2

> [!example]- Code Example 
> ```html 
>  <body> 
>  <h1>HTML First Program</h1> 
>  <p>HTML Hello World.</p> 
>   </body> 
 >```
 
 > [!Windows] 
>  ```powershell
>  Stop-Process -Name <name-of-process>
>  ```

> [!example]- code
> ```python
> import streamlit as st
> st.set_page_config(page_title="🦜🔗 뭐든지 질문하세요~ ")
> st.title('🦜🔗 뭐든지 질문하세요~ ')
>  
> from langchain_groq import ChatGroq
> from dotenv import load_dotenv
> import os 
> load_dotenv("D:\\CODE\\LANG\\.env")
>  
> def generate_response(input_text):  #llm이 답변 생성
>     llm = ChatGroq(
>     temperature=0.1,  # 창의성 (0.0 ~ 2.0)
>     max_tokens=8192,  # 최대 토큰수
>     model_name="gemma2-9b-it"
>     )
>     st.info(llm.predict(input_text))
>  
> with st.form('Question'):
>     text = st.text_area('질문 입력:', 'What types of text models does OpenAI provide?') #첫 페이지가 실행될 때 보여줄 질문
>     submitted = st.form_submit_button('보내기')
>     generate_response(text)
>  ```