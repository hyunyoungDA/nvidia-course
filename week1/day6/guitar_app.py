import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
# from langchain.chains import LLMChain deprecated

# .env 파일에서 환경 변수 로드
load_dotenv(dotenv_path=".env") 
# 디버깅 
openai_api_key = os.getenv("OPENAI_API_KEY")

# Streamlit 페이지 설정
# Streamlit을 활용할 땐 set_page_config를 가장 먼저 위치시켜야 됨
st.set_page_config(page_title="기타 작곡 챗봇", page_icon="🎵")
st.title("🎵 가사와 분위기로 작곡하기")
st.write("가사와 원하는 분위기를 입력하면 AI가 멜로디 아이디어와 코드 진행을 제안합니다!")

# if not openai_api_key:
#     st.error("OpenAI API 키를 찾을 수 없습니다. .env 파일을 확인하세요.")
#     st.stop()
# else:
#     st.success("OpenAI API 키 불러오기 성공 (테스트용)")

# model 설정
llm = ChatOpenAI(model_name="gpt-4.1-mini-2025-04-14", 
                 openai_api_key=openai_api_key, 
                 temperature=0.7)

# 프롬프트 템플릿 정의
# PromptTemplate: 주로 단일 문자열 형태의 프롬프트 생성 
prompt_template = PromptTemplate(
    input_variables=["lyrics", "genre", "mood"], # 입력 변수 지정 
    template="""
    당신은 전문 기타 작곡가입니다. 사용자가 제공한 가사, 장르, 분위기, 그리고 참고 악보(피아노 MIDI에서 추출)를 바탕으로 기타 중심의 멜로디 아이디어와 코드 진행을 제안하세요. 결과는 기타 연주자가 즉시 활용할 수 있도록 구체적이고 실용적이어야 합니다.
    - 가사: {lyrics} 
    - 장르: {genre} (예: 팝은 밝고 단순한 코드와 스트러밍, 재즈는 7th 코드와 스윙 리듬, 블루스는 펜타토닉 멜로디와 셔플 리듬 등 장르 특성 반영)
    - 분위기: {mood} (가사의 감정과 분위기를 멜로디 및 코드에 연결)
    결과는 다음 형식으로 제공하세요:
    
    결과는 다음 형식으로 제공하세요:
    - **기타 멜로디 아이디어**: 기타로 연주 가능한 음계 패턴과 리듬 (예: E4-G4-A4, 4/4박자, 8비트 아르페지오). 기타 음역대(E2~E4) 고려.
    - **기타 코드 진행**: 4~6개의 기타 코드 (예: C-G-Am-F). 기본 3화음 위주, 장르에 따라 7th 코드나 서스펜디드 코드 포함 가능. 스트러밍 패턴(예: D-D-U-U-D-U) 또는 핑거스타일 패턴 제안.
    - **장르 적합성 설명**: 이 멜로디와 코드가 선택한 장르(예: {genre})의 특징(리듬, 화성, 스타일)에 어떻게 부합하는지 구체적으로 설명.
    - **분위기와 가사 연계 설명**: 이 멜로디와 코드가 가사의 감정과 분위기(예: {mood})에 어떻게 어울리는지, 참고 악보와의 연관성 포함해 설명.

    한국어로 자연스럽고 상세하게 작성하세요. 결과는 기타 연주자가 바로 연주할 수 있을 정도로 구체적이어야 합니다.
    """
)

# LangChain 체인 생성
chain = prompt_template | llm

# Streamlit 입력 폼
with st.form(key="compose_form"):
    lyrics = st.text_area("가사를 입력하세요", placeholder="여기에 가사를 입력하세요...")
    mood = st.selectbox("원하는 분위기를 선택하세요", ["감미로운", "신나는", "우울한", "낭만적인", "드라마틱한"])
    genre = st.selectbox("원하는 장르를 선택하세요:", ["발라드","클래식","팝","재즈","힙합"])
    submit_button = st.form_submit_button("작곡 시작")

# 작곡 결과 출력
if submit_button:
    # 작성된 문자열이 없는 경우 에러 처리 
    if lyrics.strip() == "":
        st.error("가사를 입력해주세요!")
    else:
        with st.spinner("작곡 중..."): # Chatmodel이 답변 생성중일 때 표시 
            try:
                # chain은 Runnable 인터페이스; chain을 invoke할 땐 입력 데이터의 타입이 딕셔너리 
                result = chain.invoke({"lyrics":lyrics, "genre": genre, "mood": mood})
                st.subheader("작곡 결과")
                st.markdown(result.content) # result의 content 부분만 출력 
            except Exception as e:
                st.error(f"에러 발생: {str(e)}")
                st.write("OpenAI API 키가 올바른지, 또는 인터넷 연결을 확인해주세요.")

# 사이드바에 사용 방법 안내
st.sidebar.header("사용 방법")
st.sidebar.markdown("""
1. .env 파일에 `OPENAI_API_KEY`를 설정하세요.
2. 가사를 입력하고 원하는 분위기를 선택하세요.
3. "작곡 시작" 버튼을 누르면 AI가 멜로디와 코드를 제안합니다.
4. 결과는 텍스트 기반의 멜로디 아이디어와 코드 진행으로 제공됩니다.
""")