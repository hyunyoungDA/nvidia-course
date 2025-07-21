import streamlit as st
import os
import re
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from googleapiclient.discovery import build # Youtube API 활용 라이브러리 

# --- 1. API 키 설정 (환경 변수 권장) ---
load_dotenv()

openai_api_key = os.environ.get("OPENAI_API_KEY")
youtube_api_key = os.environ.get("YOUTUBE_API_KEY")

# API 키가 설정되어 있는지 확인
if not openai_api_key:
    st.error("오류: OpenAI API 키가 설정되지 않았습니다. '.env' 파일에 'OPENAI_API_KEY'를 설정하거나 환경 변수로 지정해주세요.")
    st.stop()
if not youtube_api_key:
    st.error("오류: YouTube API 키가 설정되지 않았습니다. '.env' 파일에 'YOUTUBE_API_KEY'를 설정하거나 환경 변수로 지정해주세요.")
    st.stop()

# --- 2. LangChain (OpenAI 연동) 설정 ---
llm = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key=openai_api_key,
    temperature=0.7
)

prompt_template = PromptTemplate(
    input_variables=["emotion_text"],
    template="""사용자가 다음과 같이 자신의 감정을 표현했습니다: '{emotion_text}'.
이 감정에 대해 공감하고, 따뜻한 위로 또는 조언을 건네주세요.
**가장 중요한 점은, 이 감정에 어울리는 음악을 찾기 위한 키워드를 2~3개 추천해야 합니다.**
**추천 키워드는 반드시 '키워드: 키워드1, 키워드2, 키워드3' 형태로 답변의 마지막에 포함시켜 주세요.**
예시: '슬픈, 위로, 발라드' 또는 '신나는, 행복한, 경쾌한'

답변:
"""
)

chain = prompt_template | llm

def get_openai_response(emotion_text):
    """OpenAI API를 통해 감정 분석 및 답변을 생성합니다."""
    try:
        response = chain.invoke({"emotion_text": emotion_text})
        # print(f"DEBUG: OpenAI Raw Response: {response.content}") # 디버깅용 출력
        return response.content
    except Exception as e:
        st.error(f"OpenAI API 호출 중 오류가 발생했습니다: {e}")
        return ""

# --- 3. YouTube API 연동 설정 ---
def search_youtube_music(query, max_results=10):
    """YouTube Data API를 이용해 음악을 검색합니다."""
    print(f"DEBUG: Searching YouTube for query: '{query}'") # 디버깅용 출력
    try:
        youtube = build('youtube', 'v3', developerKey=youtube_api_key)
        
        request = youtube.search().list(
            q=query,
            part='snippet',
            type='video',
            videoCategoryId='10', # 음악 카테고리 ID (음악)
            maxResults=max_results
        )
        response = request.execute()

        songs = []
        for item in response.get('items', []):
            # videoId 유효성 검사 추가 (11자 길이 확인)
            if 'videoId' in item['id'] and item['id']['videoId'] and len(item['id']['videoId']) == 11:
                title = item['snippet']['title']
                video_id = item['id']['videoId']
                # API 활용해서 video_id만 추출 
                url = f"https://www.youtube.com/watch?v={video_id}".strip() # 혹시 모를 공백 제거
                
                thumbnail_url = item['snippet']['thumbnails']['high']['url']
                songs.append({
                    'title': title,
                    'video_id': video_id,
                    'url': url,
                    'thumbnail': thumbnail_url
                })
                # print(f"DEBUG: Added song - Title: '{title}', Video ID: '{video_id}', Constructed URL: '{url}'") # 디버깅용 출력
        #     else:
        #         print(f"DEBUG: Skipping item due to invalid or missing videoId: {item.get('id', {}).get('videoId', 'N/A')}") # 디버깅용 출력
                
        # print(f"DEBUG: YouTube Search Results Count: {len(songs)}") # 디버깅용 출력
        return songs
    except Exception as e:
        st.error(f"YouTube API 호출 중 오류가 발생했습니다: {e}. API 키, 할당량, 또는 네트워크 연결을 확인해주세요.")
        print(f"ERROR: YouTube API Error: {e}") # 콘솔에도 오류 출력
        return []
      
# --- 4. Streamlit UI 구현 ---
st.set_page_config(page_title="감정 기반 음악 추천 🎶", page_icon="🎵")
st.title("감정 기반 음악 추천 🎶")
st.markdown("당신의 감정을 입력하면, OpenAI가 답변을 해주고 그에 맞는 음악을 추천해 드릴게요!")

st.markdown("---")

# 사용자 입력
user_emotion = st.text_area("지금 어떤 감정인가요?", height=150, help="예: '오늘은 너무 슬퍼요.', '기분이 정말 좋아요!', '스트레스를 많이 받았어요.'")

if st.button("답변 및 음악 추천 받기"):
    if user_emotion:
        with st.spinner("AI가 당신의 감정을 분석하고 있어요... 🧠"):
            openai_response = get_openai_response(user_emotion)
            
            if not openai_response:
                st.error("OpenAI로부터 유효한 응답을 받지 못했습니다. 잠시 후 다시 시도해주세요.")
                st.stop()

            st.subheader("AI의 답변 💬")
            st.write(openai_response)
        
        st.markdown("---")

        # OpenAI 응답에서 음악 추천 키워드 추출
        # regx 활용해서 3개의 키워드 캡처 
        keywords_match = re.search(r'키워드:\s*(.*)', openai_response)
        
        if keywords_match:
            # 캡처한 키워드를 그룹으로 활용 
            # 여기서는 group(1)로 모든 캡처 결과 반환 
            keywords_str = keywords_match.group(1).strip()
            # print(f"DEBUG: Extracted keywords_str (raw): '{keywords_str}'")
            
            processed_keywords = keywords_str.replace('[', '').replace(']', '')
            search_keywords = re.sub(r'[, ]+', ' ', processed_keywords).strip()
            
            # print(f"DEBUG: Processed search_keywords: '{search_keywords}'")

            if search_keywords:
                st.subheader(f"✨ '{search_keywords}' 키워드로 음악을 추천합니다. 🎧")
                
                # Streamlit 모델 응답 로딩 표시 
                with st.spinner("유튜브에서 음악을 검색하고 있어요... 🔎"):
                    recommended_songs = search_youtube_music(search_keywords)

                    if recommended_songs:
                        for i, song in enumerate(recommended_songs):
                            st.write(f"**{i+1}. {song['title']}**")
                            # --- 수정된 부분: st.link_button에 전달되는 URL 디버깅 출력 ---
                            print(f"DEBUG: URL passed to link_button: '{song['url']}'") 
                            st.link_button(f"🔗 YouTube에서 보기", song['url'])
                            st.markdown("---")
                    else:
                        st.warning("죄송합니다. 해당 키워드로는 YouTube에서 유효한 음악을 찾을 수 없었습니다. 다른 감정을 입력해보시거나, 키워드를 더 구체적으로 시도해보세요.")
            else:
                st.warning("OpenAI 응답에서 유효한 음악 추천 키워드를 추출하지 못했습니다. 프롬프트 설정을 다시 확인해주세요. '키워드:' 패턴이 없거나 내용이 비어있을 수 있습니다.")
        else:
            st.warning("OpenAI 응답에서 '키워드:' 패턴을 찾을 수 없습니다. 프롬프트 형식을 다시 확인해주세요. OpenAI가 지정된 형식으로 응답하지 않았을 수 있습니다.")
    else:
        st.warning("☝️ 감정을 입력해주세요!")

st.markdown("---")
st.info("이 앱은 OpenAI와 YouTube Data API를 사용하여 개발되었습니다.")
