import pandas as pd
import os
import folium
from streamlit_folium import folium_static # 지도 표시 활용 
from streamlit_geolocation import streamlit_geolocation 
from langchain_community.document_loaders.csv_loader import CSVLoader # csv 파일 로딩 
from langchain_core.prompts import PromptTemplate 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OpenAI API 키가 .env 파일에 정의되지 않았습니다! .env 파일을 확인하세요.")
    st.stop()

# Streamlit 페이지 설정
st.set_page_config(page_title = "경기도 으뜸 맛집 추천 챗봇", page_icon = "🍽️", layout = "wide")
st.title("🍴 경기도 맛집 추천 챗봇")
st.write("지역이나 음식 종류를 입력하면 경기도 내 맛집을 추천해드립니다! (예: '고양시 한식', '광주시 오리구이')")

# csv 파일 경로 설정 
csv_file = "C:/nvidia_course/nvidia-course/week1/day6/restaurant_in_gyeonggi.csv"

try:
    # 만약 파일의 인코딩인 cp949인 경우 
    # CSVLoader는 csv_args = {"delimiter", "quotechar","fieldnames" 등}를 활용할 수 있음 
    loader = CSVLoader(file_path=csv_file, encoding="cp949")
    documents = loader.load() 

    if not documents:
        st.error("CSV 파일이 비어 있거나 데이터를 로드할 수 없습니다.")
        st.stop()

except FileNotFoundError:
    st.error(f"{csv_file} 파일을 찾을 수 없습니다. 'data' 폴더에 파일이 있는지 확인하세요.")
    st.stop()

except UnicodeDecodeError:
    st.warning("CSV 파일을 'utf-8-sig'로 읽지 못했습니다. 'utf-8'로 재시도합니다.")
    try:
        loader = CSVLoader(file_path=csv_file, encoding="utf-8")
        documents = loader.load()

        if not documents:
            st.error("CSV 파일이 비어 있거나 데이터를 로드할 수 없습니다.")
            st.stop()

    except Exception as e:
        st.error(f"'utf-8' 인코딩으로도 파일을 로드할 수 없습니다: {str(e)}")
        st.stop()

except Exception as e:
    st.error(f"CSV 로드 중 예외 발생: {str(e)}")
    st.stop()


# 분할기 설정 
# RecursiveCharacterTextSplitter는 단락 -> 문장 -> 단어 순으로 재귀적 분할
# 이는 단락 단위가 의미적으로 가장 강하게 연관된 텍스트 조각으로 간주되기 때문
text_splitter = RecursiveCharacterTextSplitter(
    #separator = "\n", # 기본 값은 "\n\n"
    chunk_size = 1000,
    chunk_overlap = 0,
    length_function = len, # 텍스트 길이 계산 함수 지정 
    is_separator_regex=False # separator를 정규식이 아닌 일반 문자열로 처리 
)

# split_documents로 document 반환 
split_docs = text_splitter.split_documents(documents)

# DataFrame load
try:
    df = pd.read_csv(csv_file, encoding="utf-8-sig")
except Exception as e:
    try:
        df = pd.read_csv(csv_file, encoding="cp949")
    except Exception as e:
        st.error(f"pandas CSV 로드 중 오류 발생: {str(e)}")
        st.stop()

try:
    llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key = openai_api_key, temperature=0.7)
except Exception as e:
    st.error(f"OpenAI 모델 초기화 오류: {str(e)}. API 키가 올바른지 확인하세요.")
    st.stop()

# 프롬프트 템플릿 정의
prompt = PromptTemplate.from_template("""
당신은 경기도 맛집 추천 챗봇입니다. 
사용자가 요청한 지역이나 음식 종류에 맞는 맛집을 아래 데이터에서 찾아 추천해 주세요. 
데이터를 기반으로 식당 이름, 대표 음식, 주소, 전화번호를 포함하고, 대표 음식의 종류(예: 한식, 중식)와 특징을 추론하여 자연스러운 대화 형식으로 응답하세요. 
추천 이유와 음식의 특징을 반드시 포함하세요.

**데이터**:
{context}

**사용자 요청**:
{input}

**추천 형식**:
[식당 이름]을 추천드립니다! 
대표 메뉴는 [대표 음식]입니다. 이곳은 [추천 이유]로 유명합니다. 
[대표 음식]의 특징은 [음식의 특징]입니다. 
주소는 [소재지도로명주소]이고, 전화번호는 [맛집전화번호]입니다.

조건에 맞는 식당이 없으면, "조건에 맞는 식당을 찾을 수 없습니다."라고 응답하세요.
""")


# LangChain 체인 설정 
chain = prompt | llm

def recommend_restaurant(user_input):
    """맛집 추천 함수 
    Args:
        user_input (str): 사용자가 작성한 특정 지역과 음식 문자열 

    Returns:
        _type_: _description_
    """
    filtered_docs = []
    filtered_rows = [] 
    # 분할된 documents를 한 행씩 순회 
    for doc in split_docs:
        content = doc.page_content
        for _, row in df.iterrows():
            # 사용자가 입력한 문자열에 해당 DataFrame의 시군명 또는 음식이 포함되어있다면 
            # 추후에는 vectorstore 활용 
            if row["시군명"] in user_input or any(food.strip() in user_input for food in row["대표음식명"].split(",")):
                filtered_docs.append(doc)
                filtered_rows.append(row)
                break
    
    if not filtered_docs:
        filtered_docs = split_docs
        filtered_rows = df.to_dict('records')
    
    try:
        # Runnable interface: chain기반 하나의 입력을 하나의 출력으로 반환 
        response = chain.invoke({"input": user_input, "context": filtered_docs})
        # 만약 response가 content 속성을 가지고 있다면 응답이 올바르게 진행된 것. 
        if hasattr(response, "content"):
            return response.content, filtered_rows
        else:
            return response, filtered_rows
    except Exception as e:
        return f"추천 중 오류 발생: {str(e)}", []
      
# 지도 생성 함수
def create_map(df_restaurants, my_location):
    # 경기도 중심 (수원시)으로 지도 초기화
    default_lat, default_lon = 37.2636, 127.0286  # 수원시 좌표
    
    # 만약 my_location, 즉 내 위치 정보가 존재하는 경우 
    if my_location and my_location[0] and my_location[1]:
        map_center = [my_location[0], my_location[1]]
    else:
        map_center = [default_lat, default_lon]
    
    m = folium.Map(location=map_center, zoom_start=12)
    
    # 추천된 식당 마커 추가
    for row in df_restaurants:
        folium.Marker(
            [row["WGS84위도"], row["WGS84경도"]],
            tooltip=row["음식점명"],
            popup=f"{row['음식점명']}<br>{row['대표음식명']}<br>{row['소재지도로명주소']}"
        ).add_to(m)
    
    # 사용자 위치 마커 추가
    if my_location and my_location[0] and my_location[1]:
        folium.Marker(
            [my_location[0], my_location[1]],
            tooltip="내 위치",
            icon=folium.Icon(color="red")
        ).add_to(m)
    
    return m

# Streamlit UI
with st.form(key="recommendation_form"):
    user_input = st.text_input("어떤 맛집을 찾으신가요? (예: '고양시 한식', '광주시 오리구이')", "")
    submit_button = st.form_submit_button(label="추천받기")

# 사용자 위치 가져오기
location = streamlit_geolocation()
my_location = (location.get("latitude"), location.get("longitude")) if location else (None, None)

if submit_button and user_input:
    with st.spinner("맛집을 찾는 중입니다..."):
        result, filtered_rows = recommend_restaurant(user_input)
        if "오류" in result:
            st.error(result)
        else:
            st.markdown("### 추천 결과")
            st.markdown(result, unsafe_allow_html=True)
            
            # 지도 표시
            if filtered_rows:
                st.markdown("### 추천 식당 위치")
                try:
                    map_obj = create_map(filtered_rows, my_location)
                    folium_static(map_obj)
                except Exception as e:
                    st.error(f"지도 생성 중 오류 발생: {str(e)}")
            else:
                st.warning("추천된 식당이 없습니다. 지도를 표시할 수 없습니다.")

else:
    st.info("위 입력창에 지역이나 음식 종류를 입력한 후 '추천받기' 버튼을 눌러주세요.")

# 샘플 쿼리 제공
st.markdown("### 예시 쿼리")
st.markdown("""
- 고양시에서 한식 맛집 추천해줘
- 광주시에서 오리구이 맛집 찾아줘
- 과천시 복지리 맛집
""")