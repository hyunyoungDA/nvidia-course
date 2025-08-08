## uv는 가상환경을 자동으로 관리하므로, 별도의 가상환경 설정 없이도 프로젝트를 실행할 수 있음 
## uvx: uv tool run의 축약형으로 임시 가상환경을 만들고 원하는 CLI 툴을 내려받아 설치 없이 바로 실행 가능 
## pycowsay: 전통적인 cowsay 유틸리티를 파이썬으로 구현한 패키지 
import requests

resp = requests.get("https://httpbin.org/get", params={"q": "test"})
resp.raise_for_status()
print(resp.json())


## uv python 명령어를 사용하여 원하는 Python 버전을 설치, 확인, 고정, 삭제 등을 할 수 있음. 
## 즉 uv만 있으면 시스템 전역 설정을 건드리지 않고도 독립적인 런타임 구축 가능 
## uv python pin <버전> -> 현재 프로젝트가 항상 해당 버전을 쓰도록 고정 
## uv python list 
## uv python install <버전>