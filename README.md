# IT
# 얼굴 인식 학교 출입 시스템 (School Access System)

Azure Kinect DK(Depth Camera)와 MediaPipe를 활용하여 사람을 검출하고, **1m ~ 3m 유효 거리** 내의 사용자만 인식하여 출입을 관리하는 Physical AI 시스템입니다.

## 하드웨어 요구사항 (Hardware Requirements)
* **카메라:** Azure Kinect DK
* **PC:** Windows 10/11
* **연결:** USB 3.0 포트 (파란색) 필수, 전원 어댑터 연결 필수
* **GPU:** NVIDIA GPU 권장 (Body Tracking 사용 시)

## 필수 설치 프로그램 (Prerequisites)
프로젝트 실행 전, 아래 프로그램들이 반드시 설치되어 있어야 합니다. **모두 기본 경로(C:\Program Files\...)에 설치해 주세요.**

1. **Azure Kinect Sensor SDK v1.4.1**
   * [다운로드 링크](https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/develop/docs/usage.md)
   * 설치 후 `Azure Kinect Viewer`로 작동 확인 필요.

2. **Azure Kinect Body Tracking SDK v1.1.2**
   * [다운로드 링크](https://www.microsoft.com/en-us/download/details.aspx?id=100942)
   * `pyk4a` 라이브러리 구동을 위해 필요.

3. **Microsoft Visual C++ Build Tools (필수)**
   * `pyk4a` 라이브러리 빌드를 위해 C++ 컴파일러가 필요합니다.
   * [다운로드 링크](https://visualstudio.microsoft.com/ko/downloads/) (Build Tools for Visual Studio 2022)
   * **설치 방법:** 설치 프로그램 실행 → **'C++를 사용한 데스크톱 개발(Desktop development with C++)'** 체크 → 설치 → **재부팅**.

---

## 개발 환경 설정 (Installation)

이 프로젝트는 **VS Code**와 **Python 3.10+** 환경에서 테스트되었습니다.
(경로 오류 방지를 위해 `C:\kinect_project` 와 같은 단순한 영어 경로 사용을 권장합니다.)

### 1. 가상 환경 생성 (Virtual Environment)
VS Code 터미널에서 다음 명령어로 가상 환경을 생성하고 활성화합니다.

```bash
# 가상 환경 생성
python -m venv venv

라이브러리 설치 (Dependencies)
가상 환경이 활성화된 상태((venv) 표시 확인)에서 필요한 패키지를 설치합니다.


pip install opencv-python pyk4a mediapipe numpy
Note: 설치 중 Building wheel for pyk4a failed 오류가 발생하면, 위 '필수 설치 프로그램'의 C++ Build Tools가 제대로 설치되었는지 확인하세요.


Azure Kinect의 전원과 USB를 PC에 연결합니다.

실행 중인 다른 키넥트 프로그램(예: Azure Kinect Viewer)이 있다면 종료합니다.

아래 명령어로 실행합니다.

python main.py


#1. 본 프로젝트에서는 MediaPipe 사용
CPU 환경에서도 실시간 처리가 가능할 만큼 가볍고 빠릅니다. 출입 시스템의 핵심인 '얼굴'을 직접 검출하므로 얼굴 인식 단계로 넘어가기 유리합니다.
더 정밀한 3D 움직임 분석이 필요하다면 Azure Kinect Body Tracking SDK를 사용하는 것이 가장 적합합니다.

#2. 코드 내에서 if 1.0 <= z_meter <= 3.0: 조건을 사용하여,
유효 범위 밖(예: 4m)에 있는 사람은 바운딩 박스를 그리지 않고 무시하도록 구현되어 있습니다. 이를 통해 불필요한 연산을 줄이고 출입 의사가 있는 사용자만 선별합니다.

# 가상 환경 활성화 (Windows)
.\venv\Scripts\activate
