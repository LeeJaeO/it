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


1. 본 프로젝트에서는 MediaPipe 사용
CPU 환경에서도 실시간 처리가 가능할 만큼 가볍고 빠릅니다. 출입 시스템의 핵심인 '얼굴'을 직접 검출하므로 얼굴 인식 단계로 넘어가기 유리합니다.
더 정밀한 3D 움직임 분석이 필요하다면 Azure Kinect Body Tracking SDK를 사용하는 것이 가장 적합합니다.

2. 코드 내에서 if 1.0 <= z_meter <= 3.0: 조건을 사용하여,
유효 범위 밖(예: 4m)에 있는 사람은 바운딩 박스를 그리지 않고 무시하도록 구현되어 있습니다. 이를 통해 불필요한 연산을 줄이고 출입 의사가 있는 사용자만 선별합니다.

 가상 환경 활성화 (Windows)
.\venv\Scripts\activate


2. 얼굴 인식 학교 출입 시스템 (Face Recognition School Access System)

Azure Kinect DK의 심도(Depth) 센서와 dlib 기반의 얼굴 인식을 결합하여, 교내 재학생이 멈추지 않고 자연스럽게 통과할 수 있는 Walk-Through 출입 통제 시스템입니다.

  프로젝트 배경 (Persona & Problem)

 페르소나 A (대학생)
> "등교 중인데 양손 가득 짐이 너무 많아요. 강의동에 들어가려는데 학생증을 꺼내려고 짐을 다 바닥에 내려놓고, 지갑을 찾아서 태깅하느라 뒤에 줄이 길어졌습니다."

 기존 시스템의 문제점
* 물리적 접촉 필수: 짐이 많거나 손이 부족한 상황에서도 반드시 카드나 스마트폰을 꺼내야 함.
* 병목 현상: 한 명씩 멈춰서 태깅하는 과정에서 출입구 혼잡 발생.
* 매체 의존: 학생증 분실 시 출입 불가.

 본 프로젝트의 솔루션
* Hands-Free: 짐을 든 상태 그대로 카메라만 바라보면 인증 완료.
* Distance Filtering: 사용자가 출입 의도를 가지고 문 앞(유효 거리)에 접근했을 때만 정확히 인식.
* Existing DB Utilization: 별도의 생체 등록 없이, 입학 시 제출한 증명사진 DB를 활용하여 즉시 도입 가능.

---

  시스템 아키텍처 (Technical Approach)

### 1. 얼굴 인식 및 신원 확인 (Face Recognition)
교내 시스템에 등록된 **재학생 DB(입학 증명사진)**와 실시간 카메라 영상을 대조하여 신원을 확인합니다.
* [cite_start]**라이브러리:** `face_recognition` (dlib 기반) [cite: 19]
* [cite_start]**알고리즘:** 1. 등록된 증명사진(`student.jpg`)에서 얼굴 특징을 128차원 벡터(Embedding)로 변환하여 저장. [cite: 10, 18, 22]
  2. [cite_start]실시간 영상 속 얼굴을 벡터화하여 저장된 데이터와 유클리드 거리(유사도) 비교. [cite: 23]
  3. [cite_start]유사도가 임계값(Threshold) 이내일 경우 동일 인물로 판정. [cite: 26, 37]

### 2. 거리 기반 필터링 (Distance Filtering using Depth)
[cite_start]단순히 얼굴만 인식하는 것이 아니라, **Azure Kinect의 Depth 센서**를 활용하여 유효한 위치에 있는 사람만 선별합니다. [cite: 121, 169]
* [cite_start]**기능:** 카메라로부터 **0.3m ~ 0.5m (설정 가능)** 거리 내에 진입한 사용자만 인식. [cite: 206]
* [cite_start]**효과:** * 뒤쪽 배경에 지나가는 사람(Background Noise) 오인식 방지. [cite: 279]
  * [cite_start]사용자가 문 앞에 다가왔을 때만 작동하여 시스템 리소스 절약 및 UX 개선. [cite: 226, 237]

---

##  개발 환경 및 하드웨어 (Environment)

### Hardware
* **Camera:** Azure Kinect DK (RGB-D Camera)
* **Connection:** USB 3.0 (Data) & Power Adapter required

### Tech Stack
* **Language:** Python 3.10+
* **Libraries:**
  * `pyk4a`: Azure Kinect SDK Wrapper (Camera Control)
  * `opencv-python`: Image Processing & Visualization
  * [cite_start]`face_recognition` & `dlib`: Face Identification [cite: 17, 19]
  * `numpy`: Matrix Calculation

---

##  실행 방법 (How to Run)

### 1. 사전 준비 (Prerequisites)
* **Azure Kinect SDK v1.4.1** 설치
* **C++ Build Tools** 설치 (dlib 컴파일용)
* 프로젝트 폴더(`C:\kinect_project`)에 본인의 정면 사진을 **`student.jpg`** 이름으로 저장.

### 2. 가상 환경 설정 및 패키지 설치
```bash
python -m venv venv
.\venv\Scripts\activate
pip install opencv-python pyk4a face_recognition numpy
# dlib 설치 실패 시 .whl 파일로 수동 설치 권장
