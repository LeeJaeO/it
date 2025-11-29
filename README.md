# it
#  얼굴 인식 학교 출입 시스템

Azure Kinect DK(Depth Camera)와 MediaPipe를 활용하여 사람을 검출하고, 3D 좌표(거리)를 측정하여 출입을 관리하는 시스템입니다.

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

# 가상 환경 활성화 (Windows)
.\venv\Scripts\activate
