# 스마트 회의 분석 시스템

> 라즈베리파이 기반 화자 분리 · 음성 인식(STT) · 회의 요약 · To-Do 자동 생성

<img width="784" height="373" alt="메인화면" src="https://github.com/user-attachments/assets/45a20dc6-1658-41f4-80e1-ce0a5dbf751d" />


## 1. 프로젝트 소개
회의 음성을 자동으로 녹음·분석하여 **화자별 회의록, 요약, 할 일(To-Do)** 을 웹 화면으로 제공하는 저비용 임베디드 회의 분석 시스템입니다. 클라우드 의존 없이 라즈베리파이 환경에서 동작하도록 구현했습니다.

- 소속: 충북대학교 전자정보대학 정보통신공학부
- 지도교수: 김태홍
- 참여학생: 우두윤, 강찬주

## 2. 개발 목적
회의 내용을 사람이 직접 기록하는 기존 방식의 한계를 개선하고, 저비용 임베디드 환경에서 회의 음성을 자동 분석하여 화자별 회의록과 할 일을 제공하는 시스템을 구현합니다.

## 3. 주요 기능
- GPIO 버튼 입력으로 녹음 시작, USB 마이크로 16kHz 음성 수집
- Noise Reduction + VAD 기반 음성 전처리 (잡음·무음 구간 제거)
- MFCC · Pitch · Spectral Centroid 특징 기반 화자 분리
- Whisper.cpp Tiny/Base 모델 병렬 실행 STT
- 화자별 회의록, 요약, To-Do 자동 생성
- FastAPI · WebSocket 기반 웹 인터페이스와 SQLite 회의 이력 저장

## 4. 시스템 구성도
<img width="470" height="112" alt="image" src="https://github.com/user-attachments/assets/28c0325a-10f6-4964-ba05-01fc9f4cedec" />

## 5. 기술 스택
| 구분 | 내용 |
|---|---|
| H/W | 라즈베리파이, USB 마이크, GPIO 버튼 |
| 음성 인식 | Whisper.cpp (Tiny / Base) |
| 화자 분리 | MFCC + Delta MFCC, Pitch(YIN), Spectral Centroid, 코사인 유사도 |
| 백엔드 | FastAPI, WebSocket |
| 저장소 | SQLite |
| 프론트엔드 | 웹 (요약 · To-Do · 회의 이력 화면) |

## 6. 화자 분리 방식
1. VAD로 음성 구간만 추출
2. 구간별 MFCC + Delta MFCC, Pitch(YIN), Spectral Centroid 특징 벡터 추출
3. 코사인 유사도 + Smoothing으로 문장 단위 화자 판별

## 7. 성능 결과
| 항목 | 결과 |
|---|---|
| STT 정확도 | 85~88% |
| 화자 분리 정확도 | 60~65% |
| 병렬처리 효과 | 처리시간 약 100초 단축 |
| VAD 적용 효과 | 화자 분리 성능 약 13% → 약 68% |
<img width="453" height="232" alt="성능비교12" src="https://github.com/user-attachments/assets/fed41f3d-b7c5-4a1e-b470-4a0f624b61bb" />
<img width="446" height="186" alt="성능비교1" src="https://github.com/user-attachments/assets/f3f19fbd-edf9-4380-bc10-fc2cf27e7b95" />


## 8. 실행 방법
```bash
# 서버 실행 (파일명은 실제 코드에 맞게 수정)
uvicorn main:app --host 0.0.0.0 --port 8000
```
브라우저에서 `http://<라즈베리파이 IP>:8000` 으로 접속합니다.

## 9. 프로젝트 구조
```
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
├── src/          # 녹음, 전처리, 화자분리, STT, 요약, API
├── web/          # 웹 화면
├── docs/images/  # 구현 화면, 시스템 구성도
└── scripts/      # 모델 다운로드, 실행 스크립트
```

## 10. 한계점 및 향후 개선 방향
- 다화자 지원 확대, 모델 경량화, 인증 및 HTTPS 적용으로 확장 가능
- 회의 기록 자동화로 기록 부담 감소, 화자별 발언과 할 일 분리로 회의 후 업무 파악 시간 단축 기대

## 11. 팀원
| 역할 | 이름 |
|---|---|
| 지도교수 | 김태홍 |
| 참여학생 | 우두윤, 강찬주 |
