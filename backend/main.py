from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, WebSocket
import os
import sqlite3
import subprocess
import asyncio
import re
import time
from datetime import datetime
from collections import Counter

import RPi.GPIO as GPIO
import numpy as np
import librosa
import noisereduce as nr
from pydub import AudioSegment
from sklearn.metrics.pairwise import cosine_similarity
import webrtcvad

print("현재 파일:", __file__)
# -------------------------------
# 1. 환경 설정
# -------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data")
OUT_DIR = os.path.join(DATA_DIR, "out")
DB_PATH = os.path.join(DATA_DIR, "meeting_logs.db")

WHISPER_PATH = os.path.expanduser("~/meeting-ai/whisper.cpp/build/bin/whisper-cli")
BASE_MODEL = os.path.expanduser("~/meeting-ai/whisper.cpp/models/ggml-base.bin")
TINY_MODEL = os.path.expanduser("~/meeting-ai/whisper.cpp/models/ggml-tiny.bin")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

vad = webrtcvad.Vad(3)
app = FastAPI()
# -------------------------------
# 2. GPIO 설정
# -------------------------------
BUTTON_PIN = 18
LED_PIN = 24
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(LED_PIN, GPIO.OUT)
GPIO.output(LED_PIN, GPIO.LOW)
# -------------------------------
# 3. 화자 분리 파라미터
# -------------------------------
SAME_SPK_THRESHOLD = 0.985
NEW_SPK_THRESHOLD = 0.965
A_RESCUE_GAP = 0.000
SMOOTHING_WEIGHT = 0.001



# 녹음 장치
RECORD_DEVICE = "plughw:3,0"

# 분석 후 원본/정제 파일 삭제
DELETE_AUDIO_AFTER_ANALYSIS = True

# 화자 상태 DB
reference_db = []
label_db = []
sample_count_db = []

prev_speaker = None
pending_speaker = None
pending_count = 0
# -------------------------------
# 4. 상태 초기화
# -------------------------------
def reset_speaker_state():
    global reference_db, label_db, sample_count_db
    global prev_speaker, pending_speaker, pending_count

    reference_db.clear()
    label_db.clear()
    sample_count_db.clear()

    prev_speaker = None
    pending_speaker = None
    pending_count = 0
# -------------------------------
# 5. DB 초기화
# -------------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS meetings
        (id INTEGER PRIMARY KEY AUTOINCREMENT,
         timestamp TEXT,
         filename TEXT,
         speaker TEXT,
         transcript TEXT,
         summary TEXT,
         todo TEXT)
    """)
    conn.commit()
    conn.close()

    try:
        os.chmod(DB_PATH, 0o600)
    except Exception:
        pass

init_db()
# -------------------------------
# 6. 유틸
# -------------------------------
def safe_remove(path):
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception as e:
        print(f"[WARN] 파일 삭제 실패: {path} / {e}")

def check_arecord_device(device=RECORD_DEVICE):
    try:
        result = subprocess.run(
            ["arecord", "-D", device, "--dump-hw-params", "-d", "1", "/dev/null"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=3
        )
        return result.returncode == 0
    except Exception as e:
        print(f"[WARN] 녹음 장치 검사 실패: {e}")
        return False

def clean_text(text):
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text

def strip_speaker_prefix(line):
    return re.sub(r"^\[Speaker [A-Z]\]:\s*", "", line).strip()

def is_similar_sentence(a, b):
    a = clean_text(a)
    b = clean_text(b)

    if not a or not b:
        return False
    if a == b:
        return True
    if a in b or b in a:
        return True

    set_a = set(a.split())
    set_b = set(b.split())
    if not set_a or not set_b:
        return False

    overlap = len(set_a & set_b) / max(len(set_a), len(set_b))
    return overlap >= 0.7
# -------------------------------
# 7. VAD
# -------------------------------
def is_speech(audio, sr=16000, frame_ms=20):
    if audio is None or len(audio) == 0:
        return False

    audio16 = np.clip(audio, -1.0, 1.0)
    audio16 = (audio16 * 32767).astype(np.int16)

    frame_len = int(sr * frame_ms / 1000)

    for i in range(0, len(audio16), frame_len):
        frame = audio16[i:i + frame_len]
        if len(frame) < frame_len:
            continue
        try:
            if vad.is_speech(frame.tobytes(), sr):
                return True
        except Exception:
            continue
    return False
# -------------------------------
# 8. 오디오 정제
# -------------------------------
def enhance_audio_locally(audio_path):
    try:
        print(f"--- 오디오 복구 및 무음 제거 시작: {audio_path} ---")

        # 1. 오디오 로드 (pydub 사용)
        audio = AudioSegment.from_file(audio_path, format="wav")
        # 노이즈 제거 정밀도를 위해 float32로 변환
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        rate = audio.frame_rate

        if len(samples) > 0:
            # 2. 노이즈 제거 수행
            reduced_noise = nr.reduce_noise(
                y=samples,
                sr=rate,
                prop_decrease=0.20
            )
           
            # [핵심] NaN 및 Infinity 값 처리 (아까 발생한 에러 해결 포인트)
            # 노이즈 제거 중 계산 오류로 생긴 잘못된 값들을 0으로 바꿉니다.
            reduced_noise = np.nan_to_num(reduced_noise)

            # 3. VAD를 위한 16비트 PCM 변환 및 정규화
            # 데이터가 깨지지 않도록 최대 볼륨을 기준으로 32767(int16) 범위에 맞춥니다.
            max_val = np.max(np.abs(reduced_noise))
            if max_val > 0:
                reduced_noise_normalized = (reduced_noise / max_val) * 32767
            else:
                reduced_noise_normalized = reduced_noise
           
            samples_int16 = reduced_noise_normalized.astype(np.int16)
           
            # 4. VAD 물리적 절삭 (무음 구간 삭제) 로직
            frame_ms = 20 # webrtcvad 표준 프레임
            frame_len = int(rate * frame_ms / 1000)
           
            active_segments = []
            print(f"[VAD] 무음 판별 중...")

            for i in range(0, len(samples_int16) - frame_len, frame_len):
                frame = samples_int16[i:i + frame_len]
                frame_bytes = frame.tobytes()
               
                try:
                    # 메인 코드 상단에서 선언된 vad 객체 사용
                    if vad.is_speech(frame_bytes, rate):
                        # 목소리 구간만 리스트에 추가 (무음은 여기서 '삭제'됨)
                        active_segments.append(frame)
                except Exception:
                    # 프레임 처리 오류 시 해당 조각만 건너뜀
                    continue
           
            if not active_segments:
                print("!!! [Warning] 유효 음성 구간을 찾지 못했습니다. 원본을 유지합니다.")
                final_samples = samples_int16
            else:
                # 목소리 구간만 자석처럼 이어 붙이기
                final_samples = np.concatenate(active_segments)
                reduction_ratio = (1 - len(final_samples)/len(samples_int16)) * 100
                print(f"--- 무음 제거 완료 (전체 대화 중 {reduction_ratio:.1f}% 공백 삭제) ---")

            # 5. 정제된 파일 저장
            enhanced_path = audio_path.replace(".wav", "_clean.wav")
            new_audio = AudioSegment(
                final_samples.tobytes(),
                frame_rate=rate,
                sample_width=2,
                channels=1
            )
            new_audio.export(enhanced_path, format="wav")
            return enhanced_path

    except Exception as e:
        print(f"오디오 정제 중 치명적 오류 발생: {e}")
        return audio_path
# -------------------------------
# 9. 화자 특징 추출
# -------------------------------
def extract_feature(y, sr):
    if y is None or len(y) < sr * 0.5: # 최소 분석 길이를 0.5초로 완화
        return None
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        delta = librosa.feature.delta(mfcc)
       
        f0 = librosa.yin(y, fmin=50, fmax=300)
        f0 = f0[~np.isnan(f0)]

        if len(f0) > 0:
            pitch_mean = np.mean(f0)
            pitch_std = np.std(f0)
        else:
            pitch_mean = 0
            pitch_std = 0

        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        centroid_mean = np.mean(spectral_centroid)
        centroid_std = np.std(spectral_centroid)
        # 1. Pitch 정규화 (보통 50~300Hz이므로 100으로 나누어 MFCC와 단위를 맞춤) 
        scaled_pitch = [pitch_mean / 100.0, pitch_std / 100.0] 
        # 2. Centroid 정규화 (보통 수천 단위이므로 1000으로 나누어 단위를 맞춤) 
        scaled_centroid = [centroid_mean / 1000.0, centroid_std / 1000.0]

        # 음색의 정적인 특징(mean)과 동적인 특징(std)을 모두 사용
        feature = np.concatenate([
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1),
            np.mean(delta, axis=1),
            scaled_pitch,
            scaled_centroid
        ]).astype(np.float32)

        norm = np.linalg.norm(feature)
        if norm < 1e-6:
            return None

        return feature / norm
   
    except Exception as e:
        print(f"[WARN] feature 추출 실패: {e}")
        return None
# -------------------------------
# 10. 화자 판별 (임계값 및 업데이트 로직 최적화)
# -------------------------------
def identify_speaker(feature):
    global reference_db, label_db, sample_count_db, prev_speaker
    
    if feature is None:
        return (prev_speaker or "Speaker A")
    
    feature = feature.reshape(1, -1)
    feature /= (np.linalg.norm(feature) + 1e-6)

    # 1. 최초 화자(Speaker A) 등록
    if not reference_db:
        reference_db.append(feature.copy())
        label_db.append("Speaker A")
        sample_count_db.append(1) # 발화 횟수 기록 시작
        prev_speaker = "Speaker A"
        return "Speaker A"

    # 2. 유사도 계산 및 시간적 관성 적용
    sims = [cosine_similarity(feature, ref)[0][0] for ref in reference_db]
    if prev_speaker in label_db:
        p_idx = label_db.index(prev_speaker)
        sims[p_idx] += SMOOTHING_WEIGHT

    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])
    best_speaker = label_db[best_idx]
    sim_with_A = sims[0]
    
    print("labels:", label_db)
    print("sims:", [round(float(s), 4) for s in sims])
    print("best:", best_speaker, round(best_sim, 4))

    # 3. 주 화자 보정 (A-Rescue)
    if best_speaker != "Speaker A" and (best_sim - sim_with_A) < A_RESCUE_GAP:
        best_idx, best_sim, best_speaker = 0, sim_with_A, "Speaker A"

    # 4. [핵심] 발화량 기반 '적응형 차등 업데이트' 로직
    if best_sim >= SAME_SPK_THRESHOLD:
        sample_count_db[best_idx] += 1 # 해당 화자의 발화 횟수 1 증가
        count = sample_count_db[best_idx]

        if best_idx == 0:  
            update_rate = 0.000 # Speaker A는 절대 기준점이므로 예외적으로 극소량 고정
        else:
            # 발화 횟수(count)에 따라 업데이트 비율을 동적으로 줄임 (Decay 로직)
            if count < 5:
                update_rate = 0.02 # 초기: 낯선 목소리이므로 빠르게 학습 (높은 비율)
            else:
                update_rate = 0.001 # 후기: 데이터가 충분히 쌓이면 프로필을 굳힘 (낮은 비율)
            
        reference_db[best_idx] = (1 - update_rate) * reference_db[best_idx] + update_rate * feature
        reference_db[best_idx] /= (np.linalg.norm(reference_db[best_idx]) + 1e-6)

    # 5. 신규 화자 등록
    elif best_sim < NEW_SPK_THRESHOLD:
        new_idx = len(label_db)
        new_label = f"Speaker {chr(ord('A') + new_idx)}" if new_idx < 26 else f"Speaker {new_idx + 1}"
            
        reference_db.append(feature.copy())
        label_db.append(new_label)
        sample_count_db.append(1) # 신규 화자 발화 횟수 초기화
        
        best_speaker = new_label
        print(f" !!! [NEW] {new_label} 동적 등록 (Sim: {best_sim:.3f})")

    prev_speaker = best_speaker
    return best_speaker
# -------------------------------
# 11. 요약 / 할 일
# -------------------------------
def extract_structured_summary(final_script):
    issues = []
    causes = []
    conclusions = []

    # 어떤 주제의 회의든 범용적으로 잡히도록 폭넓은 키워드 매칭
    issue_kws = ["문제", "이슈", "현상", "상황", "어려움", "안건", "주제", "발생", "단점", "현황", "목적"]
    cause_kws = ["원인", "이유", "때문에", "배경", "분석", "구조적", "한계", "부족", "간섭", "오류"]
    conclusion_kws = ["결론", "해결", "결정", "적용", "도입", "방향", "합의", "개선", "하자", "합시다", "좋습니다", "교체"]

    for line in final_script:
        if "]: " not in line: continue
        text = line.split("]: ")[-1].strip()
       
        # 10자 미만의 짧은 리액션(네, 아 등)은 요약에서 제외
        if len(text) < 10: continue

        if any(k in text for k in issue_kws): issues.append(text)
        elif any(k in text for k in cause_kws): causes.append(text)
        elif any(k in text for k in conclusion_kws): conclusions.append(text)

    # 중복 제거 및 정보량이 많은(긴) 문장 우선 추출
    def get_top(items, count=2):
        unique = list(dict.fromkeys(items))
        unique.sort(key=len, reverse=True) # 문장이 길수록 구체적일 확률이 높음
        return "\n".join(f"- {item}" for item in unique[:count]) if unique else "- 해당 내용 없음"

    summary_text = (
        f"[주요 안건 및 현황]\n{get_top(issues, 2)}\n\n"
        f"[원인 및 배경 분석]\n{get_top(causes, 1)}\n\n"
        f"[결정 사항 및 해결 방안]\n{get_top(conclusions, 2)}"
    )
    return summary_text

def extract_speaker_todos(final_script):
    todo_patterns = ["해주세", "해 주세", "확인", "검토", "준비", "정리", "공유", "보내",
                     "진행", "작성", "알아보", "일정", "테스트", "~해주고",
                     "~하겠습니다", "~하죠", "~바랍니다", "목표로", "계획입니다", "예정입니다"]
   
    exclude_keywords = ["여기까지", "수고", "반갑습니다", "어때요", "좋습니다", "맞아요", "아시죠"]
   
    todos_by_speaker = {}
    for line in final_script:
        if "]: " not in line: continue
        parts = line.split("]: ")
        speaker = parts[0].replace("[", "").strip()
        text = parts[1].strip()

        if len(text) > 8 and any(p in text for p in todo_patterns):
            if any(ex in text for ex in exclude_keywords):
                continue 
           
            if speaker not in todos_by_speaker:
                todos_by_speaker[speaker] = []
           
            if text not in todos_by_speaker[speaker]:
                todos_by_speaker[speaker].append(text)

    if not todos_by_speaker:
        return "추출된 할 일이 없습니다."

    result_lines = []
    for spk, tasks in todos_by_speaker.items():
        result_lines.append(f"[{spk}]")
        for t in tasks[:3]:
            result_lines.append(f"  - {t}")
   
    return "\n".join(result_lines)
# -------------------------------
# 12. SRT 파싱
# -------------------------------
def parse_srt(srt_path):
    sentences = []
    if not os.path.exists(srt_path):
        return sentences

    with open(srt_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    current = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue

        match = re.match(r'(\d+:\d+:\d+,\d+) --> (\d+:\d+:\d+,\d+)', line)
        if match:
            current["start"] = match.group(1).replace(",", ":")
            current["end"] = match.group(2).replace(",", ":")
            continue

        if line.isdigit():
            continue

        current["text"] = line
        sentences.append(current)
        current = {}

    return sentences

def str_to_ms(t):
    parts = list(map(int, t.split(":")))
    return parts[0] * 3600000 + parts[1] * 60000 + parts[2] * 1000 + parts[3]
# -------------------------------
# 13. 개별 문장 단위 화자 분리
# -------------------------------
def assign_speakers_per_sentence(sentences, audio, sr):
    sentence_speakers = [None] * len(sentences)
   
    # 이전 화자 기억 (초기값은 None)
    global prev_speaker

    for i, sent in enumerate(sentences):
        start_ms = str_to_ms(sent["start"])
        end_ms = str_to_ms(sent["end"])
        duration_ms = end_ms - start_ms

        # 1. 오디오 자르기
        segment = audio[start_ms:end_ms + 300] 
        samples = np.array(segment.get_array_of_samples()).astype(np.float32)

        # 2. 아주 짧은 문장(0.9초 미만)은 분석하지 않고 이전 화자 유지
        # "네", "아니오" 등은 음색 정보가 부족하여 화자가 튀는 원인이 됩니다.
        if duration_ms < 900:
            speaker = prev_speaker if prev_speaker else "Speaker A"
            sentence_speakers[i] = speaker
            print(f"[Sentence {i}] (Too Short) {sent['start']} -> {speaker}")
            continue

        # 3. 무음 및 유효성 검사
        if len(samples) == 0:
            speaker = prev_speaker if prev_speaker else "Speaker A"
        else:
            max_abs = np.max(np.abs(samples))
            if max_abs < 1e-6:
                speaker = prev_speaker if prev_speaker else "Speaker A"
            else:
                samples = samples / max_abs # 노멀라이즈
               
                # VAD 검사 (실제 말소리가 있는지)
                if not is_speech(samples, sr):
                    speaker = prev_speaker if prev_speaker else "Speaker A"
                else:
                    # 특징 추출 및 화자 판별
                    feat = extract_feature(samples, sr)
                    speaker = identify_speaker(feat)

        sentence_speakers[i] = speaker
        print(f"[Sentence {i}] {sent['start']} ~ {sent['end']} -> {speaker}")

    return sentence_speakers
# -------------------------------
# 14. 약한 smoothing
# -------------------------------
def smooth(seq):
    res = seq.copy()
    for i in range(1, len(seq) - 1):
        left = seq[i - 1]
        cur = seq[i]
        right = seq[i + 1]

        if cur is None:
            continue

        if left == right and cur != left and left is not None:
            res[i] = left
    return res
# -------------------------------
# 15. 핵심 분석
# -------------------------------
async def run_analysis_with_diarization(mid, wav_path):
    start_time = time.time()
    await asyncio.sleep(0.5)

    clean_wav = None

    try:
        print(f"[{mid}] 노이즈 제거 시작...")
        clean_wav = enhance_audio_locally(wav_path)
        txt_prefix = os.path.join(OUT_DIR, mid)

        print(f"[{mid}] Tiny 모델 타임라인 생성 중...")
        proc_t = await asyncio.create_subprocess_exec(
            WHISPER_PATH, "-m", TINY_MODEL, "-f", clean_wav,
            "-osrt", "-of", txt_prefix + "_t", "-l", "ko"
        )
        print(f"[{mid}] Base 모델 텍스트 추출 중...")
        proc_b = await asyncio.create_subprocess_exec(
            WHISPER_PATH, "-m", BASE_MODEL, "-f", clean_wav,
            "-otxt", "-of", txt_prefix + "_b", "-l", "ko",
            "-p", "4", "--beam-size", "5", "--best-of", "3"
        )
        ret_t, ret_b = await asyncio.gather(proc_t.wait(), proc_b.wait()) # whisper 병렬 실행(처리시간 감소)

        if ret_t != 0:
            print(f"[ERROR] Tiny 모델 실패")
            return "분석 불가", "요약 불가", "할 일 없음", 0

        if ret_b != 0:
            print(f"[WARN] Base 모델 실패 (tiny만 사용)")

        srt_path = txt_prefix + "_t.srt"
        txt_path = txt_prefix + "_b.txt"

        if not os.path.exists(srt_path):
            print(f"[{mid}] 오류: SRT 파일 없음")
            return "분석 불가", "요약 불가", "할 일 없음", 0

        if not os.path.exists(txt_path):
            print(f"[{mid}] 경고: TXT 파일 없음 (base 실패)")

        print(f"[{mid}] 화자 분리 시작...")
        sentences = parse_srt(srt_path)
        if not sentences:
            return "분석 불가", "요약 불가", "할 일 없음", 0

        audio = AudioSegment.from_wav(clean_wav)

        speaker_seq = assign_speakers_per_sentence(sentences, audio, audio.frame_rate)
        speaker_seq = smooth(speaker_seq)

        last_valid = "Speaker A"
        for i in range(len(speaker_seq)):
            if speaker_seq[i] is None:
                speaker_seq[i] = last_valid
            else:
                last_valid = speaker_seq[i]

        final_script = []
        for i, sent in enumerate(sentences[:len(speaker_seq)]):
            final_script.append(f"[{speaker_seq[i]}]: {sent['text']}")

        raw_text = "\n".join(final_script)
        result_txt_path = txt_prefix + "_result.txt" 
        with open(result_txt_path, "w", encoding="utf-8") as f: 
            for line in final_script: 
                f.write(line + "\n") 
        summary = extract_structured_summary(final_script)
        todo_res = extract_speaker_todos(final_script)

        duration = round(time.time() - start_time, 2)
        print(f"[{mid}] 분석 완료! 소요시간: {duration}초")

        return raw_text, summary, todo_res, duration

    except Exception as e:
        print(f"[{mid}] 분석 중 예외 발생: {e}")
        return "분석 불가", "요약 불가", "할 일 없음", 0

    finally:
        if DELETE_AUDIO_AFTER_ANALYSIS:
            safe_remove(wav_path)
            if clean_wav and clean_wav != wav_path:
                safe_remove(clean_wav)
# -------------------------------
# 16. HTTP API
# -------------------------------
@app.get("/api/history")
async def get_history():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT timestamp, transcript, summary, todo, speaker FROM meetings ORDER BY timestamp DESC")
    rows = c.fetchall()
    conn.close()
    return [dict(row) for row in rows]

@app.get("/api/history/date/{date_str}")
async def get_history_by_date(date_str: str):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM meetings WHERE timestamp LIKE ? ORDER BY timestamp DESC", (f"{date_str}%",))
        rows = c.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    except Exception as e:
        print(f"!!! error case: {e}")
        return []
# -------------------------------
# 17. WebSocket
# -------------------------------
@app.websocket("/ws/meeting")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    rec_proc = None
    current_mid = None
    current_wav_path = None

    while True:
        if GPIO.input(BUTTON_PIN) == GPIO.LOW:
            try:
                if rec_proc is None:
                    if not check_arecord_device(RECORD_DEVICE):
                        await websocket.send_json({
                            "status": "error",
                            "message": f"녹음 장치를 찾을 수 없습니다: {RECORD_DEVICE}"
                        })

                        while GPIO.input(BUTTON_PIN) == GPIO.LOW:
                            await asyncio.sleep(0.2)
                        continue

                    reset_speaker_state()

                    current_mid = datetime.now().strftime("%Y%m%d_%H%M%S")
                    current_wav_path = os.path.join(OUT_DIR, f"{current_mid}.wav")

                    GPIO.output(LED_PIN, GPIO.HIGH)
                    await websocket.send_json({"status": "recording"})

                    rec_proc = subprocess.Popen([
                        "arecord",
                        "-D", RECORD_DEVICE,
                        "-c", "1",
                        "-r", "16000",
                        "-f", "S16_LE",
                        "-t", "wav",
                        current_wav_path
                    ])

                else:
                    rec_proc.terminate()
                    try:
                        rec_proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        rec_proc.kill()

                    await asyncio.sleep(1.0)
                    GPIO.output(LED_PIN, GPIO.LOW)
                    await websocket.send_json({"status": "processing"})

                    text, summary, todo, duration = await run_analysis_with_diarization(
                        current_mid,
                        current_wav_path
                    )
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    conn = sqlite3.connect(DB_PATH)
                    conn.execute("""
                        INSERT INTO meetings (timestamp, filename, speaker, transcript, summary, todo)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (timestamp, current_mid, "multi", text, summary, todo))
                    conn.commit()
                    conn.close()

                    await websocket.send_json({
                        "status": "done",
                        "timestamp": timestamp,
                        "text": text,
                        "summary": summary,
                        "todo": todo,
                        "duration": f"{duration}초"
                    })

                    rec_proc = None
                    current_mid = None
                    current_wav_path = None

            except Exception as e:
                print(f"Error during process: {e}")

                if rec_proc:
                    try:
                        rec_proc.terminate()
                    except Exception:
                        pass

                rec_proc = None
                GPIO.output(LED_PIN, GPIO.LOW)

                await websocket.send_json({
                    "status": "error",
                    "message": str(e)
                })

            while GPIO.input(BUTTON_PIN) == GPIO.LOW:
                await asyncio.sleep(0.2)

        await asyncio.sleep(0.1)
# -------------------------------
# 18. 정적 파일 서빙
# -------------------------------
# backend/main_v3.py 와 frontend/index.html 구조 기준
app.mount("/", StaticFiles(directory=os.path.join(APP_DIR, "..", "frontend"), html=True))
# -------------------------------
# 19. 실행
# -------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
