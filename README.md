# MT-Eval Pro

**기계 번역 품질 자동 평가 시스템**

LLM 모델을 활용하여 6개 언어의 기계 번역 품질을 자동으로 평가하는 전문적인 시스템입니다.

## 주요 기능

- **다국어 지원**: 영어에서 6개 언어로의 번역 평가 (아랍어, 중국어 간체, 일본어, 한국어, 프랑스어, 포르투갈어 브라질)
- **LLM 기반 평가**: OpenAI GPT-4를 활용한 정확하고 일관된 번역 품질 평가
- **다차원 메트릭스**: Accuracy, Omission/Addition, Compliance, Fluency, Overall 5가지 기준으로 평가
- **웹 인터페이스**: 사용자 친화적인 Streamlit 기반 웹 애플리케이션
- **시각화**: 다양한 차트와 그래프로 평가 결과 시각화
- **결과 내보내기**: Excel, JSON, CSV 등 다양한 형식으로 결과 저장

## 빠른 시작

### 1. 환경 설정

```bash
# 저장소 클론
git clone <repository-url>
cd MT-Eval Pro

# 패키지 설치
pip install -r requirements.txt
```

### 2. API 키 설정

`.env` 파일을 생성하고 OpenAI API 키를 설정하세요:

```bash
# .env 파일 생성
cp .env.example .env

# .env 파일 편집하여 API 키 입력
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. 실행

#### 웹 인터페이스 (권장)
```bash
python main.py web
```
또는
```bash
streamlit run app.py
```

브라우저에서 `http://localhost:8501`로 접속하세요.

#### 명령줄 인터페이스
```bash
# 기본 파일로 평가 실행
python main.py evaluate

# 특정 파일로 평가 실행
python main.py evaluate --file your_data.xlsx

# 시각화 포함 평가
python main.py evaluate --visualize

# 시스템 테스트
python main.py test

# 요구사항 확인
python main.py check
```

## 평가 메트릭스

### 평가 기준

1. **Accuracy (정확성)**: 소스 텍스트의 의미가 번역에서 정확하게 전달되는 정도
2. **Omission/Addition (누락/추가)**: 소스와 비교하여 번역에서 정보가 누락되거나 불필요하게 추가된 정도
3. **Compliance (준수성)**: Google 스타일 가이드 및 관련 지침을 따르는 정도
4. **Fluency (유창성)**: 번역이 대상 언어에서 문법적으로 올바르고 자연스럽게 들리는 정도
5. **Overall (전체)**: 전체적인 번역 품질에 대한 종합 평가

### 평가 척도

- **5점 (Very Good)**: 고품질 번역, 편집 불필요
- **4점 (Good)**: 정확하고 유창한 번역, 경미한 문제만 존재
- **3점 (Acceptable)**: 주요 아이디어 전달, 중간 정도의 문제 존재
- **2점 (Poor)**: 일부 의미 전달, 심각한 문제 다수 존재
- **1점 (Very Poor)**: 사용 불가능, 재번역 필요

## 데이터 형식

### 번역 데이터 파일 (Excel)

번역 데이터 파일은 다음과 같은 구조여야 합니다:

| Source (English) | Arabic (ae-ar) | Chinese (zh-cn) | Japanese (ja-jp) | Korean (ko-kr) | French (fr-fr) | Portuguese (pt-br) |
|------------------|----------------|-----------------|------------------|----------------|----------------|--------------------|
| Hello, world!    | مرحبا بالعالم   | 你好，世界！      | こんにちは、世界！  | 안녕하세요, 세계! | Bonjour le monde | Olá, mundo!      |

- 첫 번째 열: 영어 소스 텍스트
- 후속 열: 각 언어별 번역 텍스트
- 언어 코드를 컬럼명에 포함하는 것을 권장

## 웹 인터페이스 사용법

### 1. 평가 실행 탭
- OpenAI API 키 입력
- 번역 데이터 파일 업로드 또는 기본 파일 사용
- 평가 설정 조정
- 평가 실행

### 2. 결과 분석 탭
- 메트릭별 점수 차트
- 언어별 비교 차트
- 품질 분포 시각화
- 개선 추천사항

### 3. 상세 결과 탭
- 개별 번역 평가 결과
- 필터링 및 정렬 기능
- 결과 다운로드 (CSV, JSON)

### 4. 설정 탭
- 지원 언어 목록
- 평가 메트릭스 설명
- 평가 척도 안내

## 결과 파일

평가 완료 후 다음 파일들이 생성됩니다:

### results/ 디렉토리
- `evaluation_results_YYYYMMDD_HHMMSS.json`: 전체 평가 결과 (JSON)
- `evaluation_results_YYYYMMDD_HHMMSS.xlsx`: 전체 평가 결과 (Excel)

### reports/ 디렉토리
- `score_distribution.png`: 메트릭별 점수 분포 차트
- `language_comparison.png`: 언어별 비교 차트
- `metrics_analysis.png`: 메트릭별 상세 분석
- `quality_distribution.png`: 품질 분포 파이 차트
- `interactive_dashboard.html`: 인터랙티브 대시보드

## 고급 설정

### 환경변수

```bash
# OpenAI 설정
OPENAI_API_KEY=your_api_key
DEFAULT_MODEL=gpt-4.1
MAX_TOKENS=4000
TEMPERATURE=0.1
```

### 커스텀 메트릭스

`config.py` 파일에서 평가 메트릭스와 척도를 커스터마이징할 수 있습니다.

## 테스트

```bash
# 전체 시스템 테스트
python main.py test

# 개별 모듈 테스트
python data_processor.py
python llm_evaluator.py
python evaluation_engine.py
python visualization.py
```

## 주의사항

1. **API 사용량**: OpenAI API 사용량에 따라 비용이 발생할 수 있습니다.
2. **파일 형식**: Excel 파일의 인코딩과 형식을 확인해주세요.
3. **인터넷 연결**: API 호출을 위해 안정적인 인터넷 연결이 필요합니다.
4. **시간 소요**: 평가 개수에 따라 완료까지 시간이 소요될 수 있습니다.

## 문제 해결

### 자주 발생하는 문제

1. **API 키 오류**
   ```
   OpenAI API 키가 설정되지 않았습니다.
   ```
   → `.env` 파일에 올바른 API 키를 설정하세요.

2. **파일 로드 오류**
   ```
   번역 데이터를 로드할 수 없습니다.
   ```
   → Excel 파일 형식과 경로를 확인하세요.

3. **메모리 부족**
   → 평가할 데이터 양을 줄이거나 배치 크기를 조정하세요.

4. **패키지 설치 오류**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

### 로그 확인

```bash
# 상세 로그와 함께 실행
python main.py evaluate --verbose

# 로그 파일 확인
cat mt_eval_pro.log
```

## 지원

문제가 발생하거나 개선사항이 있으시면 이슈를 생성해주세요.

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

---

**MT-Eval Pro** - 정확하고 신뢰할 수 있는 번역 품질 평가를 위한 전문 도구 