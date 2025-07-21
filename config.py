import os

class Config:
    """MT-Eval Pro 설정 클래스"""
    
    # OpenAI API 설정 (기본값 사용)
    OPENAI_API_KEY = None  # 애플리케이션에서 직접 설정
    DEFAULT_MODEL = 'gpt-4'
    MAX_TOKENS = 4000
    TEMPERATURE = 0.1
    
    # 지원 언어 코드
    SUPPORTED_LANGUAGES = {
        'ae-ar': '아랍어',
        'zh-cn': '중국어 간체',
        'ja-jp': '일본어',
        'ko-kr': '한국어',
        'fr-fr': '프랑스어',
        'pt-br': '포르투갈어 브라질'
    }
    
    # 평가 메트릭스
    EVALUATION_METRICS = {
        'Accuracy': 'The degree of to which the meaning of the source text is accurately conveyed in the translation. It asseesses whether the translated content correctly reflects the facts, intent, and information of the original.',
        'Omission/Addition': 'The extent to which information is either missing (omiission) or unnecessarily added (addition) in the translation compared to the source. It focuses on content completeness and fidelity.',
        'Compliance': 'The extent to which the translation follows relevant guidelines, such as terminology, formatting, style guides, and domain-specific requirements, country standards. The target translations should follow Google Style & Terminology and their product/accessory/feature names including, commonly used expressions within Google. Also, jargons should be translated and used consistently within the same article.',
        'Fluency': 'The degree to which the translation is well-formed, grammatically correct, and natural-sounding in the target language. It evaluates the linguistic quality independetly of the source text. The target translation should conform to grammar and syntactic rules of the target language. Following issues such as collocation issues, punctuation & spelling issues, wrong punctuations, missing spacing, typos, unidiomatic or unnatural translation, uneasy to understand should be avoided.',
        'Overall': 'An overall assessment fo the translation\'s usability, based on a holistic evaluation of its accuracy, fluency, completeness, and comliance with relecant guidelines.'
    }
    
    # 평가 척도
    EVALUATION_SCALE = {
        5: 'Very Good. The translation is of high quality. It is accurate, complete, fluent, and fully compliant with all relevant guidelines. It reads naturally and requires no edit.',
        4: 'Good. The translation is accurate and fluent with only minor issues that do not significantly affect meaning or usability. It may benefit from light editing but is generally acceptable as-is.',
        3: 'Acceptable. The translation conveys the main idea of the source but includes several moderate issues that reduce clarity or quality. It may require noticeable editing for accuracy, compliance, or fluency.',
        2: 'Poor. The translation communicates some parts o the source meaning but contains multiples serious issues, such as inaccuracies, ommissions, or fluency problems. Heavy editing is required to make it usable.',
        1: 'Very Poor. The translation is unusable. It contains severe errors such as major mistranslations, omissions, or grammatical issues that make the meaning unclear or completely wrong. Substantial retranslation is needed.'
    }
    
    # 파일 경로
    METRICS_FILE = "Metrics n scale definitions_updated.xlsx"
    SAMPLE_FILE = "Sample B.xlsx"
    
    # 결과 저장 경로
    RESULTS_DIR = "results"
    REPORTS_DIR = "reports" 