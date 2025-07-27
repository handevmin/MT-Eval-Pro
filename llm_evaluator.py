import json
import logging
from typing import Dict, List, Optional, Tuple
from openai import OpenAI
from config import Config
import time
import re
import asyncio
import concurrent.futures
from threading import Thread
import pandas as pd

class LLMEvaluator:
    """LLM을 사용한 번역 품질 평가 클래스"""
    
    def __init__(self, api_key: str = None):
        self.config = Config()
        
        # OpenAI 클라이언트 초기화
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = OpenAI(api_key=self.config.OPENAI_API_KEY)
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 평가 프롬프트 템플릿
        self.evaluation_prompt_template = self._create_evaluation_prompt_template()
    
    def _create_evaluation_prompt_template(self) -> str:
        """번역 평가를 위한 프롬프트 템플릿을 생성합니다."""
        
        template = """
당신은 극도로 엄격한 기계 번역 품질 평가 전문가입니다. 주어진 소스 텍스트와 번역 텍스트를 다음 기준에 따라 **극도로 엄격하게** 평가해주세요. **기본적으로 1점부터 시작하고, 완벽할 때만 5점을 주세요.**

## 중요한 평가 원칙
1. **기본적으로 1점부터 시작**: 모든 번역은 기본적으로 1점으로 시작하고, 완벽할 때만 5점을 주세요.
2. **5점은 정말 완벽한 경우만**: 오타, 문법 오류, 어색한 표현, 의미 전달 오류가 전혀 없는 경우에만 5점을 부여하세요.
3. **문제점을 적극적으로 찾아서** 지적해주세요.
4. **의심스러운 부분이 있으면 낮은 점수**를 주세요.
5. **자연스럽지 않은 표현, 어색한 번역, 용어 일관성 부족** 등을 주의깊게 살펴보세요.
6. **참고 데이터 기준 준수**: 제공된 참고 데이터의 점수 분포를 참고하여 비슷한 품질의 번역에 비슷한 점수를 부여하세요.
7. **대부분의 번역은 1-3점**: 4-5점은 정말 예외적인 경우에만 부여하세요.

## 평가 메트릭스
{metrics_description}

## 평가 척도 (엄격 적용)
{scale_description}

## 평가 대상
- **소스 언어**: 영어
- **타겟 언어**: {target_language}
- **소스 텍스트**: {source_text}
- **번역 텍스트**: {target_text}

{reference_examples}

## 평가 요청
각 메트릭스(Accuracy, Omission/Addition, Compliance, Fluency, Overall)에 대해 1-5점으로 **극도로 엄격하게** 평가하고, **문제점을 중심으로** 구체적인 근거를 제시해주세요. **기본적으로 1점부터 시작하고, 완벽할 때만 5점을 주세요.**

응답은 반드시 다음 JSON 형식으로 제공해주세요:

```json
{{
    "evaluation": {{
        "Accuracy": {{
            "score": [1-5 사이의 정수],
            "reasoning": "평가 근거"
        }},
        "Omission/Addition": {{
            "score": [1-5 사이의 정수],
            "reasoning": "평가 근거"
        }},
        "Compliance": {{
            "score": [1-5 사이의 정수],
            "reasoning": "평가 근거"
        }},
        "Fluency": {{
            "score": [1-5 사이의 정수],
            "reasoning": "평가 근거"
        }},
        "Overall": {{
            "score": [1-5 사이의 정수],
            "reasoning": "전체적인 평가 근거"
        }}
    }},
    "summary": "번역의 전반적인 품질과 주요 문제점에 대한 요약",
    "suggestions": "개선 방안 제안 (선택사항)"
}}
```

평가 시 다음 사항을 **극도로 엄격하게** 고려해주세요:
1. 각 메트릭스의 정의를 정확히 따르고, **문제점을 적극적으로 찾으세요**.
2. 타겟 언어에서 **부자연스러운 표현이나 문법 오류**를 주의깊게 찾아보세요.
3. **완벽하지 않은 부분은 반드시 점수에 반영**하고 구체적인 근거를 제시하세요.
4. **의심스럽거나 개선이 필요한 부분이 있으면 낮은 점수**를 주세요.
5. **기본적으로 1점부터 시작**: 모든 번역은 기본적으로 1점으로 시작하세요.
6. **5점은 정말 완벽한 경우만**: 오타, 문법 오류, 어색한 표현이 전혀 없는 경우에만 5점을 주세요.
7. **대부분의 번역은 1-3점**: 4-5점은 정말 예외적인 경우에만 부여하세요.
"""
        
        return template
    
    def evaluate_translation(self, source_text: str, target_text: str, 
                           target_language: str, custom_metrics: dict = None, 
                           custom_scale: dict = None, reference_data: dict = None) -> Dict:
        """단일 번역을 평가합니다."""
        
        # 언어명 변환
        language_name = self.config.SUPPORTED_LANGUAGES.get(target_language, target_language)
        
        # 메트릭스와 스케일 설정 (사용자 정의 우선)
        metrics_to_use = custom_metrics or self.config.EVALUATION_METRICS
        scale_to_use = custom_scale or self.config.EVALUATION_SCALE
        
        self.logger.info(f"사용할 메트릭: {list(metrics_to_use.keys())}")
        self.logger.info(f"사용할 스케일: {list(scale_to_use.keys())}")
        
        # 메트릭스 정의 생성
        metrics_description = ""
        for metric, definition in metrics_to_use.items():
            metrics_description += f"**{metric}**: {definition}\n\n"
        
        # 척도 정의 생성
        scale_description = ""
        for score, definition in scale_to_use.items():
            scale_description += f"**{score}점**: {definition}\n\n"
        
        self.logger.info(f"메트릭 설명 길이: {len(metrics_description)}자")
        self.logger.info(f"스케일 설명 길이: {len(scale_description)}자")
        
        # 참고 데이터 처리
        reference_examples = ""
        self.logger.info(f"참고 데이터 확인: {reference_data}")
        if reference_data and 'dataframe' in reference_data:
            ref_df = reference_data['dataframe']
            score_cols = reference_data.get('score_columns', [])
            
            self.logger.info(f"참고 데이터 처리 중: {len(ref_df)}행, 점수 컬럼: {score_cols}")
            
            # 유사한 번역 쌍 찾기 (소스 텍스트 길이 기준)
            source_length = len(source_text)
            ref_df['source_length_diff'] = abs(ref_df.iloc[:, 0].str.len() - source_length)
            similar_examples = ref_df.nsmallest(3, 'source_length_diff')
            
            if not similar_examples.empty:
                reference_examples = "\n\n## 참고 예시 (사람이 평가한 유사한 번역들)\n"
                reference_examples += "**이 예시들의 점수 분포를 참고하여 비슷한 품질의 번역에 비슷한 점수를 부여하세요.**\n"
                self.logger.info(f"참고 예시 {len(similar_examples)}개 찾음")
                
                for idx, row in similar_examples.iterrows():
                    ref_source = str(row.iloc[0])[:100] + "..." if len(str(row.iloc[0])) > 100 else str(row.iloc[0])
                    ref_target = str(row.iloc[1])[:100] + "..." if len(str(row.iloc[1])) > 100 else str(row.iloc[1])
                    
                    reference_examples += f"\n**예시 {idx+1}:**\n"
                    reference_examples += f"- 소스: {ref_source}\n"
                    reference_examples += f"- 번역: {ref_target}\n"
                    
                    # 점수 정보 추가
                    for col in score_cols:
                        if col in row and pd.notna(row[col]):
                            reference_examples += f"- {col}: {row[col]}점\n"
                    reference_examples += "\n"
                
                self.logger.info(f"참고 예시 생성 완료: {len(reference_examples)}자")
            else:
                self.logger.warning("참고 예시를 찾을 수 없습니다.")
        else:
            self.logger.info("참고 데이터가 없습니다.")
        
        # 프롬프트 생성
        prompt = self.evaluation_prompt_template.format(
            metrics_description=metrics_description,
            scale_description=scale_description,
            source_text=source_text,
            target_text=target_text,
            target_language=language_name,
            reference_examples=reference_examples
        )
        
        # 참고 데이터 포함 여부 로그
        if reference_examples:
            self.logger.info("프롬프트에 참고 예시가 포함되었습니다.")
        else:
            self.logger.info("프롬프트에 참고 예시가 포함되지 않았습니다.")
        
        # 프롬프트 정보 로그
        self.logger.info(f"전체 프롬프트 길이: {len(prompt)}자")
        self.logger.info(f"평가 대상 텍스트 길이: 소스 {len(source_text)}자, 타겟 {len(target_text)}자")
        self.logger.info(f"프롬프트 시작 부분: {prompt[:200]}...")
        
        try:
            # API 호출
            response = self.client.chat.completions.create(
                model=self.config.DEFAULT_MODEL,
                messages=[
                    {"role": "system", "content": "당신은 극도로 엄격한 번역 품질 평가자입니다. 모든 번역은 기본적으로 1점부터 시작하고, 완벽할 때만 5점을 주세요. 대부분의 번역은 1-3점 범위에 있어야 하며, 4-5점은 정말 예외적인 경우에만 부여하세요. 문제점을 적극적으로 찾아서 지적해주세요. 참고 데이터의 점수 분포를 참고하여 일관된 평가를 해주세요."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config.MAX_TOKENS,
                temperature=self.config.TEMPERATURE
            )
            
            # 응답 파싱
            content = response.choices[0].message.content
            evaluation_result = self._parse_evaluation_response(content)
            
            # 메타데이터 추가
            evaluation_result.update({
                'source_text': source_text,
                'target_text': target_text,
                'target_language': target_language,
                'language_name': language_name,
                'model_used': self.config.DEFAULT_MODEL,
                'timestamp': time.time()
            })
            
            return evaluation_result
            
        except Exception as e:
            self.logger.error(f"번역 평가 중 오류 발생: {e}")
            return self._create_error_response(str(e))
    
    def _parse_evaluation_response(self, content: str) -> Dict:
        """LLM 응답을 파싱하여 구조화된 결과를 반환합니다."""
        
        # 디버깅을 위해 응답 내용 로그 기록
        self.logger.info(f"LLM 응답 내용 (처음 500자): {content[:500]}...")
        
        try:
            # 여러 패턴으로 JSON 블록 추출 시도
            json_patterns = [
                r'```json\s*(\{.*?\})\s*```',  # 기본 패턴
                r'```\s*(\{.*?\})\s*```',      # json 키워드 없는 경우
                r'(\{[\s\S]*?"evaluation"[\s\S]*?\})',  # evaluation이 포함된 JSON
                r'(\{[\s\S]*?\})'              # 마지막 시도: 전체 JSON
            ]
            
            json_str = None
            for pattern in json_patterns:
                json_match = re.search(pattern, content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    self.logger.info(f"JSON 패턴 매칭 성공: {pattern}")
                    break
            
            if json_str:
                # JSON 문자열 정리
                json_str = json_str.strip()
                
                # 잘못된 문자 제거 시도
                json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)  # 제어 문자 제거
                
                self.logger.info(f"파싱할 JSON: {json_str[:200]}...")
                
                result = json.loads(json_str)
                
                # 점수 유효성 검사 및 수정
                if 'evaluation' in result:
                    for metric in result['evaluation']:
                        score = result['evaluation'][metric].get('score')
                        if not isinstance(score, (int, float)) or score < 1 or score > 5:
                            self.logger.warning(f"유효하지 않은 점수 수정: {metric} = {score} -> 3")
                            result['evaluation'][metric]['score'] = 3
                        else:
                            # 소수점 점수를 정수로 변환 (내림)
                            if isinstance(score, float):
                                floored_score = int(score)
                                self.logger.info(f"소수점 점수를 정수로 변환 (내림): {metric} = {score} -> {floored_score}")
                                result['evaluation'][metric]['score'] = floored_score
                            # 문자열로 된 소수점도 처리
                            elif isinstance(score, str):
                                try:
                                    float_score = float(score)
                                    floored_score = int(float_score)
                                    self.logger.info(f"문자열 소수점 점수를 정수로 변환 (내림): {metric} = {score} -> {floored_score}")
                                    result['evaluation'][metric]['score'] = floored_score
                                except ValueError:
                                    self.logger.warning(f"문자열 점수를 숫자로 변환 실패: {metric} = {score} -> 3")
                                    result['evaluation'][metric]['score'] = 3
                
                self.logger.info("JSON 파싱 성공")
                return result
            
            else:
                self.logger.warning("JSON 블록을 찾을 수 없음, 텍스트에서 점수 추출 시도")
                return self._extract_scores_from_text(content)
                
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON 파싱 오류: {e}")
            self.logger.error(f"파싱 실패한 JSON: {json_str}")
            return self._extract_scores_from_text(content)
        except Exception as e:
            self.logger.error(f"응답 파싱 중 예상치 못한 오류: {e}")
            return self._create_error_response(str(e))
    
    def _extract_scores_from_text(self, content: str) -> Dict:
        """텍스트에서 점수를 추출합니다."""
        metrics = ['Accuracy', 'Omission/Addition', 'Compliance', 'Fluency', 'Overall']
        evaluation = {}
        
        self.logger.info("텍스트에서 점수 추출 시도...")
        
        for metric in metrics:
            # 다양한 메트릭별 점수 패턴 찾기 (소수점 포함)
            patterns = [
                rf'"{metric}"[\s\S]*?"score"[\s\S]*?[\s:](\d+(?:\.\d+)?)',  # JSON 형식 내부 (소수점 포함)
                rf'{metric}[\s\S]*?(\d+(?:\.\d+)?)[\s/]*5',  # "metric: 4.5/5" 형식
                rf'{metric}[\s\S]*?(\d+(?:\.\d+)?)점',  # "metric: 4.5점" 형식
                rf'{metric}[\s\S]*?score[\s\S]*?(\d+(?:\.\d+)?)',  # "metric score: 4.5" 형식
                rf'{metric}[\s\S]*?:[\s]*(\d+(?:\.\d+)?)',  # "metric: 4.5" 형식
                rf'{metric}[\s\S]*?(\d+(?:\.\d+)?)',  # 가장 넓은 패턴 (소수점 포함)
            ]
            
            score = 3  # 기본값
            reasoning = f"자동 추출 시도 ({metric})"
            
            for i, pattern in enumerate(patterns):
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    for match in matches:
                        try:
                            # 소수점이 포함된 경우도 처리
                            if '.' in match:
                                float_score = float(match)
                                found_score = int(float_score)  # 내림
                                self.logger.info(f"{metric} 소수점 점수 발견: {match} -> {found_score} (내림)")
                            else:
                                found_score = int(match)
                            
                            if 1 <= found_score <= 5:
                                score = found_score
                                reasoning = f"패턴 {i+1}로 추출된 점수: {found_score}"
                                self.logger.info(f"{metric} 점수 발견: {found_score} (패턴: {pattern})")
                                break
                        except ValueError:
                            continue
                    if score != 3:  # 기본값이 아닌 점수를 찾았으면 중단
                        break
            
            evaluation[metric] = {
                'score': score,
                'reasoning': reasoning
            }
        
        # 추출된 점수들 로그 기록
        scores_summary = {metric: evaluation[metric]['score'] for metric in metrics}
        self.logger.info(f"추출된 점수들: {scores_summary}")
        
        return {
            'evaluation': evaluation,
            'summary': "LLM 응답에서 JSON 파싱에 실패하여 텍스트에서 점수를 추출했습니다.",
            'suggestions': "",
            'parsing_error': True
        }
    
    def _create_error_response(self, error_message: str) -> Dict:
        """오류 응답을 생성합니다."""
        evaluation = {}
        for metric in ['Accuracy', 'Omission/Addition', 'Compliance', 'Fluency', 'Overall']:
            evaluation[metric] = {
                'score': 3,
                'reasoning': f"평가 중 오류 발생: {error_message}"
            }
        
        return {
            'evaluation': evaluation,
            'summary': f"평가 중 오류가 발생했습니다: {error_message}",
            'suggestions': "",
            'error': True,
            'error_message': error_message
        }
    
    def batch_evaluate(self, translation_pairs: List[Dict], 
                      max_concurrent: int = 5, custom_metrics: dict = None,
                      custom_scale: dict = None, reference_data: dict = None,
                      progress_callback=None) -> List[Dict]:
        """여러 번역을 병렬로 배치 평가합니다."""
        total = len(translation_pairs)
        self.logger.info(f"총 {total}개의 번역을 {max_concurrent}개씩 병렬 평가를 시작합니다.")
        
        if total <= max_concurrent:
            # 작은 배치는 한 번에 처리
            return self._parallel_evaluate_batch(
                translation_pairs, custom_metrics, custom_scale, reference_data, progress_callback
            )
        else:
            # 큰 배치는 청크로 나누어 처리
            results = []
            for i in range(0, total, max_concurrent):
                chunk = translation_pairs[i:i + max_concurrent]
                chunk_results = self._parallel_evaluate_batch(
                    chunk, custom_metrics, custom_scale, reference_data, progress_callback, start_index=i
                )
                results.extend(chunk_results)
                
                # 청크 간 짧은 딜레이 (API 제한 고려)
                if i + max_concurrent < total:
                    time.sleep(0.5)
            
            self.logger.info(f"병렬 배치 평가 완료: {len(results)}개 결과")
            return results
    
    def _parallel_evaluate_batch(self, translation_pairs: List[Dict], 
                               custom_metrics: dict = None, custom_scale: dict = None,
                               reference_data: dict = None, progress_callback=None, start_index: int = 0) -> List[Dict]:
        """번역 쌍들을 병렬로 평가합니다."""
        
        def evaluate_single(pair, index):
            try:
                # 진행 상황 업데이트
                current_text = pair.get('source_text', '')
                if progress_callback:
                    try:
                        progress_callback(start_index + index + 1, 
                                        len(translation_pairs) + start_index, 
                                        current_text)
                    except:
                        pass  # UI 콜백 오류는 무시
                
                self.logger.info(f"병렬 평가 시작: {start_index + index + 1}")
                
                result = self.evaluate_translation(
                    source_text=pair['source_text'],
                    target_text=pair['target_text'],
                    target_language=pair['language_code'],
                    custom_metrics=custom_metrics,
                    custom_scale=custom_scale,
                    reference_data=reference_data
                )
                
                # 원본 데이터와 병합
                result.update({
                    'translation_id': pair.get('id'),
                    'row_index': pair.get('row_index'),
                    'source_column': pair.get('source_column'),
                    'target_column': pair.get('target_column'),
                    'batch_index': start_index + index
                })
                
                self.logger.info(f"병렬 평가 완료: {start_index + index + 1}")
                return result
                
            except Exception as e:
                self.logger.error(f"번역 쌍 {start_index + index + 1} 평가 중 오류: {e}")
                error_result = self._create_error_response(str(e))
                error_result.update({
                    'translation_id': pair.get('id'),
                    'row_index': pair.get('row_index'),
                    'batch_index': start_index + index
                })
                return error_result
        
        # ThreadPoolExecutor를 사용한 병렬 처리
        # Streamlit 경고 메시지를 방지하기 위해 최대 worker 수 제한
        max_workers = min(len(translation_pairs), 10)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 모든 작업을 동시에 시작
            future_to_index = {
                executor.submit(evaluate_single, pair, i): i 
                for i, pair in enumerate(translation_pairs)
            }
            
            results = [None] * len(translation_pairs)
            
            # 완료된 작업들을 순서대로 수집
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    results[index] = result
                except Exception as e:
                    self.logger.error(f"Future 실행 중 오류: {e}")
                    error_result = self._create_error_response(str(e))
                    error_result.update({'batch_index': start_index + index})
                    results[index] = error_result
        
        return [r for r in results if r is not None]
    
    # def calculate_aggregate_scores(self, evaluation_results: List[Dict]) -> Dict:
    #     """평가 결과의 집계 점수를 계산합니다. (제거됨)"""
    #     # 집계 점수 계산 기능이 제거되었습니다.
    #     pass


def main():
    """테스트 함수"""
    # 설정에서 API 키가 있는지 확인
    config = Config()
    if not config.OPENAI_API_KEY:
        print("OpenAI API 키가 설정되지 않았습니다. .env 파일을 확인해주세요.")
        return
    
    evaluator = LLMEvaluator()
    
    # 테스트 번역 평가
    test_source = "The quick brown fox jumps over the lazy dog."
    test_target = "빠른 갈색 여우가 게으른 개를 뛰어넘습니다."
    test_language = "ko-kr"
    
    print("=== 테스트 번역 평가 ===")
    print(f"소스: {test_source}")
    print(f"번역: {test_target}")
    print(f"언어: {test_language}")
    
    result = evaluator.evaluate_translation(test_source, test_target, test_language)
    
    print("\n=== 평가 결과 ===")
    if 'evaluation' in result:
        for metric, data in result['evaluation'].items():
            print(f"{metric}: {data['score']}점 - {data['reasoning']}")
    
    print(f"\n요약: {result.get('summary', '')}")


if __name__ == "__main__":
    main() 