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
당신은 기계 번역 품질 평가 전문가입니다. 주어진 소스 텍스트와 번역 텍스트를 다음 기준에 따라 평가해주세요.

## 평가 메트릭스
{metrics_description}

## 평가 척도
{scale_description}

## 평가 대상
- **소스 언어**: 영어
- **타겟 언어**: {target_language}
- **소스 텍스트**: {source_text}
- **번역 텍스트**: {target_text}

## 평가 요청
각 메트릭스(Accuracy, Omission/Addition, Compliance, Fluency, Overall)에 대해 1-5점으로 평가하고, 각 점수에 대한 구체적인 근거를 제시해주세요.

응답은 반드시 다음 JSON 형식으로 제공해주세요:

```json
{{
    "evaluation": {{
        "Accuracy": {{
            "score": [1-5 사이의 숫자],
            "reasoning": "평가 근거"
        }},
        "Omission/Addition": {{
            "score": [1-5 사이의 숫자],
            "reasoning": "평가 근거"
        }},
        "Compliance": {{
            "score": [1-5 사이의 숫자],
            "reasoning": "평가 근거"
        }},
        "Fluency": {{
            "score": [1-5 사이의 숫자],
            "reasoning": "평가 근거"
        }},
        "Overall": {{
            "score": [1-5 사이의 숫자],
            "reasoning": "전체적인 평가 근거"
        }}
    }},
    "summary": "번역의 전반적인 품질과 주요 문제점에 대한 요약",
    "suggestions": "개선 방안 제안 (선택사항)"
}}
```

평가 시 다음 사항을 고려해주세요:
1. 각 메트릭스의 정의를 정확히 따라주세요.
2. 타겟 언어의 문화적, 언어적 특성을 고려해주세요.
3. Google 스타일 가이드와 일관성을 고려해주세요.
4. 구체적이고 객관적인 근거를 제시해주세요.
"""
        
        return template
    
    def evaluate_translation(self, source_text: str, target_text: str, 
                           target_language: str, custom_metrics: dict = None, 
                           custom_scale: dict = None) -> Dict:
        """단일 번역을 평가합니다."""
        
        # 언어명 변환
        language_name = self.config.SUPPORTED_LANGUAGES.get(target_language, target_language)
        
        # 메트릭스와 스케일 설정 (사용자 정의 우선)
        metrics_to_use = custom_metrics or self.config.EVALUATION_METRICS
        scale_to_use = custom_scale or self.config.EVALUATION_SCALE
        
        # 메트릭스 정의 생성
        metrics_description = ""
        for metric, definition in metrics_to_use.items():
            metrics_description += f"**{metric}**: {definition}\n\n"
        
        # 척도 정의 생성
        scale_description = ""
        for score, definition in scale_to_use.items():
            scale_description += f"**{score}점**: {definition}\n\n"
        
        # 프롬프트 생성
        prompt = self.evaluation_prompt_template.format(
            metrics_description=metrics_description,
            scale_description=scale_description,
            source_text=source_text,
            target_text=target_text,
            target_language=language_name
        )
        
        try:
            # API 호출
            response = self.client.chat.completions.create(
                model=self.config.DEFAULT_MODEL,
                messages=[
                    {"role": "system", "content": "당신은 전문적인 번역 품질 평가자입니다. 정확하고 객관적인 평가를 제공해주세요."},
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
            # 다양한 메트릭별 점수 패턴 찾기
            patterns = [
                rf'"{metric}"[\s\S]*?"score"[\s\S]*?[\s:](\d+)',  # JSON 형식 내부
                rf'{metric}[\s\S]*?(\d+)[\s/]*5',  # "metric: 4/5" 형식
                rf'{metric}[\s\S]*?(\d+)점',  # "metric: 4점" 형식
                rf'{metric}[\s\S]*?score[\s\S]*?(\d+)',  # "metric score: 4" 형식
                rf'{metric}[\s\S]*?:[\s]*(\d+)',  # "metric: 4" 형식
                rf'{metric}[\s\S]*?(\d+)',  # 가장 넓은 패턴
            ]
            
            score = 3  # 기본값
            reasoning = f"자동 추출 시도 ({metric})"
            
            for i, pattern in enumerate(patterns):
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    for match in matches:
                        try:
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
                      custom_scale: dict = None, progress_callback=None) -> List[Dict]:
        """여러 번역을 병렬로 배치 평가합니다."""
        total = len(translation_pairs)
        self.logger.info(f"총 {total}개의 번역을 {max_concurrent}개씩 병렬 평가를 시작합니다.")
        
        if total <= max_concurrent:
            # 작은 배치는 한 번에 처리
            return self._parallel_evaluate_batch(
                translation_pairs, custom_metrics, custom_scale, progress_callback
            )
        else:
            # 큰 배치는 청크로 나누어 처리
            results = []
            for i in range(0, total, max_concurrent):
                chunk = translation_pairs[i:i + max_concurrent]
                chunk_results = self._parallel_evaluate_batch(
                    chunk, custom_metrics, custom_scale, progress_callback, start_index=i
                )
                results.extend(chunk_results)
                
                # 청크 간 짧은 딜레이 (API 제한 고려)
                if i + max_concurrent < total:
                    time.sleep(0.5)
            
            self.logger.info(f"병렬 배치 평가 완료: {len(results)}개 결과")
            return results
    
    def _parallel_evaluate_batch(self, translation_pairs: List[Dict], 
                               custom_metrics: dict = None, custom_scale: dict = None,
                               progress_callback=None, start_index: int = 0) -> List[Dict]:
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
                    custom_scale=custom_scale
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
    
    def calculate_aggregate_scores(self, evaluation_results: List[Dict]) -> Dict:
        """평가 결과의 집계 점수를 계산합니다."""
        if not evaluation_results:
            return {}
        
        metrics = ['Accuracy', 'Omission/Addition', 'Compliance', 'Fluency', 'Overall']
        aggregates = {}
        
        # 전체 평균
        for metric in metrics:
            scores = []
            for result in evaluation_results:
                if 'evaluation' in result and metric in result['evaluation']:
                    score = result['evaluation'][metric].get('score')
                    if isinstance(score, (int, float)) and 1 <= score <= 5:
                        scores.append(score)
            
            if scores:
                aggregates[metric] = {
                    'mean': sum(scores) / len(scores),
                    'min': min(scores),
                    'max': max(scores),
                    'count': len(scores)
                }
        
        # 언어별 평균
        language_aggregates = {}
        languages = set(result.get('target_language') for result in evaluation_results)
        
        for lang in languages:
            if lang:
                lang_results = [r for r in evaluation_results if r.get('target_language') == lang]
                language_aggregates[lang] = {}
                
                for metric in metrics:
                    scores = []
                    for result in lang_results:
                        if 'evaluation' in result and metric in result['evaluation']:
                            score = result['evaluation'][metric].get('score')
                            if isinstance(score, (int, float)) and 1 <= score <= 5:
                                scores.append(score)
                    
                    if scores:
                        language_aggregates[lang][metric] = {
                            'mean': sum(scores) / len(scores),
                            'min': min(scores),
                            'max': max(scores),
                            'count': len(scores)
                        }
        
        return {
            'overall': aggregates,
            'by_language': language_aggregates,
            'total_evaluations': len(evaluation_results)
        }


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