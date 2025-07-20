import os
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
import logging
from data_processor import DataProcessor
from llm_evaluator import LLMEvaluator
from config import Config

class EvaluationEngine:
    """번역 품질 평가 통합 엔진"""
    
    def __init__(self, api_key: str = None):
        self.config = Config()
        self.data_processor = DataProcessor()
        self.llm_evaluator = LLMEvaluator(api_key)
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 결과 저장 디렉토리 생성
        self._create_directories()
    
    def _create_directories(self):
        """필요한 디렉토리를 생성합니다."""
        directories = [self.config.RESULTS_DIR, self.config.REPORTS_DIR]
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                self.logger.info(f"디렉토리 생성: {directory}")
    
    def run_full_evaluation(self, translation_file: str = None, 
                          save_results: bool = True) -> Dict:
        """전체 평가 프로세스를 실행합니다."""
        
        self.logger.info("=== MT-Eval Pro 전체 평가 시작 ===")
        
        try:
            # 1. 데이터 로드
            self.logger.info("1. 데이터 로드 중...")
            metrics_data = self.data_processor.load_metrics_definitions(None)
            translation_df = self.data_processor.load_translation_data(translation_file)
            
            if translation_df.empty:
                raise ValueError("번역 데이터를 로드할 수 없습니다.")
            
            # 2. 번역 쌍 추출
            self.logger.info("2. 번역 쌍 추출 중...")
            translation_pairs = self.data_processor.extract_translation_pairs(translation_df)
            
            if not translation_pairs:
                raise ValueError("유효한 번역 쌍을 찾을 수 없습니다.")
            
            # 3. 데이터 유효성 검사
            self.logger.info("3. 데이터 유효성 검사 중...")
            valid_pairs = self.data_processor.validate_translation_data(translation_pairs)
            
            # 4. 통계 생성
            self.logger.info("4. 데이터 통계 생성 중...")
            language_stats = self.data_processor.get_language_statistics(valid_pairs)
            
            # 5. LLM 평가 실행
            self.logger.info("5. LLM 평가 실행 중...")
            evaluation_results = self.llm_evaluator.batch_evaluate(valid_pairs)
            
            # 6. 집계 점수 계산
            self.logger.info("6. 집계 점수 계산 중...")
            aggregate_scores = self.llm_evaluator.calculate_aggregate_scores(evaluation_results)
            
            # 7. 결과 통합
            final_results = {
                'metadata': {
                    'evaluation_timestamp': datetime.now().isoformat(),
                    'translation_file': translation_file or self.config.SAMPLE_FILE,
                    'total_pairs': len(translation_pairs),
                    'valid_pairs': len(valid_pairs),
                    'evaluated_pairs': len(evaluation_results),
                    'model_used': self.config.DEFAULT_MODEL
                },
                'language_statistics': language_stats,
                'evaluation_results': evaluation_results,
                'aggregate_scores': aggregate_scores,
                'detailed_results': self._create_detailed_analysis(evaluation_results, language_stats)
            }
            
            # 8. 결과 저장
            if save_results:
                self.logger.info("7. 결과 저장 중...")
                self._save_results(final_results)
            
            self.logger.info("=== 전체 평가 완료 ===")
            return final_results
            
        except Exception as e:
            self.logger.error(f"평가 중 오류 발생: {e}")
            raise e
    
    def run_full_evaluation_with_columns(self, translation_file: str = None, 
                                       source_column: str = None,
                                       target_column: str = None,
                                       language_code: str = None,
                                       max_evaluations: int = None,
                                       custom_metrics: dict = None,
                                       custom_scale: dict = None,
                                       save_results: bool = True) -> Dict:
        """사용자 지정 컬럼으로 전체 평가 프로세스를 실행합니다."""
        
        self.logger.info("=== MT-Eval Pro 사용자 지정 컬럼 평가 시작 ===")
        
        try:
            # 1. 데이터 로드
            self.logger.info("1. 데이터 로드 중...")
            metrics_data = self.data_processor.load_metrics_definitions(None)
            translation_df = self.data_processor.load_translation_data(translation_file)
            
            if translation_df.empty:
                raise ValueError("번역 데이터를 로드할 수 없습니다.")
            
            # 2. 번역 쌍 추출 (사용자 지정 컬럼 사용)
            self.logger.info("2. 번역 쌍 추출 중...")
            translation_pairs = self.data_processor.extract_translation_pairs(
                translation_df, 
                source_column=source_column,
                target_column=target_column,
                language_code=language_code
            )
            
            if not translation_pairs:
                raise ValueError("유효한 번역 쌍을 찾을 수 없습니다.")
            
            # 3. 데이터 유효성 검사
            self.logger.info("3. 데이터 유효성 검사 중...")
            valid_pairs = self.data_processor.validate_translation_data(translation_pairs)
            
            # 4. 최대 평가 개수 제한
            if max_evaluations and len(valid_pairs) > max_evaluations:
                valid_pairs = valid_pairs[:max_evaluations]
                self.logger.info(f"평가 개수를 {max_evaluations}개로 제한했습니다.")
            
            # 5. 통계 생성
            self.logger.info("4. 데이터 통계 생성 중...")
            language_stats = self.data_processor.get_language_statistics(valid_pairs)
            
            # 6. LLM 평가 실행
            self.logger.info("5. LLM 평가 실행 중...")
            
            # UI 진행 상황 콜백 가져오기
            progress_callback = None
            try:
                import streamlit as st
                if hasattr(st, 'session_state') and 'progress_callback' in st.session_state:
                    progress_callback = st.session_state.progress_callback
            except:
                pass  # Streamlit이 없는 환경에서는 무시
            
            # UI에서 설정된 병렬 처리 수 가져오기
            max_concurrent = 5  # 기본값
            try:
                if hasattr(st, 'session_state') and 'max_concurrent' in st.session_state:
                    max_concurrent = st.session_state.max_concurrent
            except:
                pass
            
            evaluation_results = self.llm_evaluator.batch_evaluate(
                valid_pairs, 
                max_concurrent=max_concurrent,
                custom_metrics=custom_metrics,
                custom_scale=custom_scale,
                progress_callback=progress_callback
            )
            
            # 7. 집계 점수 계산
            self.logger.info("6. 집계 점수 계산 중...")
            aggregate_scores = self.llm_evaluator.calculate_aggregate_scores(evaluation_results)
            
            # 8. 결과 통합
            final_results = {
                'metadata': {
                    'evaluation_timestamp': datetime.now().isoformat(),
                    'translation_file': translation_file or self.config.SAMPLE_FILE,
                    'source_column': source_column,
                    'target_column': target_column,
                    'language_code': language_code,
                    'language_name': self.config.SUPPORTED_LANGUAGES.get(language_code, language_code),
                    'total_pairs': len(translation_pairs),
                    'valid_pairs': len(valid_pairs),
                    'evaluated_pairs': len(evaluation_results),
                    'model_used': self.config.DEFAULT_MODEL
                },
                'language_statistics': language_stats,
                'evaluation_results': evaluation_results,
                'aggregate_scores': aggregate_scores,
                'detailed_results': self._create_detailed_analysis(evaluation_results, language_stats)
            }
            
            # 9. 결과 저장
            if save_results:
                self.logger.info("7. 결과 저장 중...")
                self._save_results(final_results)
                
                # 10. 원본 파일에 점수 업데이트
                if translation_file and evaluation_results:
                    self.logger.info("8. 원본 파일 점수 업데이트 중...")
                    updated_file_path = self.update_original_file_with_scores(
                        translation_file=translation_file,
                        evaluation_results=evaluation_results,
                        source_column=source_column,
                        target_column=target_column
                    )
                    
                    if updated_file_path:
                        final_results['metadata']['updated_file_path'] = updated_file_path
                        self.logger.info(f"점수가 업데이트된 파일: {updated_file_path}")
            
            self.logger.info("=== 사용자 지정 컬럼 평가 완료 ===")
            return final_results
            
        except Exception as e:
            self.logger.error(f"평가 중 오류 발생: {e}")
            raise e
    
    def _create_detailed_analysis(self, evaluation_results: List[Dict], 
                                language_stats: Dict) -> Dict:
        """상세 분석 결과를 생성합니다."""
        analysis = {
            'quality_distribution': {},
            'common_issues': {},
            'language_comparison': {},
            'recommendations': []
        }
        
        # 품질 분포 분석
        for metric in ['Accuracy', 'Omission/Addition', 'Compliance', 'Fluency', 'Overall']:
            scores = []
            for result in evaluation_results:
                if 'evaluation' in result and metric in result['evaluation']:
                    score = result['evaluation'][metric].get('score')
                    if isinstance(score, (int, float)):
                        scores.append(score)
            
            if scores:
                analysis['quality_distribution'][metric] = {
                    'excellent': len([s for s in scores if s >= 4.5]),
                    'good': len([s for s in scores if 3.5 <= s < 4.5]),
                    'acceptable': len([s for s in scores if 2.5 <= s < 3.5]),
                    'poor': len([s for s in scores if 1.5 <= s < 2.5]),
                    'very_poor': len([s for s in scores if s < 1.5])
                }
        
        # 언어별 비교
        for lang_code, stats in language_stats.items():
            lang_name = stats['language_name']
            lang_results = [r for r in evaluation_results 
                           if r.get('target_language') == lang_code]
            
            if lang_results:
                avg_overall = sum(r['evaluation']['Overall']['score'] 
                                for r in lang_results 
                                if 'evaluation' in r and 'Overall' in r['evaluation']) / len(lang_results)
                
                analysis['language_comparison'][lang_code] = {
                    'language_name': lang_name,
                    'average_overall_score': avg_overall,
                    'evaluation_count': len(lang_results),
                    'quality_level': self._get_quality_level(avg_overall)
                }
        
        # 추천사항 생성
        analysis['recommendations'] = self._generate_recommendations(analysis)
        
        return analysis
    
    def _get_quality_level(self, score: float) -> str:
        """점수를 기반으로 품질 수준을 반환합니다."""
        if score >= 4.5:
            return "우수"
        elif score >= 3.5:
            return "양호"
        elif score >= 2.5:
            return "보통"
        elif score >= 1.5:
            return "미흡"
        else:
            return "매우 미흡"
    
    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """분석 결과를 바탕으로 추천사항을 생성합니다."""
        recommendations = []
        
        # 품질 분포 기반 추천
        quality_dist = analysis.get('quality_distribution', {})
        if 'Overall' in quality_dist:
            overall_dist = quality_dist['Overall']
            total = sum(overall_dist.values())
            
            if total > 0:
                poor_ratio = (overall_dist['poor'] + overall_dist['very_poor']) / total
                if poor_ratio > 0.3:
                    recommendations.append("전체적인 번역 품질 개선이 필요합니다. 번역 시스템 재검토를 권장합니다.")
                
                excellent_ratio = overall_dist['excellent'] / total
                if excellent_ratio > 0.7:
                    recommendations.append("전체적으로 우수한 번역 품질을 보이고 있습니다.")
        
        # 언어별 비교 기반 추천
        lang_comparison = analysis.get('language_comparison', {})
        if lang_comparison:
            avg_scores = [(lang, data['average_overall_score']) 
                         for lang, data in lang_comparison.items()]
            avg_scores.sort(key=lambda x: x[1])
            
            if len(avg_scores) > 1:
                lowest_lang = avg_scores[0]
                if lowest_lang[1] < 3.0:
                    lang_name = lang_comparison[lowest_lang[0]]['language_name']
                    recommendations.append(f"{lang_name} 번역의 품질 개선이 특히 필요합니다.")
        
        return recommendations
    
    def _save_results(self, results: Dict):
        """평가 결과를 파일로 저장합니다."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON 결과 저장
        json_filename = f"evaluation_results_{timestamp}.json"
        json_path = os.path.join(self.config.RESULTS_DIR, json_filename)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"JSON 결과 저장: {json_path}")
        
        # Excel 결과 저장
        excel_filename = f"evaluation_results_{timestamp}.xlsx"
        excel_path = os.path.join(self.config.RESULTS_DIR, excel_filename)
        
        self._save_excel_results(results, excel_path)
        self.logger.info(f"Excel 결과 저장: {excel_path}")
    
    def update_original_file_with_scores(self, translation_file: str, evaluation_results: List[Dict], 
                                       source_column: str = None, target_column: str = None) -> str:
        """원본 파일의 빈 점수 칼럼들을 AI 평가 결과로 업데이트합니다."""
        
        try:
            # 원본 파일 로드
            df = pd.read_excel(translation_file)
            self.logger.info(f"원본 파일 로드: {translation_file}")
            
            # 점수 칼럼들이 존재하는지 확인
            score_columns = ['OVERALL', 'ACCURACY', 'OMISSION/ADDITION', 'COMPLIANCE', 'FLUENCY']
            missing_columns = []
            
            for col in score_columns:
                if col not in df.columns:
                    missing_columns.append(col)
            
            # 누락된 칼럼들을 추가 (빈 값으로)
            if missing_columns:
                self.logger.info(f"누락된 점수 칼럼들을 추가합니다: {missing_columns}")
                for col in missing_columns:
                    df[col] = None
            
            # 평가 결과를 원본 파일에 매핑
            updated_rows = 0
            
            for result in evaluation_results:
                row_index = result.get('row_index')
                if row_index is not None and 'evaluation' in result:
                    # 각 메트릭 점수를 해당 행의 칼럼에 업데이트
                    metric_mapping = {
                        'Overall': 'OVERALL',
                        'Accuracy': 'ACCURACY', 
                        'Omission/Addition': 'OMISSION/ADDITION',
                        'Compliance': 'COMPLIANCE',
                        'Fluency': 'FLUENCY'
                    }
                    
                    for metric, col_name in metric_mapping.items():
                        if metric in result['evaluation']:
                            score = result['evaluation'][metric].get('score')
                            if isinstance(score, (int, float)) and 1 <= score <= 5:
                                df.at[row_index, col_name] = score
                    
                    updated_rows += 1
            
            # 업데이트된 파일 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 원본 파일명에서 확장자 분리
            base_name = os.path.splitext(os.path.basename(translation_file))[0]
            updated_filename = f"{base_name}_evaluated_{timestamp}.xlsx"
            updated_path = os.path.join(os.path.dirname(translation_file), updated_filename)
            
            # 점수가 채워진 파일 저장
            df.to_excel(updated_path, index=False)
            
            self.logger.info(f"원본 파일 업데이트 완료: {updated_rows}행의 점수를 업데이트했습니다.")
            self.logger.info(f"업데이트된 파일 저장: {updated_path}")
            
            return updated_path
            
        except Exception as e:
            self.logger.error(f"원본 파일 업데이트 중 오류: {e}")
            return None
    
    def _save_excel_results(self, results: Dict, excel_path: str):
        """결과를 Excel 형식으로 저장합니다."""
        with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
            
            # 메타데이터 시트
            metadata_df = pd.DataFrame([results['metadata']]).T
            metadata_df.columns = ['값']
            metadata_df.to_excel(writer, sheet_name='메타데이터')
            
            # 언어별 통계 시트
            if results.get('language_statistics'):
                lang_stats_data = []
                for lang_code, stats in results['language_statistics'].items():
                    lang_stats_data.append({
                        '언어코드': lang_code,
                        '언어명': stats['language_name'],
                        '번역쌍수': stats['count'],
                        '평균소스길이': round(stats['avg_source_length'], 2),
                        '평균타겟길이': round(stats['avg_target_length'], 2)
                    })
                
                lang_stats_df = pd.DataFrame(lang_stats_data)
                lang_stats_df.to_excel(writer, sheet_name='언어별통계', index=False)
            
            # 상세 평가 결과 시트
            if results.get('evaluation_results'):
                detailed_data = []
                for result in results['evaluation_results']:
                    if 'evaluation' in result:
                        row = {
                            '번역ID': result.get('translation_id', ''),
                            '소스텍스트': result.get('source_text', '')[:100] + '...',
                            '타겟텍스트': result.get('target_text', '')[:100] + '...',
                            '언어': result.get('language_name', ''),
                            '언어코드': result.get('target_language', '')
                        }
                        
                        # 각 메트릭의 점수 추가
                        for metric in ['Accuracy', 'Omission/Addition', 'Compliance', 'Fluency', 'Overall']:
                            if metric in result['evaluation']:
                                row[f'{metric}_점수'] = result['evaluation'][metric].get('score', '')
                                row[f'{metric}_근거'] = result['evaluation'][metric].get('reasoning', '')[:50] + '...'
                        
                        row['요약'] = result.get('summary', '')[:100] + '...'
                        detailed_data.append(row)
                
                detailed_df = pd.DataFrame(detailed_data)
                detailed_df.to_excel(writer, sheet_name='상세평가결과', index=False)
            
            # 집계 점수 시트
            if results.get('aggregate_scores'):
                aggregate_data = []
                if 'overall' in results['aggregate_scores']:
                    for metric, stats in results['aggregate_scores']['overall'].items():
                        aggregate_data.append({
                            '메트릭': metric,
                            '평균점수': round(stats['mean'], 2),
                            '최소점수': stats['min'],
                            '최대점수': stats['max'],
                            '평가개수': stats['count']
                        })
                
                if aggregate_data:
                    aggregate_df = pd.DataFrame(aggregate_data)
                    aggregate_df.to_excel(writer, sheet_name='집계점수', index=False)
    
    def load_previous_results(self, results_file: str) -> Dict:
        """이전 평가 결과를 로드합니다."""
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            self.logger.info(f"이전 결과 로드 완료: {results_file}")
            return results
        except Exception as e:
            self.logger.error(f"결과 로드 오류: {e}")
            return {}
    
    def compare_evaluations(self, results1: Dict, results2: Dict) -> Dict:
        """두 평가 결과를 비교합니다."""
        comparison = {
            'summary': {},
            'metric_changes': {},
            'language_changes': {}
        }
        
        # 전체 점수 비교
        if ('aggregate_scores' in results1 and 'aggregate_scores' in results2 and
            'overall' in results1['aggregate_scores'] and 'overall' in results2['aggregate_scores']):
            
            for metric in ['Accuracy', 'Omission/Addition', 'Compliance', 'Fluency', 'Overall']:
                if (metric in results1['aggregate_scores']['overall'] and 
                    metric in results2['aggregate_scores']['overall']):
                    
                    score1 = results1['aggregate_scores']['overall'][metric]['mean']
                    score2 = results2['aggregate_scores']['overall'][metric]['mean']
                    change = score2 - score1
                    
                    comparison['metric_changes'][metric] = {
                        'previous_score': round(score1, 2),
                        'current_score': round(score2, 2),
                        'change': round(change, 2),
                        'improvement': change > 0
                    }
        
        return comparison


def main():
    """테스트 함수"""
    # 설정에서 API 키가 있는지 확인
    config = Config()
    if not config.OPENAI_API_KEY:
        print("OpenAI API 키가 설정되지 않았습니다.")
        print("1. .env 파일을 생성하고 OPENAI_API_KEY를 설정해주세요.")
        print("2. 또는 환경변수로 OPENAI_API_KEY를 설정해주세요.")
        return
    
    engine = EvaluationEngine()
    
    print("=== MT-Eval Pro 테스트 실행 ===")
    
    try:
        # 간단한 데이터 로드 테스트
        print("1. 데이터 로드 테스트...")
        translation_df = engine.data_processor.load_translation_data()
        
        if not translation_df.empty:
            print(f"번역 데이터 로드 성공: {translation_df.shape}")
            
            # 번역 쌍 추출 테스트
            pairs = engine.data_processor.extract_translation_pairs(translation_df)
            print(f"추출된 번역 쌍: {len(pairs)}개")
            
            if pairs:
                # 첫 번째 번역만 평가 (테스트용)
                print("2. 단일 번역 평가 테스트...")
                first_pair = pairs[0]
                result = engine.llm_evaluator.evaluate_translation(
                    first_pair['source_text'],
                    first_pair['target_text'],
                    first_pair['language_code']
                )
                
                print("평가 결과:")
                if 'evaluation' in result:
                    for metric, data in result['evaluation'].items():
                        print(f"  {metric}: {data['score']}점")
                
                print("\n전체 평가를 실행하려면 engine.run_full_evaluation()을 호출하세요.")
        else:
            print("번역 데이터를 찾을 수 없습니다.")
            
    except Exception as e:
        print(f"테스트 중 오류 발생: {e}")


if __name__ == "__main__":
    main() 