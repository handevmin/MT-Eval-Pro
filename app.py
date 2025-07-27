import streamlit as st
import pandas as pd
import os
import json
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from evaluation_engine import EvaluationEngine
from visualization import VisualizationGenerator
from config import Config
import tempfile
import zipfile

# 페이지 설정
st.set_page_config(
    page_title="MT-Eval Pro",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed"
)

class MTEvalProApp:
    """MT-Eval Pro Streamlit 애플리케이션"""
    
    def __init__(self):
        self.config = Config()
        self.visualizer = VisualizationGenerator()
        
        # 세션 상태 초기화
        if 'evaluation_results' not in st.session_state:
            st.session_state.evaluation_results = None
        if 'evaluation_engine' not in st.session_state:
            st.session_state.evaluation_engine = None
    
    def main(self):
        """메인 애플리케이션 실행"""
        
        # 제목과 설명
        st.title("MT-Eval Pro")
        st.markdown("**기계 번역 품질 자동 평가 시스템**")
        st.markdown("---")
        
        # 메인 컨텐츠
        tab1, tab2, tab3 = st.tabs(["평가 실행", "상세 결과", "메트릭 설정"])
        
        with tab1:
            self.evaluation_tab()
        
        with tab2:
            self.detailed_results_tab()
        
        with tab3:
            self.metrics_settings_tab()
    def evaluation_tab(self):
        """평가 실행 탭"""
        st.header("MT-Eval Pro: 기계 번역 품질 평가")
        st.markdown("다음 단계를 따라 번역 품질 평가를 수행하세요.")
        
        # 단계별 UI
        st.markdown("---")
        
        # 1단계: API 키 설정
        st.markdown("### 1단계: OpenAI API 키 설정")
        st.markdown("GPT-4를 사용한 번역 평가를 위해 API 키가 필요합니다.")
        
        api_key = st.text_input(
            "API 키를 입력하세요:", 
            type="password",
            placeholder="sk-...",
            help="OpenAI 웹사이트에서 발급받은 API 키를 입력하세요."
        )
        
        if api_key:
            try:
                st.session_state.evaluation_engine = EvaluationEngine(api_key)
                st.success("API 키가 정상적으로 설정되었습니다.")
            except Exception as e:
                st.error(f"API 키 오류: {e}")
                return
        else:
            st.warning("API 키를 입력해주세요.")
            return
        
        st.markdown("---")
        
        # 2단계: 데이터 파일 선택
        st.markdown("### 2단계: 데이터 파일 선택")
        st.markdown("평가에 사용할 Excel 또는 CSV 파일을 업로드하세요.")
        
        uploaded_file = st.file_uploader(
            "Excel 또는 CSV 파일을 선택하세요:",
            type=['xlsx', 'xls', 'csv'],
            help="소스 텍스트와 번역 텍스트 컬럼이 포함된 Excel 또는 CSV 파일을 업로드하세요."
        )
        
        if uploaded_file:
            st.session_state.translation_file = uploaded_file
            st.success(f"파일이 업로드되었습니다: {uploaded_file.name}")
        else:
            st.info("Excel 파일을 업로드해주세요.")
            return
        
        st.markdown("---")
        
        # 3단계: 컬럼 선택
        st.markdown("### 3단계: 컬럼 선택")
        st.markdown("파일에서 소스 텍스트와 번역 텍스트가 있는 컬럼을 선택하세요.")
        
        # 데이터 로드하여 컬럼 정보 얻기
        try:
            from data_processor import DataProcessor
            processor = DataProcessor()
            
            translation_file = None
            if 'translation_file' in st.session_state:
                import tempfile
                # 파일 확장자에 따라 적절한 suffix 설정
                file_extension = st.session_state.translation_file.name.split('.')[-1].lower()
                suffix = f'.{file_extension}'
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(st.session_state.translation_file.getbuffer())
                    translation_file = tmp.name
            
            df = processor.load_translation_data(translation_file)
            if not df.empty:
                column_info = processor.get_available_columns(df)
                
                # 컬럼 선택을 2개 컬럼으로 배치
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**A. 소스 텍스트 컬럼 (원문)**")
                    source_column = st.selectbox(
                        "영어 원문이 포함된 컬럼:",
                        options=column_info['text_columns'],
                        index=0 if column_info['source_candidates'] else 0,
                        help="번역의 기준이 되는 영어 원문 텍스트가 있는 컬럼을 선택하세요."
                    )
                
                with col2:
                    st.markdown("**B. 번역 텍스트 컬럼 (번역문)**")
                    # 타겟 컬럼들과 자동 감지된 언어 정보
                    target_columns_info = processor.get_target_columns_with_languages(df)
                    available_targets = [info for info in target_columns_info if info['column_name'] != source_column]
                    
                    if available_targets:
                        selected_target_info = st.selectbox(
                            "번역된 텍스트가 있는 컬럼:",
                            options=available_targets,
                            format_func=lambda x: x['display_name'],
                            help="평가할 번역 텍스트가 포함된 컬럼을 선택하세요."
                        )
                
                if available_targets:
                    # 언어 정보 표시
                    st.markdown("**C. 언어 확인**")
                    col3, col4 = st.columns([1, 1])
                    
                    with col3:
                        st.success(f"감지된 언어: **{selected_target_info['language_name']}** ({selected_target_info['language_code']})")
                    
                    with col4:
                        # 언어 수동 변경 옵션
                        if st.button("언어 변경", type="secondary"):
                            st.session_state.show_language_selector = True
                    
                    # 언어 수동 선택
                    if st.session_state.get('show_language_selector', False):
                        st.markdown("**언어 수동 선택:**")
                        manual_language = st.selectbox(
                            "올바른 언어를 선택하세요:",
                            options=list(self.config.SUPPORTED_LANGUAGES.keys()),
                            format_func=lambda x: f"{self.config.SUPPORTED_LANGUAGES[x]} ({x})",
                            index=list(self.config.SUPPORTED_LANGUAGES.keys()).index(selected_target_info['language_code']) if selected_target_info['language_code'] in self.config.SUPPORTED_LANGUAGES else 0
                        )
                        final_language_code = manual_language
                        
                        if st.button("확인"):
                            st.session_state.show_language_selector = False
                            st.rerun()
                    else:
                        final_language_code = selected_target_info['language_code']
                    
                    # 세션 상태에 저장
                    st.session_state.column_selection = {
                        'source_column': source_column,
                        'target_column': selected_target_info['column_name'],
                        'language_code': final_language_code
                    }
                else:
                    st.error("**번역 텍스트 컬럼을 찾을 수 없습니다.**")
                    st.info("Excel 파일에 번역된 텍스트가 포함된 컬럼이 있는지 확인해주세요.")
                    return
            else:
                st.error("**데이터를 로드할 수 없습니다.**")
                st.info("파일이 올바른 Excel 형식인지 확인해주세요.")
                return
                
        except Exception as e:
            st.error(f"**컬럼 정보 로드 오류:** {e}")
            return
        
        st.markdown("---")
        
        # 4단계: 평가 실행
        st.markdown("### 4단계: 평가 실행")
        st.markdown("설정이 완료되었습니다. 평가를 시작하세요.")
        
        # 준비 상태 확인
        ready_for_evaluation = (
            st.session_state.evaluation_engine and 
            'column_selection' in st.session_state
        )
        
        if ready_for_evaluation:
            st.success("평가 준비 완료! 평가를 시작할 수 있습니다.")
            st.info("기본 설정: 최대 10개 번역 쌍을 5개씩 병렬로 평가합니다.")
            
            # 데이터 미리보기 버튼을 별도 섹션으로 분리
            if st.button("데이터 미리보기", type="secondary", use_container_width=True):
                self.preview_data()
        else:
            missing_items = []
            if not st.session_state.evaluation_engine:
                missing_items.append("API 키 설정")
            if 'column_selection' not in st.session_state:
                missing_items.append("컬럼 선택")
            st.error(f"준비 필요: {', '.join(missing_items)}")
            return
        
        # 평가 실행 버튼
        if st.button("평가 시작", type="primary", use_container_width=True):
            if not st.session_state.evaluation_engine:
                st.error("**API 키를 먼저 설정해주세요.**")
                return
            
            if 'column_selection' not in st.session_state:
                st.error("**컬럼 선택이 완료되지 않았습니다.**")
                st.info("3단계에서 소스 및 타겟 컬럼을 선택해주세요.")
                return
            
            # 기본 설정값 사용
            max_evaluations = 10
            max_concurrent = 5
            
            # 병렬 처리 설정 저장
            st.session_state.max_concurrent = max_concurrent
            self.run_evaluation(max_evaluations)
    
    def preview_data(self):
        """데이터 미리보기"""
        try:
            from data_processor import DataProcessor
            processor = DataProcessor()
            
            # 업로드된 파일 처리
            if 'translation_file' not in st.session_state:
                st.error("파일이 업로드되지 않았습니다.")
                return
            
            translation_file = None
            # 업로드된 파일을 임시 저장
            with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
                tmp.write(st.session_state.translation_file.getbuffer())
                translation_file = tmp.name
            
            # 데이터 로드
            df = processor.load_translation_data(translation_file)
            
            if not df.empty:
                st.success(f"데이터 로드 성공: {df.shape[0]}행 × {df.shape[1]}열")
                st.dataframe(df.head(10), use_container_width=True)
                
                # 번역 쌍 추출
                pairs = processor.extract_translation_pairs(df)
                if pairs:
                    st.info(f"추출된 번역 쌍: {len(pairs)}개")
                    
                    # 언어별 통계
                    stats = processor.get_language_statistics(pairs)
                    
                    stats_df = pd.DataFrame([
                        {
                            '언어': stat['language_name'],
                            '번역 쌍 수': stat['count'],
                            '평균 소스 길이': f"{stat['avg_source_length']:.1f}",
                            '평균 타겟 길이': f"{stat['avg_target_length']:.1f}"
                        }
                        for stat in stats.values()
                    ])
                    
                    st.dataframe(stats_df, use_container_width=True)
            else:
                st.error("데이터를 로드할 수 없습니다.")
                
        except Exception as e:
            st.error(f"데이터 미리보기 오류: {e}")
    
    def run_evaluation(self, max_evaluations):
        """평가 실행"""
        
        # 프로그레스 바와 상태 표시
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            detail_text = st.empty()
            metrics_container = st.empty()
        
        try:
            status_text.text("평가 준비 중...")
            detail_text.text("설정을 확인하고 데이터를 준비하는 중입니다.")
            
            # 업로드된 파일 처리
            if 'translation_file' not in st.session_state:
                st.error("파일이 업로드되지 않았습니다.")
                return
            
            translation_file = None
            with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
                tmp.write(st.session_state.translation_file.getbuffer())
                translation_file = tmp.name
            
            progress_bar.progress(10)
            status_text.text("데이터 로드 중...")
            detail_text.text("번역 데이터와 메트릭 정의를 로드하고 있습니다.")
            
            # 사용자 선택 컬럼 정보 가져오기
            column_selection = st.session_state.column_selection
            
            # 사용자 정의 메트릭과 스케일 가져오기
            custom_metrics = st.session_state.get('custom_metrics')
            custom_scale = st.session_state.get('custom_scale')
            
            # 참고 데이터 가져오기
            reference_data = st.session_state.get('reference_data')
            
            progress_bar.progress(20)
            status_text.text("번역 쌍 추출 중...")
            detail_text.text(f"소스: {column_selection['source_column']} → 타겟: {column_selection['target_column']}")
            
            # 실시간 진행 상황을 위한 콜백 함수
            def update_progress(current, total, current_text=""):
                progress = 20 + int((current / total) * 70)  # 20%에서 90%까지
                progress_bar.progress(progress)
                status_text.text(f"병렬 번역 평가 중... ({current}/{total})")
                
                # 현재 평가 중인 텍스트 표시 (안전하게)
                if current_text:
                    display_text = current_text.replace('\n', ' ').strip()
                    if len(display_text) > 80:
                        display_text = display_text[:80] + "..."
                    detail_text.text(f"병렬 처리 중: {display_text}")
                else:
                    detail_text.text(f"병렬 평가 진행 중... ({current}/{total})")
                
                # 현재까지의 진행률을 메트릭으로 표시
                with metrics_container.container():
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("진행률", f"{(current/total)*100:.1f}%")
                    with col2:
                        st.metric("완료", current)
                    with col3:
                        st.metric("남은 평가", total - current)
                    with col4:
                        # 병렬 처리로 더 빠른 시간 추정
                        max_concurrent = st.session_state.get('max_concurrent', 5)
                        time_per_eval = 3 / max_concurrent  # 병렬 처리 고려
                        remaining_time = int((total - current) * time_per_eval)
                        if remaining_time > 60:
                            st.metric("예상 남은 시간", f"{remaining_time//60}분 {remaining_time%60}초")
                        else:
                            st.metric("예상 남은 시간", f"{remaining_time}초")
            
            # 평가 엔진에 콜백 함수 전달을 위해 세션 상태에 저장
            st.session_state.progress_callback = update_progress
            
            # 평가 실행
            results = st.session_state.evaluation_engine.run_full_evaluation_with_columns(
                translation_file=translation_file,
                source_column=column_selection['source_column'],
                target_column=column_selection['target_column'],
                language_code=column_selection['language_code'],
                max_evaluations=max_evaluations,
                custom_metrics=custom_metrics,
                custom_scale=custom_scale,
                reference_data=reference_data,
                save_results=True
            )
            
            progress_bar.progress(95)
            status_text.text("결과 처리 중...")
            detail_text.text("평가 결과를 정리하고 파일을 생성하는 중입니다.")
            
            progress_bar.progress(100)
            status_text.text("평가 완료!")
            detail_text.text("모든 평가가 성공적으로 완료되었습니다.")
            
            # 최종 메트릭 표시
            if 'aggregate_scores' in results and 'overall' in results['aggregate_scores']:
                with metrics_container.container():
                    st.subheader("최종 평가 결과")
                    metrics_data = results['aggregate_scores']['overall']
                    
                    cols = st.columns(len(metrics_data))
                    for i, (metric, data) in enumerate(metrics_data.items()):
                        with cols[i]:
                            score = data['mean']
                            st.metric(
                                label=metric.replace('/', '/\n'),
                                value=f"{score:.2f}",
                                delta=f"{self.get_quality_level(score)}"
                            )
            
            # 결과 저장
            st.session_state.evaluation_results = results
            
            st.success("평가가 성공적으로 완료되었습니다!")
            
            # 업데이트된 파일 다운로드 링크 제공
            if 'updated_file_path' in results.get('metadata', {}):
                updated_file_path = results['metadata']['updated_file_path']
                with open(updated_file_path, 'rb') as f:
                    st.download_button(
                        label="점수가 채워진 원본 파일 다운로드",
                        data=f.read(),
                        file_name=os.path.basename(updated_file_path),
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="download_updated_file"
                    )
            
            # 빠른 결과 표시
            self.show_quick_results(results)
            
        except Exception as e:
            progress_bar.progress(0)
            status_text.text("오류 발생")
            detail_text.text(str(e))
            st.error(f"평가 중 오류 발생: {e}")
        finally:
            # 클린업
            if 'progress_callback' in st.session_state:
                del st.session_state.progress_callback
    
    def show_quick_results(self, results):
        """빠른 결과 표시"""
        st.markdown("### 빠른 결과")
        
        if 'aggregate_scores' in results and 'overall' in results['aggregate_scores']:
            metrics_data = results['aggregate_scores']['overall']
            
            # 메트릭별 점수 표시
            cols = st.columns(len(metrics_data))
            
            for i, (metric, data) in enumerate(metrics_data.items()):
                with cols[i]:
                    score = data['mean']
                    st.metric(
                        label=metric,
                        value=f"{score:.2f}",
                        delta=f"({self.get_quality_level(score)})"
                    )
        
        # 언어별 비교
        if ('detailed_results' in results and 
            'language_comparison' in results['detailed_results']):
            
            st.markdown("#### 언어별 전체 점수")
            
            lang_data = results['detailed_results']['language_comparison']
            lang_df = pd.DataFrame([
                {
                    '언어': data['language_name'],
                    '평균 점수': f"{data['average_overall_score']:.2f}",
                    '품질 수준': data['quality_level'],
                    '평가 개수': data['evaluation_count']
                }
                for data in lang_data.values()
            ])
            
            st.dataframe(lang_df, use_container_width=True)
    def detailed_results_tab(self):
        """상세 결과 탭"""
        st.header("상세 평가 결과")
        
        if st.session_state.evaluation_results is None:
            st.info("먼저 평가를 실행해주세요.")
            return
        
        results = st.session_state.evaluation_results
        
        # 상세 결과 테이블
        if 'evaluation_results' in results:
            st.subheader("개별 평가 결과")
            
            # 필터링 옵션
            col1, col2 = st.columns(2)
            
            with col1:
                # 언어 필터
                available_languages = set()
                for result in results['evaluation_results']:
                    if 'language_name' in result:
                        available_languages.add(result['language_name'])
                
                selected_languages = st.multiselect(
                    "언어 필터",
                    options=list(available_languages),
                    default=list(available_languages)
                )
            
            with col2:
                # 점수 범위 필터
                score_range = st.slider(
                    "전체 점수 범위",
                    min_value=1.0,
                    max_value=5.0,
                    value=(1.0, 5.0),
                    step=0.1
                )
            
            # 필터링된 결과 표시
            filtered_results = []
            
            for result in results['evaluation_results']:
                if ('language_name' in result and 
                    result['language_name'] in selected_languages and
                    'evaluation' in result and 
                    'Overall' in result['evaluation']):
                    
                    overall_score = result['evaluation']['Overall'].get('score', 0)
                    if score_range[0] <= overall_score <= score_range[1]:
                        filtered_results.append(result)
            
            # 결과 테이블 생성
            if filtered_results:
                table_data = []
                
                for result in filtered_results[:100]:  # 최대 100개만 표시
                    row = {
                        'ID': result.get('translation_id', ''),
                        '언어': result.get('language_name', ''),
                        '소스 텍스트': result.get('source_text', '')[:100] + '...',
                        '번역 텍스트': result.get('target_text', '')[:100] + '...'
                    }
                    
                    # 메트릭 점수 추가
                    if 'evaluation' in result:
                        for metric in ['Accuracy', 'Omission/Addition', 'Compliance', 'Fluency', 'Overall']:
                            if metric in result['evaluation']:
                                row[metric] = result['evaluation'][metric].get('score', 0)
                    
                    table_data.append(row)
                
                df = pd.DataFrame(table_data)
                st.dataframe(df, use_container_width=True)
                
                # 다운로드 버튼
                csv = df.to_csv(index=False)
                st.download_button(
                    label="CSV 다운로드",
                    data=csv,
                    file_name=f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="download_csv_results"
                )
            else:
                st.info("선택한 조건에 맞는 결과가 없습니다.")
        
        # 결과 파일 다운로드
        st.subheader("결과 다운로드")
        
        if st.button("전체 결과 다운로드 (JSON)", key="prepare_json_download"):
            json_str = json.dumps(results, ensure_ascii=False, indent=2)
            st.download_button(
                label="JSON 파일 다운로드",
                data=json_str,
                file_name=f"mt_eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                key="download_json_results"
            )
    
    def metrics_settings_tab(self):
        """메트릭 및 스케일 설정 탭"""
        st.header("평가 메트릭 및 스케일 설정")
        
        # 현재 설정 로드
        if 'custom_metrics' not in st.session_state:
            st.session_state.custom_metrics = self.config.EVALUATION_METRICS.copy()
        
        if 'custom_scale' not in st.session_state:
            st.session_state.custom_scale = self.config.EVALUATION_SCALE.copy()
        
        # 메트릭 설정 섹션
        st.subheader("평가 메트릭 정의")
        st.markdown("각 메트릭의 정의를 수정할 수 있습니다.")
        
        metrics_updated = False
        
        for metric, current_definition in st.session_state.custom_metrics.items():
            with st.expander(f"{metric} 설정"):
                new_definition = st.text_area(
                    f"{metric} 정의",
                    value=current_definition,
                    height=100,
                    key=f"metric_{metric}"
                )
                
                if new_definition != current_definition:
                    st.session_state.custom_metrics[metric] = new_definition
                    metrics_updated = True
        
        # 스케일 설정 섹션
        st.subheader("평가 스케일 정의")
        st.markdown("각 점수별 기준을 수정할 수 있습니다.")
        
        scale_updated = False
        
        for score, current_definition in st.session_state.custom_scale.items():
            with st.expander(f"{score}점 기준 설정"):
                new_definition = st.text_area(
                    f"{score}점 정의",
                    value=current_definition,
                    height=80,
                    key=f"scale_{score}"
                )
                
                if new_definition != current_definition:
                    st.session_state.custom_scale[score] = new_definition
                    scale_updated = True
        
        # 액션 버튼들
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("기본값으로 리셋"):
                st.session_state.custom_metrics = self.config.EVALUATION_METRICS.copy()
                st.session_state.custom_scale = self.config.EVALUATION_SCALE.copy()
                st.success("설정이 기본값으로 리셋되었습니다.")
                st.rerun()
        
        with col2:
            if st.button("설정 저장"):
                self.save_custom_settings()
                st.success("사용자 설정이 저장되었습니다.")
        
        with col3:
            if st.button("설정 내보내기"):
                self.export_settings()
        
        # 참고 데이터 설정
        st.subheader("참고 데이터 설정")
        st.markdown("사람이 이미 평가한 데이터를 업로드하여 AI 평가의 정확도를 향상시킬 수 있습니다.")
        
        reference_file = st.file_uploader(
            "참고 데이터 파일 업로드 (Excel/CSV)",
            type=['xlsx', 'xls', 'csv'],
            help="이미 사람이 평가한 번역 데이터가 포함된 Excel 또는 CSV 파일을 업로드하세요. 점수 컬럼이 포함되어 있어야 합니다.",
            key="reference_data_uploader"
        )
        
        if reference_file:
            try:
                # 참고 데이터 미리보기
                import tempfile
                # 파일 확장자에 따라 적절한 suffix 설정
                file_extension = reference_file.name.split('.')[-1].lower()
                suffix = f'.{file_extension}'
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(reference_file.getbuffer())
                    ref_file_path = tmp.name
                
                # 데이터 로드 및 컬럼 확인
                from data_processor import DataProcessor
                processor = DataProcessor()
                ref_df = processor.load_translation_data(ref_file_path)
                
                if not ref_df.empty:
                    st.success(f"참고 데이터 로드 성공: {ref_df.shape[0]}행 × {ref_df.shape[1]}열")
                    
                    # 점수 컬럼 확인
                    score_columns = []
                    for col in ref_df.columns:
                        if any(metric.lower() in col.lower() for metric in ['accuracy', 'omission', 'addition', 'compliance', 'fluency', 'overall', 'score']):
                            score_columns.append(col)
                    
                    if score_columns:
                        st.info(f"감지된 점수 컬럼: {', '.join(score_columns)}")
                        
                        # 참고 데이터 미리보기
                        with st.expander("참고 데이터 미리보기"):
                            st.dataframe(ref_df.head(5), use_container_width=True)
                        
                        # 세션 상태에 저장
                        st.session_state.reference_data = {
                            'file_path': ref_file_path,
                            'dataframe': ref_df,
                            'score_columns': score_columns
                        }
                        
                        st.success("참고 데이터가 성공적으로 설정되었습니다. AI 평가 시 이 데이터를 참고합니다.")
                    else:
                        st.warning("점수 컬럼을 찾을 수 없습니다. Accuracy, Omission, Compliance, Fluency, Overall 등의 컬럼이 있는지 확인해주세요.")
                        if st.button("참고 데이터 제거"):
                            if 'reference_data' in st.session_state:
                                del st.session_state.reference_data
                            st.rerun()
                else:
                    st.error("참고 데이터를 로드할 수 없습니다.")
                    
            except Exception as e:
                st.error(f"참고 데이터 로드 중 오류: {e}")
        else:
            # 참고 데이터 제거 옵션
            if 'reference_data' in st.session_state:
                if st.button("참고 데이터 제거"):
                    del st.session_state.reference_data
                    st.success("참고 데이터가 제거되었습니다.")
                    st.rerun()
                else:
                    st.info("현재 참고 데이터가 설정되어 있습니다.")
        
        st.markdown("---")
        
        # 설정 가져오기
        st.subheader("설정 가져오기")
        st.markdown("이전에 내보낸 설정 파일을 업로드하여 메트릭과 스케일 정의를 복원할 수 있습니다.")
        
        uploaded_settings = st.file_uploader(
            "설정 파일 업로드 (JSON)",
            type=['json'],
            help="MT-Eval Pro에서 내보낸 설정 파일을 선택하세요.",
            key="settings_uploader"
        )
        
        if uploaded_settings is not None:
            try:
                import json
                settings_data = json.loads(uploaded_settings.read().decode('utf-8'))
                
                # 설정 파일 유효성 검사
                if 'metrics' in settings_data and 'scale' in settings_data:
                    # 메트릭 검증
                    required_metrics = ['Accuracy', 'Omission/Addition', 'Compliance', 'Fluency', 'Overall']
                    if all(metric in settings_data['metrics'] for metric in required_metrics):
                        # 스케일 검증
                        required_scales = [1, 2, 3, 4, 5]
                        if all(scale in settings_data['scale'] for scale in required_scales):
                            st.session_state.custom_metrics = settings_data['metrics']
                            st.session_state.custom_scale = settings_data['scale']
                            
                            # 성공 메시지와 미리보기
                            st.success("설정이 성공적으로 가져와졌습니다!")
                            
                            with st.expander("가져온 설정 미리보기"):
                                st.write("**메트릭 정의:**")
                                for metric, definition in settings_data['metrics'].items():
                                    st.write(f"• {metric}: {definition[:100]}...")
                                
                                st.write("**스케일 정의:**")
                                for score, definition in settings_data['scale'].items():
                                    st.write(f"• {score}점: {definition[:80]}...")
                            
                            st.rerun()
                        else:
                            st.error("설정 파일에 필수 스케일 정의(1-5점)가 누락되었습니다.")
                    else:
                        st.error("설정 파일에 필수 메트릭 정의가 누락되었습니다.")
                else:
                    st.error("올바르지 않은 설정 파일 형식입니다. 'metrics'와 'scale' 섹션이 필요합니다.")
                    
            except json.JSONDecodeError:
                st.error("유효하지 않은 JSON 파일입니다.")
            except Exception as e:
                st.error(f"설정 파일 로드 중 오류: {e}")
        
        # 현재 설정 미리보기
        if st.checkbox("현재 설정 미리보기"):
            st.subheader("현재 메트릭 설정")
            for metric, definition in st.session_state.custom_metrics.items():
                st.markdown(f"**{metric}**: {definition}")
            
            st.subheader("현재 스케일 설정")
            for score, definition in st.session_state.custom_scale.items():
                st.markdown(f"**{score}점**: {definition}")
        
        # 시스템 정보
        st.subheader("시스템 정보")
        
        with st.expander("지원 언어"):
            lang_df = pd.DataFrame([
                {'언어 코드': code, '언어명': name}
                for code, name in self.config.SUPPORTED_LANGUAGES.items()
            ])
            st.dataframe(lang_df, use_container_width=True)
        
        with st.expander("사용법 안내"):
            st.markdown("""
            **메트릭 설정 사용법:**
            1. 위의 메트릭 정의와 스케일 정의를 원하는 대로 수정
            2. "설정 저장" 버튼을 클릭하여 변경사항 저장
            3. "평가 실행" 탭에서 번역 평가 시 자동으로 사용자 설정 적용
            
            **설정 공유:**
            - "설정 내보내기" 버튼으로 현재 설정을 JSON 파일로 다운로드
            - 다른 시스템에서 "설정 가져오기"로 동일한 설정 사용 가능
            
            **주의사항:**
            - 메트릭 이름(Accuracy, Omission/Addition 등)은 변경하지 마세요
            - 스케일은 반드시 1-5점 체계를 유지해야 합니다
            """)
    
    def save_custom_settings(self):
        """사용자 설정을 파일로 저장"""
        try:
            settings_data = {
                'metrics': st.session_state.custom_metrics,
                'scale': st.session_state.custom_scale,
                'timestamp': datetime.now().isoformat()
            }
            
            # 설정 디렉토리 생성
            settings_dir = "user_settings"
            if not os.path.exists(settings_dir):
                os.makedirs(settings_dir)
            
            # 설정 파일 저장
            settings_path = os.path.join(settings_dir, "custom_metrics.json")
            with open(settings_path, 'w', encoding='utf-8') as f:
                json.dump(settings_data, f, ensure_ascii=False, indent=2)
            
            return settings_path
        except Exception as e:
            st.error(f"설정 저장 중 오류: {e}")
            return None
    
    def export_settings(self):
        """현재 설정을 다운로드 가능한 형태로 내보내기"""
        try:
            settings_data = {
                'metrics': st.session_state.custom_metrics,
                'scale': st.session_state.custom_scale,
                'exported_at': datetime.now().isoformat(),
                'version': '1.0'
            }
            
            settings_json = json.dumps(settings_data, ensure_ascii=False, indent=2)
            
            st.download_button(
                label="설정 파일 다운로드",
                data=settings_json,
                file_name=f"mt_eval_settings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                key="download_settings_file"
            )
        except Exception as e:
            st.error(f"설정 내보내기 중 오류: {e}")
    

    
    def get_quality_level(self, score):
        """점수를 품질 수준으로 변환"""
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


def main():
    """메인 함수"""
    app = MTEvalProApp()
    app.main()


if __name__ == "__main__":
    main() 