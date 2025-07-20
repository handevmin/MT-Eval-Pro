import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from config import Config

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class VisualizationGenerator:
    """평가 결과 시각화 및 보고서 생성 클래스"""
    
    def __init__(self):
        self.config = Config()
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 색상 팔레트 설정
        self.colors = {
            'primary': '#2E86C1',
            'secondary': '#F39C12',
            'success': '#27AE60',
            'warning': '#F1C40F',
            'danger': '#E74C3C',
            'info': '#3498DB'
        }
        
        # 품질 수준별 색상
        self.quality_colors = {
            'Very Good': '#27AE60',
            'Good': '#F39C12',
            'Acceptable': '#F1C40F',
            'Poor': '#E67E22',
            'Very Poor': '#E74C3C'
        }
    
    def generate_all_visualizations(self, evaluation_results: Dict, 
                                  output_dir: str = None) -> Dict[str, str]:
        """모든 시각화를 생성하고 파일 경로를 반환합니다."""
        
        if output_dir is None:
            output_dir = self.config.REPORTS_DIR
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        visualization_files = {}
        
        try:
            # 1. 전체 점수 분포 차트
            self.logger.info("전체 점수 분포 차트 생성 중...")
            file_path = self.create_score_distribution_chart(
                evaluation_results, 
                os.path.join(output_dir, "score_distribution.png")
            )
            if file_path:
                visualization_files['score_distribution'] = file_path
            
            # 2. 언어별 비교 차트
            self.logger.info("언어별 비교 차트 생성 중...")
            file_path = self.create_language_comparison_chart(
                evaluation_results,
                os.path.join(output_dir, "language_comparison.png")
            )
            if file_path:
                visualization_files['language_comparison'] = file_path
            
            # 3. 메트릭별 분석 차트
            self.logger.info("메트릭별 분석 차트 생성 중...")
            file_path = self.create_metrics_analysis_chart(
                evaluation_results,
                os.path.join(output_dir, "metrics_analysis.png")
            )
            if file_path:
                visualization_files['metrics_analysis'] = file_path
            
            # 4. 품질 분포 파이 차트
            self.logger.info("품질 분포 파이 차트 생성 중...")
            file_path = self.create_quality_distribution_pie(
                evaluation_results,
                os.path.join(output_dir, "quality_distribution.png")
            )
            if file_path:
                visualization_files['quality_distribution'] = file_path
            
            # 5. 인터랙티브 대시보드 (HTML)
            self.logger.info("인터랙티브 대시보드 생성 중...")
            file_path = self.create_interactive_dashboard(
                evaluation_results,
                os.path.join(output_dir, "interactive_dashboard.html")
            )
            if file_path:
                visualization_files['interactive_dashboard'] = file_path
            
            self.logger.info(f"총 {len(visualization_files)}개의 시각화 파일 생성 완료")
            
        except Exception as e:
            self.logger.error(f"시각화 생성 중 오류: {e}")
        
        return visualization_files
    
    def create_score_distribution_chart(self, evaluation_results: Dict, 
                                      output_path: str) -> Optional[str]:
        """전체 점수 분포 차트를 생성합니다."""
        try:
            if 'aggregate_scores' not in evaluation_results or 'overall' not in evaluation_results['aggregate_scores']:
                return None
            
            metrics_data = evaluation_results['aggregate_scores']['overall']
            metrics = list(metrics_data.keys())
            scores = [metrics_data[metric]['mean'] for metric in metrics]
            
            # 한글 메트릭명 매핑
            metric_names_kr = {
                'Accuracy': '정확성',
                'Omission/Addition': '누락/추가',
                'Compliance': '준수성',
                'Fluency': '유창성',
                'Overall': '전체'
            }
            
            metric_labels = [metric_names_kr.get(m, m) for m in metrics]
            
            plt.figure(figsize=(12, 8))
            bars = plt.bar(metric_labels, scores, color=[self.colors['primary'], self.colors['secondary'], 
                                                        self.colors['success'], self.colors['warning'], 
                                                        self.colors['info']])
            
            plt.title('메트릭별 평균 점수', fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('평가 메트릭', fontsize=12)
            plt.ylabel('평균 점수', fontsize=12)
            plt.ylim(0, 5)
            
            # 점수 라벨 추가
            for bar, score in zip(bars, scores):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{score:.2f}', ha='center', va='bottom', fontweight='bold')
            
            # 그리드 추가
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"점수 분포 차트 생성 오류: {e}")
            return None
    
    def create_language_comparison_chart(self, evaluation_results: Dict, 
                                       output_path: str) -> Optional[str]:
        """언어별 비교 차트를 생성합니다."""
        try:
            if 'detailed_results' not in evaluation_results or 'language_comparison' not in evaluation_results['detailed_results']:
                return None
            
            lang_data = evaluation_results['detailed_results']['language_comparison']
            
            languages = []
            scores = []
            quality_levels = []
            
            for lang_code, data in lang_data.items():
                languages.append(data['language_name'])
                scores.append(data['average_overall_score'])
                quality_levels.append(data['quality_level'])
            
            # 점수순으로 정렬
            sorted_data = sorted(zip(languages, scores, quality_levels), key=lambda x: x[1], reverse=True)
            languages, scores, quality_levels = zip(*sorted_data)
            
            plt.figure(figsize=(12, 8))
            bars = plt.barh(languages, scores, color=[self.colors['primary']] * len(languages))
            
            plt.title('언어별 평균 전체 점수 비교', fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('평균 전체 점수', fontsize=12)
            plt.ylabel('언어', fontsize=12)
            plt.xlim(0, 5)
            
            # 점수 라벨 및 품질 수준 추가
            for i, (bar, score, quality) in enumerate(zip(bars, scores, quality_levels)):
                plt.text(score + 0.1, bar.get_y() + bar.get_height()/2,
                        f'{score:.2f} ({quality})', va='center', fontweight='bold')
            
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"언어별 비교 차트 생성 오류: {e}")
            return None
    
    def create_metrics_analysis_chart(self, evaluation_results: Dict, 
                                    output_path: str) -> Optional[str]:
        """메트릭별 상세 분석 차트를 생성합니다."""
        try:
            if 'evaluation_results' not in evaluation_results:
                return None
            
            # 데이터 준비
            metrics_data = {}
            for result in evaluation_results['evaluation_results']:
                if 'evaluation' in result:
                    for metric, data in result['evaluation'].items():
                        if metric not in metrics_data:
                            metrics_data[metric] = []
                        metrics_data[metric].append(data.get('score', 0))
            
            if not metrics_data:
                return None
            
            # 박스플롯 생성
            fig, ax = plt.subplots(figsize=(14, 8))
            
            metric_names_kr = {
                'Accuracy': '정확성',
                'Omission/Addition': '누락/추가',
                'Compliance': '준수성',
                'Fluency': '유창성',
                'Overall': '전체'
            }
            
            box_data = []
            labels = []
            
            for metric in ['Accuracy', 'Omission/Addition', 'Compliance', 'Fluency', 'Overall']:
                if metric in metrics_data:
                    box_data.append(metrics_data[metric])
                    labels.append(metric_names_kr.get(metric, metric))
            
            bp = ax.boxplot(box_data, labels=labels, patch_artist=True, notch=True)
            
            # 색상 설정
            colors = [self.colors['primary'], self.colors['secondary'], self.colors['success'], 
                     self.colors['warning'], self.colors['info']]
            for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_title('메트릭별 점수 분포 분석', fontsize=16, fontweight='bold', pad=20)
            ax.set_ylabel('점수', fontsize=12)
            ax.set_ylim(0, 5.5)
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"메트릭별 분석 차트 생성 오류: {e}")
            return None
    
    def create_quality_distribution_pie(self, evaluation_results: Dict, 
                                      output_path: str) -> Optional[str]:
        """품질 분포 파이 차트를 생성합니다."""
        try:
            if ('detailed_results' not in evaluation_results or 
                'quality_distribution' not in evaluation_results['detailed_results'] or
                'Overall' not in evaluation_results['detailed_results']['quality_distribution']):
                return None
            
            quality_dist = evaluation_results['detailed_results']['quality_distribution']['Overall']
            
            # 데이터 준비
            labels = []
            sizes = []
            colors = []
            
            quality_mapping = {
                'excellent': ('우수 (4.5-5점)', self.quality_colors['Very Good']),
                'good': ('양호 (3.5-4.5점)', self.quality_colors['Good']),
                'acceptable': ('보통 (2.5-3.5점)', self.quality_colors['Acceptable']),
                'poor': ('미흡 (1.5-2.5점)', self.quality_colors['Poor']),
                'very_poor': ('매우 미흡 (1-1.5점)', self.quality_colors['Very Poor'])
            }
            
            for key, (label, color) in quality_mapping.items():
                if quality_dist.get(key, 0) > 0:
                    labels.append(label)
                    sizes.append(quality_dist[key])
                    colors.append(color)
            
            if not sizes:
                return None
            
            plt.figure(figsize=(10, 8))
            wedges, texts, autotexts = plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                              startangle=90, textprops={'fontsize': 10})
            
            plt.title('전체 품질 분포', fontsize=16, fontweight='bold', pad=20)
            
            # 범례 추가
            plt.legend(wedges, labels, title="품질 수준", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"품질 분포 파이 차트 생성 오류: {e}")
            return None
    
    def create_interactive_dashboard(self, evaluation_results: Dict, 
                                   output_path: str) -> Optional[str]:
        """인터랙티브 대시보드를 생성합니다."""
        try:
            # 서브플롯 생성
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('메트릭별 평균 점수', '언어별 비교', '점수 분포', '품질 수준 분포'),
                specs=[[{"type": "bar"}, {"type": "bar"}],
                       [{"type": "box"}, {"type": "pie"}]]
            )
            
            # 1. 메트릭별 평균 점수
            if 'aggregate_scores' in evaluation_results and 'overall' in evaluation_results['aggregate_scores']:
                metrics_data = evaluation_results['aggregate_scores']['overall']
                metrics = list(metrics_data.keys())
                scores = [metrics_data[metric]['mean'] for metric in metrics]
                
                fig.add_trace(
                    go.Bar(x=metrics, y=scores, name='평균 점수', 
                          marker_color=self.colors['primary']),
                    row=1, col=1
                )
            
            # 2. 언어별 비교
            if ('detailed_results' in evaluation_results and 
                'language_comparison' in evaluation_results['detailed_results']):
                
                lang_data = evaluation_results['detailed_results']['language_comparison']
                languages = [data['language_name'] for data in lang_data.values()]
                lang_scores = [data['average_overall_score'] for data in lang_data.values()]
                
                fig.add_trace(
                    go.Bar(x=languages, y=lang_scores, name='언어별 점수',
                          marker_color=self.colors['secondary']),
                    row=1, col=2
                )
            
            # 3. 점수 분포 (박스플롯)
            if 'evaluation_results' in evaluation_results:
                for i, metric in enumerate(['Accuracy', 'Fluency', 'Overall']):
                    scores = []
                    for result in evaluation_results['evaluation_results']:
                        if ('evaluation' in result and metric in result['evaluation'] and
                            'score' in result['evaluation'][metric]):
                            scores.append(result['evaluation'][metric]['score'])
                    
                    if scores:
                        fig.add_trace(
                            go.Box(y=scores, name=metric, boxpoints='outliers'),
                            row=2, col=1
                        )
            
            # 4. 품질 수준 분포 (파이 차트)
            if ('detailed_results' in evaluation_results and 
                'quality_distribution' in evaluation_results['detailed_results'] and
                'Overall' in evaluation_results['detailed_results']['quality_distribution']):
                
                quality_dist = evaluation_results['detailed_results']['quality_distribution']['Overall']
                labels = ['우수', '양호', '보통', '미흡', '매우 미흡']
                values = [quality_dist.get(key, 0) for key in ['excellent', 'good', 'acceptable', 'poor', 'very_poor']]
                
                fig.add_trace(
                    go.Pie(labels=labels, values=values, name="품질 분포"),
                    row=2, col=2
                )
            
            # 레이아웃 설정
            fig.update_layout(
                title_text="MT-Eval Pro 평가 결과 대시보드",
                title_x=0.5,
                showlegend=False,
                height=800
            )
            
            # HTML 파일로 저장
            fig.write_html(output_path)
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"인터랙티브 대시보드 생성 오류: {e}")
            return None
    
    def generate_summary_report(self, evaluation_results: Dict, 
                              output_path: str) -> Optional[str]:
        """요약 보고서를 생성합니다."""
        try:
            metadata = evaluation_results.get('metadata', {})
            aggregate_scores = evaluation_results.get('aggregate_scores', {})
            detailed_results = evaluation_results.get('detailed_results', {})
            
            html_content = f"""
            <!DOCTYPE html>
            <html lang="ko">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>MT-Eval Pro 평가 보고서</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                    .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 30px; }}
                    .section {{ margin-bottom: 30px; }}
                    .metric-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                    .metric-table th, .metric-table td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                    .metric-table th {{ background-color: #f2f2f2; }}
                    .score-excellent {{ color: #27AE60; font-weight: bold; }}
                    .score-good {{ color: #F39C12; font-weight: bold; }}
                    .score-acceptable {{ color: #F1C40F; font-weight: bold; }}
                    .score-poor {{ color: #E67E22; font-weight: bold; }}
                    .score-very-poor {{ color: #E74C3C; font-weight: bold; }}
                    .recommendations {{ background-color: #e8f4fd; padding: 15px; border-radius: 5px; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>MT-Eval Pro 번역 품질 평가 보고서</h1>
                    <p><strong>평가 일시:</strong> {metadata.get('evaluation_timestamp', 'N/A')}</p>
                    <p><strong>사용 모델:</strong> {metadata.get('model_used', 'N/A')}</p>
                    <p><strong>총 평가 쌍:</strong> {metadata.get('evaluated_pairs', 'N/A')}개</p>
                </div>
            """
            
            # 전체 점수 요약
            if 'overall' in aggregate_scores:
                html_content += """
                <div class="section">
                    <h2>전체 점수 요약</h2>
                    <table class="metric-table">
                        <tr><th>메트릭</th><th>평균 점수</th><th>최소</th><th>최대</th><th>평가 개수</th></tr>
                """
                
                for metric, stats in aggregate_scores['overall'].items():
                    score = stats['mean']
                    score_class = self._get_score_class(score)
                    html_content += f"""
                        <tr>
                            <td>{metric}</td>
                            <td class="{score_class}">{score:.2f}</td>
                            <td>{stats['min']}</td>
                            <td>{stats['max']}</td>
                            <td>{stats['count']}</td>
                        </tr>
                    """
                
                html_content += "</table></div>"
            
            # 언어별 비교
            if 'language_comparison' in detailed_results:
                html_content += """
                <div class="section">
                    <h2>언어별 비교</h2>
                    <table class="metric-table">
                        <tr><th>언어</th><th>평균 전체 점수</th><th>품질 수준</th><th>평가 개수</th></tr>
                """
                
                for lang_code, data in detailed_results['language_comparison'].items():
                    score = data['average_overall_score']
                    score_class = self._get_score_class(score)
                    html_content += f"""
                        <tr>
                            <td>{data['language_name']}</td>
                            <td class="{score_class}">{score:.2f}</td>
                            <td>{data['quality_level']}</td>
                            <td>{data['evaluation_count']}</td>
                        </tr>
                    """
                
                html_content += "</table></div>"
            
            # 추천사항
            if 'recommendations' in detailed_results:
                html_content += """
                <div class="section">
                    <h2>개선 추천사항</h2>
                    <div class="recommendations">
                        <ul>
                """
                
                for recommendation in detailed_results['recommendations']:
                    html_content += f"<li>{recommendation}</li>"
                
                html_content += "</ul></div></div>"
            
            html_content += """
            </body>
            </html>
            """
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"요약 보고서 생성 오류: {e}")
            return None
    
    def _get_score_class(self, score: float) -> str:
        """점수에 따른 CSS 클래스를 반환합니다."""
        if score >= 4.5:
            return "score-excellent"
        elif score >= 3.5:
            return "score-good"
        elif score >= 2.5:
            return "score-acceptable"
        elif score >= 1.5:
            return "score-poor"
        else:
            return "score-very-poor"


def main():
    """테스트 함수"""
    visualizer = VisualizationGenerator()
    
    # 더미 데이터로 테스트
    dummy_results = {
        'metadata': {
            'evaluation_timestamp': '2024-01-01T12:00:00',
            'model_used': 'gpt-4.1',
            'evaluated_pairs': 100
        },
        'aggregate_scores': {
            'overall': {
                'Accuracy': {'mean': 4.2, 'min': 2, 'max': 5, 'count': 100},
                'Fluency': {'mean': 3.8, 'min': 1, 'max': 5, 'count': 100},
                'Overall': {'mean': 4.0, 'min': 2, 'max': 5, 'count': 100}
            }
        },
        'detailed_results': {
            'language_comparison': {
                'ko-kr': {'language_name': '한국어', 'average_overall_score': 4.1, 'quality_level': '양호', 'evaluation_count': 50},
                'ja-jp': {'language_name': '일본어', 'average_overall_score': 3.9, 'quality_level': '양호', 'evaluation_count': 50}
            },
            'quality_distribution': {
                'Overall': {'excellent': 30, 'good': 40, 'acceptable': 20, 'poor': 8, 'very_poor': 2}
            },
            'recommendations': ['전체적으로 양호한 번역 품질을 보이고 있습니다.']
        },
        'evaluation_results': []
    }
    
    print("=== 시각화 테스트 ===")
    files = visualizer.generate_all_visualizations(dummy_results, "test_reports")
    
    for viz_type, file_path in files.items():
        print(f"{viz_type}: {file_path}")


if __name__ == "__main__":
    main() 