#!/usr/bin/env python3
"""
MT-Eval Pro - 기계 번역 품질 자동 평가 시스템
명령줄 인터페이스

사용법:
    python main.py --help                    # 도움말 표시
    python main.py evaluate                  # 기본 파일로 평가 실행
    python main.py evaluate --file data.xlsx # 특정 파일로 평가 실행
    python main.py web                       # 웹 인터페이스 실행
"""

import argparse
import sys
import os
from pathlib import Path
import logging
from evaluation_engine import EvaluationEngine
from visualization import VisualizationGenerator
from config import Config

def setup_logging(verbose=False):
    """로깅 설정"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('mt_eval_pro.log')
        ]
    )

def check_requirements():
    """필수 요구사항 확인"""
    config = Config()
    
    print("시스템 요구사항 확인 중...")
    
    # API 키 확인
    if not config.OPENAI_API_KEY:
        print("ERROR: OpenAI API 키가 설정되지 않았습니다.")
        print("   다음 중 하나의 방법으로 API 키를 설정해주세요:")
        print("   1. .env 파일에 OPENAI_API_KEY=your_key_here 추가")
        print("   2. 환경변수 설정: export OPENAI_API_KEY=your_key_here")
        print("   3. 웹 인터페이스에서 직접 입력")
        return False
    
    # 기본 파일 확인
    if not os.path.exists(config.SAMPLE_FILE):
        print(f"WARNING: 기본 번역 파일을 찾을 수 없습니다: {config.SAMPLE_FILE}")
        print("   웹 인터페이스를 사용하거나 --file 옵션으로 파일을 지정해주세요.")
    
    if not os.path.exists(config.METRICS_FILE):
        print(f"WARNING: 기본 메트릭스 파일을 찾을 수 없습니다: {config.METRICS_FILE}")
    
    print("요구사항 확인 완료")
    return True

def evaluate_command(args):
    """평가 명령 실행"""
    print("MT-Eval Pro 평가 시작")
    print("=" * 50)
    
    try:
        # 평가 엔진 초기화
        engine = EvaluationEngine()
        
        # 평가 실행
        results = engine.run_full_evaluation(
            translation_file=args.file,
            metrics_file=args.metrics,
            save_results=True
        )
        
        print("\n평가 결과 요약")
        print("-" * 30)
        
        # 메타데이터 출력
        metadata = results.get('metadata', {})
        print(f"평가 시간: {metadata.get('evaluation_timestamp', 'N/A')}")
        print(f"총 평가 쌍: {metadata.get('evaluated_pairs', 0)}개")
        print(f"사용 모델: {metadata.get('model_used', 'N/A')}")
        
        # 전체 점수 출력
        if 'aggregate_scores' in results and 'overall' in results['aggregate_scores']:
            print("\n메트릭별 평균 점수:")
            for metric, data in results['aggregate_scores']['overall'].items():
                print(f"  {metric}: {data['mean']:.2f}")
        
        # 언어별 결과 출력
        if ('detailed_results' in results and 
            'language_comparison' in results['detailed_results']):
            print("\n언어별 결과:")
            for lang_code, data in results['detailed_results']['language_comparison'].items():
                print(f"  {data['language_name']}: {data['average_overall_score']:.2f} ({data['quality_level']})")
        
        # 추천사항 출력
        if ('detailed_results' in results and 
            'recommendations' in results['detailed_results']):
            print("\n추천사항:")
            for i, rec in enumerate(results['detailed_results']['recommendations'], 1):
                print(f"  {i}. {rec}")
        
        # 시각화 생성 (옵션)
        if args.visualize:
            print("\n시각화 생성 중...")
            visualizer = VisualizationGenerator()
            viz_files = visualizer.generate_all_visualizations(results)
            
            print("생성된 시각화 파일:")
            for viz_type, file_path in viz_files.items():
                print(f"  {viz_type}: {file_path}")
        
        # 업데이트된 원본 파일 정보 출력
        if 'updated_file_path' in metadata:
            print(f"\n원본 파일 업데이트 완료!")
            print(f"점수가 채워진 파일: {metadata['updated_file_path']}")
        
        print("\n평가 완료!")
        print(f"결과 파일은 '{Config().RESULTS_DIR}' 디렉토리에 저장되었습니다.")
        
    except Exception as e:
        print(f"ERROR: 평가 중 오류 발생: {e}")
        logging.error(f"평가 오류: {e}", exc_info=True)
        sys.exit(1)

def web_command(args):
    """웹 인터페이스 명령 실행"""
    print("웹 인터페이스 시작 중...")
    
    try:
        import streamlit.web.cli as stcli
        sys.argv = ["streamlit", "run", "app.py", "--server.port", str(args.port)]
        stcli.main()
    except ImportError:
        print("ERROR: Streamlit이 설치되지 않았습니다.")
        print("   pip install streamlit 명령으로 설치해주세요.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: 웹 인터페이스 실행 오류: {e}")
        sys.exit(1)

def test_command(args):
    """테스트 명령 실행"""
    print("시스템 테스트 실행")
    print("=" * 30)
    
    try:
        from data_processor import DataProcessor
        from llm_evaluator import LLMEvaluator
        
        # 데이터 처리기 테스트
        print("1. 데이터 처리기 테스트...")
        processor = DataProcessor()
        
        # 기본 파일 로드 테스트
        try:
            df = processor.load_translation_data()
            if not df.empty:
                print(f"   번역 데이터 로드 성공: {df.shape}")
                
                pairs = processor.extract_translation_pairs(df)
                print(f"   번역 쌍 추출 성공: {len(pairs)}개")
            else:
                print("   WARNING: 번역 데이터가 비어있습니다.")
        except Exception as e:
            print(f"   ERROR: 데이터 로드 오류: {e}")
        
        # LLM 평가기 테스트
        print("\n2. LLM 평가기 테스트...")
        try:
            evaluator = LLMEvaluator()
            
            # 간단한 테스트 번역
            test_result = evaluator.evaluate_translation(
                source_text="Hello, world!",
                target_text="안녕하세요, 세계!",
                target_language="ko-kr"
            )
            
            if 'evaluation' in test_result:
                print("   LLM 평가 성공")
                for metric, data in test_result['evaluation'].items():
                    print(f"      {metric}: {data['score']}점")
            else:
                print("   LLM 평가 실패")
                
        except Exception as e:
            print(f"   ERROR: LLM 평가 오류: {e}")
        
        print("\n테스트 완료")
        
    except Exception as e:
        print(f"ERROR: 테스트 중 오류 발생: {e}")
        logging.error(f"테스트 오류: {e}", exc_info=True)

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="MT-Eval Pro - 기계 번역 품질 자동 평가 시스템",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python main.py evaluate                          # 기본 설정으로 평가 실행
  python main.py evaluate --file my_data.xlsx     # 특정 파일로 평가 실행
  python main.py evaluate --visualize             # 시각화 포함 평가
  python main.py web                              # 웹 인터페이스 실행
  python main.py test                             # 시스템 테스트
  python main.py check                            # 요구사항 확인
        """
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='상세 로그 출력'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='사용 가능한 명령')
    
    # evaluate 명령
    eval_parser = subparsers.add_parser('evaluate', help='번역 품질 평가 실행')
    eval_parser.add_argument(
        '--file', '-f',
        type=str,
        help='번역 데이터 파일 경로 (Excel 형식)'
    )
    eval_parser.add_argument(
        '--metrics', '-m',
        type=str,
        help='메트릭스 정의 파일 경로 (Excel 형식)'
    )
    eval_parser.add_argument(
        '--visualize',
        action='store_true',
        help='시각화 파일 생성'
    )
    
    # web 명령
    web_parser = subparsers.add_parser('web', help='웹 인터페이스 실행')
    web_parser.add_argument(
        '--port',
        type=int,
        default=8501,
        help='웹 서버 포트 (기본값: 8501)'
    )
    
    # test 명령
    test_parser = subparsers.add_parser('test', help='시스템 테스트 실행')
    
    # check 명령
    check_parser = subparsers.add_parser('check', help='시스템 요구사항 확인')
    
    args = parser.parse_args()
    
    # 로깅 설정
    setup_logging(args.verbose)
    
    # 명령 실행
    if args.command == 'evaluate':
        if check_requirements():
            evaluate_command(args)
    elif args.command == 'web':
        web_command(args)
    elif args.command == 'test':
        test_command(args)
    elif args.command == 'check':
        check_requirements()
    else:
        # 기본 동작: 웹 인터페이스 실행
        print("기본 웹 인터페이스를 실행합니다.")
        print("   다른 명령을 사용하려면 --help 옵션을 확인하세요.")
        args.port = 8501
        web_command(args)

if __name__ == "__main__":
    main() 