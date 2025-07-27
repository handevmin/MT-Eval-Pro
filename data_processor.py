import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from config import Config

class DataProcessor:
    """Excel 데이터 처리 및 관리 클래스"""
    
    def __init__(self):
        self.config = Config()
        self.translation_data = None
        self.metrics_data = None
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_metrics_definitions(self, file_path: str = None) -> Dict:
        """메트릭스 정의 파일을 로드합니다."""
        if file_path is None:
            file_path = self.config.METRICS_FILE
        
        try:
            # Excel 파일의 모든 시트 확인
            excel_file = pd.ExcelFile(file_path)
            self.logger.info(f"사용 가능한 시트: {excel_file.sheet_names}")
            
            # 첫 번째 시트 또는 'Metrics' 시트 로드
            if 'Metrics' in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name='Metrics')
            else:
                df = pd.read_excel(file_path, sheet_name=0)
            
            self.metrics_data = df
            self.logger.info(f"메트릭스 데이터 로드 완료: {df.shape}")
            return df.to_dict('records')
            
        except Exception as e:
            self.logger.error(f"메트릭스 파일 로드 오류: {e}")
            return {}
    
    def load_translation_data(self, file_path: str = None) -> pd.DataFrame:
        """번역 데이터 파일을 로드합니다."""
        if file_path is None:
            file_path = self.config.SAMPLE_FILE
        
        try:
            # 파일 확장자 확인
            if file_path.lower().endswith('.csv'):
                # CSV 파일 로드
                df = pd.read_csv(file_path)
                self.logger.info(f"CSV 파일 로드: {file_path}")
            else:
                # Excel 파일 로드
                excel_file = pd.ExcelFile(file_path)
                self.logger.info(f"사용 가능한 시트: {excel_file.sheet_names}")
                
                # 첫 번째 시트 로드
                df = pd.read_excel(file_path, sheet_name=0)
            
            # 컬럼 정보 출력
            self.logger.info(f"컬럼: {list(df.columns)}")
            self.logger.info(f"데이터 형태: {df.shape}")
            
            self.translation_data = df
            return df
            
        except Exception as e:
            self.logger.error(f"번역 데이터 파일 로드 오류: {e}")
            return pd.DataFrame()
    
    def extract_translation_pairs(self, df: pd.DataFrame = None, source_column: str = None, target_column: str = None, language_code: str = None) -> List[Dict]:
        """번역 쌍을 추출합니다."""
        if df is None:
            df = self.translation_data
        
        if df is None or df.empty:
            self.logger.warning("번역 데이터가 없습니다.")
            return []
        
        translation_pairs = []
        columns = list(df.columns)
        
        # 사용자가 지정한 컬럼이 있으면 사용
        if source_column and target_column and language_code:
            if source_column in columns and target_column in columns:
                source_col = source_column
                target_cols = [target_column]
                self.logger.info(f"사용자 지정 컬럼 사용: {source_col} -> {target_column} ({language_code})")
            else:
                self.logger.error(f"지정된 컬럼을 찾을 수 없습니다: {source_column}, {target_column}")
                return []
        else:
            # 자동 감지
            source_col = None
            target_cols = []
            
            for col in columns:
                col_lower = str(col).lower()
                if any(keyword in col_lower for keyword in ['source', 'english', 'en', 'original']):
                    source_col = col
                elif any(lang_code in col_lower for lang_code in self.config.SUPPORTED_LANGUAGES.keys()):
                    target_cols.append(col)
            
            # 소스 컬럼이 없으면 첫 번째 컬럼을 소스로 가정
            if source_col is None and len(columns) > 0:
                source_col = columns[0]
                
            # 타겟 컬럼이 없으면 나머지 컬럼들을 확인
            if not target_cols:
                # 소스가 아닌 텍스트 컬럼들을 찾기
                for col in columns:
                    if col != source_col and col not in ['OVERALL', 'ACCURACY', 'OMISSION/ADDITION', 'COMPLIANCE', 'FLUENCY']:
                        target_cols.append(col)
        
        # 번역 쌍 생성
        for index, row in df.iterrows():
            if pd.isna(row[source_col]) or str(row[source_col]).strip() == '':
                continue
                
            source_text = str(row[source_col]).strip()
            
            for target_col in target_cols:
                if pd.isna(row[target_col]) or str(row[target_col]).strip() == '':
                    continue
                    
                target_text = str(row[target_col]).strip()
                
                # 언어 코드 추출
                if language_code:
                    # 사용자 지정 언어 코드 사용
                    lang_code = language_code
                else:
                    # 자동 추출
                    lang_code = self._extract_language_code(target_col)
                
                translation_pair = {
                    'id': f"{index}_{lang_code}",
                    'source_text': source_text,
                    'target_text': target_text,
                    'language_code': lang_code,
                    'language_name': self.config.SUPPORTED_LANGUAGES.get(lang_code, lang_code),
                    'row_index': index,
                    'source_column': source_col,
                    'target_column': target_col
                }
                
                translation_pairs.append(translation_pair)
        
        self.logger.info(f"추출된 번역 쌍: {len(translation_pairs)}개")
        return translation_pairs
    
    def get_available_columns(self, df: pd.DataFrame = None) -> Dict:
        """사용 가능한 컬럼들을 분석하여 반환합니다."""
        if df is None:
            df = self.translation_data
        
        if df is None or df.empty:
            return {}
        
        columns = list(df.columns)
        
        # 컬럼 분류
        text_columns = []
        score_columns = []
        
        for col in columns:
            col_lower = str(col).lower()
            if col in ['OVERALL', 'ACCURACY', 'OMISSION/ADDITION', 'COMPLIANCE', 'FLUENCY']:
                score_columns.append(col)
            else:
                # 실제 텍스트 데이터가 있는지 확인
                sample_data = df[col].dropna().astype(str)
                if len(sample_data) > 0:
                    # 첫 번째 샘플이 숫자가 아닌 텍스트인지 확인
                    first_sample = sample_data.iloc[0].strip()
                    if len(first_sample) > 10 and not first_sample.replace('.', '').replace(',', '').isdigit():
                        text_columns.append(col)
        
        # 소스 컬럼 추천
        source_candidates = []
        target_candidates = []
        
        for col in text_columns:
            col_lower = str(col).lower()
            if any(keyword in col_lower for keyword in ['source', 'english', 'en', 'original']):
                source_candidates.append(col)
            else:
                target_candidates.append(col)
        
        return {
            'all_columns': columns,
            'text_columns': text_columns,
            'score_columns': score_columns,
            'source_candidates': source_candidates,
            'target_candidates': target_candidates
        }

    def detect_language_from_column_name(self, column_name: str) -> str:
        """칼럼명에서 언어 코드를 자동으로 감지합니다."""
        col_upper = str(column_name).upper()
        
        # 직접적인 언어 코드 매핑
        language_mappings = {
            'JA': 'ja-jp',
            'JP': 'ja-jp', 
            'JAPANESE': 'ja-jp',
            'KO': 'ko-kr',
            'KR': 'ko-kr',
            'KOREAN': 'ko-kr',
            'ZH': 'zh-cn',
            'CN': 'zh-cn',
            'CHINESE': 'zh-cn',
            'AR': 'ae-ar',
            'ARABIC': 'ae-ar',
            'FR': 'fr-fr',
            'FRENCH': 'fr-fr',
            'PT': 'pt-br',
            'BR': 'pt-br',
            'PORTUGUESE': 'pt-br'
        }
        
        # 패턴 매칭으로 언어 코드 찾기
        for lang_key, lang_code in language_mappings.items():
            if lang_key in col_upper:
                self.logger.info(f"칼럼 '{column_name}'에서 언어 코드 '{lang_code}' 감지")
                return lang_code
        
        # 감지되지 않은 경우 기본값
        self.logger.warning(f"칼럼 '{column_name}'에서 언어 코드를 감지할 수 없습니다. 기본값 'ko-kr' 사용")
        return 'ko-kr'

    def get_target_columns_with_languages(self, df: pd.DataFrame = None) -> List[Dict]:
        """타겟 칼럼들과 감지된 언어 정보를 반환합니다."""
        if df is None:
            df = self.translation_data
        
        if df is None or df.empty:
            return []
        
        column_info = self.get_available_columns(df)
        target_columns = []
        
        for col in column_info['target_candidates']:
            language_code = self.detect_language_from_column_name(col)
            language_name = self.config.SUPPORTED_LANGUAGES.get(language_code, language_code)
            
            target_columns.append({
                'column_name': col,
                'language_code': language_code,
                'language_name': language_name,
                'display_name': f"{col} ({language_name})"
            })
        
        return target_columns
    
    def _extract_language_code(self, column_name: str) -> str:
        """컬럼명에서 언어 코드를 추출합니다."""
        col_lower = str(column_name).lower()
        
        # 지원 언어 코드 확인
        for lang_code in self.config.SUPPORTED_LANGUAGES.keys():
            if lang_code in col_lower:
                return lang_code
        
        # 일반적인 언어 코드 매핑
        language_mapping = {
            'arabic': 'ae-ar',
            'chinese': 'zh-cn',
            'japanese': 'ja-jp',
            'korean': 'ko-kr',
            'french': 'fr-fr',
            'portuguese': 'pt-br',
            'ar': 'ae-ar',
            'zh': 'zh-cn',
            'ja': 'ja-jp',
            'ko': 'ko-kr',
            'fr': 'fr-fr',
            'pt': 'pt-br'
        }
        
        for key, code in language_mapping.items():
            if key in col_lower:
                return code
        
        # 기본값으로 컬럼명 반환
        return column_name
    
    def validate_translation_data(self, translation_pairs: List[Dict]) -> List[Dict]:
        """번역 데이터의 유효성을 검사합니다."""
        valid_pairs = []
        invalid_count = 0
        
        for pair in translation_pairs:
            # 기본 유효성 검사
            if not pair.get('source_text') or not pair.get('target_text'):
                invalid_count += 1
                continue
            
            # 텍스트 길이 검사 (너무 짧거나 긴 텍스트 제외)
            source_len = len(pair['source_text'])
            target_len = len(pair['target_text'])
            
            if source_len < 5 or source_len > 5000 or target_len < 5 or target_len > 5000:
                invalid_count += 1
                continue
            
            # 언어 코드 유효성 검사
            if pair['language_code'] not in self.config.SUPPORTED_LANGUAGES:
                # 지원되지 않는 언어지만 데이터가 유효하면 포함
                self.logger.warning(f"지원되지 않는 언어 코드: {pair['language_code']}")
            
            valid_pairs.append(pair)
        
        self.logger.info(f"유효한 번역 쌍: {len(valid_pairs)}개, 무효한 쌍: {invalid_count}개")
        return valid_pairs
    
    def get_language_statistics(self, translation_pairs: List[Dict]) -> Dict:
        """언어별 통계를 생성합니다."""
        stats = {}
        
        for pair in translation_pairs:
            lang_code = pair['language_code']
            lang_name = pair.get('language_name', lang_code)
            
            if lang_code not in stats:
                stats[lang_code] = {
                    'language_name': lang_name,
                    'count': 0,
                    'avg_source_length': 0,
                    'avg_target_length': 0,
                    'source_lengths': [],
                    'target_lengths': []
                }
            
            stats[lang_code]['count'] += 1
            stats[lang_code]['source_lengths'].append(len(pair['source_text']))
            stats[lang_code]['target_lengths'].append(len(pair['target_text']))
        
        # 평균 계산
        for lang_code in stats:
            if stats[lang_code]['source_lengths']:
                stats[lang_code]['avg_source_length'] = np.mean(stats[lang_code]['source_lengths'])
                stats[lang_code]['avg_target_length'] = np.mean(stats[lang_code]['target_lengths'])
        
        return stats
    
    def export_processed_data(self, translation_pairs: List[Dict], output_path: str):
        """처리된 데이터를 Excel 파일로 내보냅니다."""
        try:
            df = pd.DataFrame(translation_pairs)
            df.to_excel(output_path, index=False)
            self.logger.info(f"처리된 데이터를 {output_path}에 저장했습니다.")
        except Exception as e:
            self.logger.error(f"데이터 내보내기 오류: {e}")


def main():
    """테스트 함수"""
    processor = DataProcessor()
    
    # 메트릭스 데이터 로드
    print("=== 메트릭스 데이터 로드 ===")
    metrics = processor.load_metrics_definitions()
    
    # 번역 데이터 로드
    print("\n=== 번역 데이터 로드 ===")
    df = processor.load_translation_data()
    print(f"로드된 데이터:\n{df.head()}")
    
    # 번역 쌍 추출
    print("\n=== 번역 쌍 추출 ===")
    pairs = processor.extract_translation_pairs()
    
    if pairs:
        print(f"첫 번째 번역 쌍 예시:")
        print(f"ID: {pairs[0]['id']}")
        print(f"소스: {pairs[0]['source_text'][:100]}...")
        print(f"타겟: {pairs[0]['target_text'][:100]}...")
        print(f"언어: {pairs[0]['language_name']} ({pairs[0]['language_code']})")
        
        # 유효성 검사
        valid_pairs = processor.validate_translation_data(pairs)
        
        # 언어별 통계
        stats = processor.get_language_statistics(valid_pairs)
        print(f"\n=== 언어별 통계 ===")
        for lang_code, stat in stats.items():
            print(f"{stat['language_name']}: {stat['count']}개")


if __name__ == "__main__":
    main() 