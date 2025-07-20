import pandas as pd
import os

def create_sample_a():
    """Sample A: 기술 문서 번역 예제 (영어 -> 일본어)"""
    
    data = {
        'SOURCE-EN': [
            "Click the Settings button to configure your preferences.",
            "Your password must contain at least 8 characters including uppercase, lowercase, and numbers.",
            "The application will automatically save your work every 5 minutes.",
            "To delete a file, right-click on it and select 'Delete' from the context menu.",
            "Please restart the application to apply the changes.",
            "Your subscription will be renewed automatically unless you cancel it.",
            "The system detected an error. Please try again later.",
            "You can export your data in CSV or Excel format.",
            "Two-factor authentication adds an extra layer of security to your account.",
            "The feature you requested is currently in development.",
        ],
        'TARGET-JA': [
            "設定ボタンをクリックして、お好みの設定を構成してください。",
            "パスワードは大文字、小文字、数字を含む8文字以上である必要があります。",
            "アプリケーションは5分ごとに自動的に作業を保存します。",
            "ファイルを削除するには、右クリックしてコンテキストメニューから「削除」を選択してください。",
            "変更を適用するには、アプリケーションを再起動してください。",
            "キャンセルしない限り、サブスクリプションは自動的に更新されます。",
            "システムでエラーが検出されました。後でもう一度お試しください。",
            "データはCSVまたはExcel形式でエクスポートできます。",
            "二要素認証はアカウントにセキュリティの追加レイヤーを提供します。",
            "ご要望の機能は現在開発中です。",
        ],
        'OVERALL': [None] * 10,
        'ACCURACY': [None] * 10,
        'OMISSION/ADDITION': [None] * 10,
        'COMPLIANCE': [None] * 10,
        'FLUENCY': [None] * 10
    }
    
    df = pd.DataFrame(data)
    df.to_excel('Sample A.xlsx', index=False)
    print("Sample A.xlsx 생성 완료: 기술 문서 번역 (영어 -> 일본어)")
    return df

def create_sample_c():
    """Sample C: 마케팅 문구 번역 예제 (영어 -> 프랑스어)"""
    
    data = {
        'SOURCE-EN': [
            "Discover the future of innovation with our cutting-edge technology.",
            "Join millions of satisfied customers worldwide.",
            "Transform your business with our comprehensive solutions.",
            "Experience premium quality at an affordable price.",
            "Get started today with our free trial offer.",
            "Your success is our priority.",
            "Unlock your potential with personalized training programs.",
            "Connect with experts who understand your industry.",
            "Streamline your workflow and boost productivity.",
            "Trusted by leading companies across the globe.",
        ],
        'TARGET-FR': [
            "Découvrez l'avenir de l'innovation avec notre technologie de pointe.",
            "Rejoignez des millions de clients satisfaits dans le monde entier.",
            "Transformez votre entreprise avec nos solutions complètes.",
            "Découvrez une qualité premium à un prix abordable.",
            "Commencez dès aujourd'hui avec notre offre d'essai gratuit.",
            "Votre succès est notre priorité.",
            "Libérez votre potentiel avec des programmes de formation personnalisés.",
            "Connectez-vous avec des experts qui comprennent votre secteur.",
            "Rationalisez votre flux de travail et augmentez la productivité.",
            "Fait confiance par les entreprises leaders à travers le monde.",
        ],
        'OVERALL': [None] * 10,
        'ACCURACY': [None] * 10,
        'OMISSION/ADDITION': [None] * 10,
        'COMPLIANCE': [None] * 10,
        'FLUENCY': [None] * 10
    }
    
    df = pd.DataFrame(data)
    df.to_excel('Sample C.xlsx', index=False)
    print("Sample C.xlsx 생성 완료: 마케팅 문구 번역 (영어 -> 프랑스어)")
    return df

def create_sample_d():
    """Sample D: 법률/공식 문서 번역 예제 (영어 -> 아랍어)"""
    
    data = {
        'SOURCE-EN': [
            "This agreement shall be governed by the laws of the jurisdiction.",
            "All parties must comply with applicable regulations and standards.",
            "The contract becomes effective upon signature by all parties.",
            "Any disputes shall be resolved through binding arbitration.",
            "Confidential information must not be disclosed to third parties.",
            "The terms and conditions are subject to change without notice.",
            "Violation of these terms may result in immediate termination.",
            "All intellectual property rights are reserved by the company.",
            "Users are responsible for maintaining the security of their accounts.",
            "This policy takes effect immediately upon publication.",
        ],
        'TARGET-AR': [
            "تحكم قوانين الاختصاص القضائي هذه الاتفاقية.",
            "يجب على جميع الأطراف الامتثال للوائح والمعايير المعمول بها.",
            "يصبح العقد ساري المفعول عند توقيع جميع الأطراف.",
            "يجب حل أي نزاعات من خلال التحكيم الملزم.",
            "يجب عدم الكشف عن المعلومات السرية لأطراف ثالثة.",
            "الشروط والأحكام عرضة للتغيير دون إشعار مسبق.",
            "قد يؤدي انتهاك هذه الشروط إلى الإنهاء الفوري.",
            "جميع حقوق الملكية الفكرية محفوظة للشركة.",
            "المستخدمون مسؤولون عن الحفاظ على أمان حساباتهم.",
            "تدخل هذه السياسة حيز التنفيذ فور نشرها.",
        ],
        'OVERALL': [None] * 10,
        'ACCURACY': [None] * 10,
        'OMISSION/ADDITION': [None] * 10,
        'COMPLIANCE': [None] * 10,
        'FLUENCY': [None] * 10
    }
    
    df = pd.DataFrame(data)
    df.to_excel('Sample D.xlsx', index=False)
    print("Sample D.xlsx 생성 완료: 법률/공식 문서 번역 (영어 -> 아랍어)")
    return df

def create_sample_e():
    """Sample E: 일상 대화 번역 예제 (영어 -> 중국어)"""
    
    data = {
        'SOURCE-EN': [
            "How was your day today?",
            "Would you like to have dinner together?",
            "I'm sorry, I'm running late.",
            "Could you please help me with this?",
            "The weather is beautiful today.",
            "Thank you so much for your kindness.",
            "I hope you have a wonderful weekend.",
            "See you tomorrow at the meeting.",
        ],
        'TARGET-ZH': [
            "您今天过得怎么样？",
            "您愿意一起吃晚饭吗？",
            "对不起，我迟到了。",
            "您能帮我一下吗？",
            "今天天气真好。",
            "非常感谢您的善意。",
            "祝您周末愉快。",
            "明天会议上见。",
        ],
        'OVERALL': [None] * 8,
        'ACCURACY': [None] * 8,
        'OMISSION/ADDITION': [None] * 8,
        'COMPLIANCE': [None] * 8,
        'FLUENCY': [None] * 8
    }
    
    df = pd.DataFrame(data)
    df.to_excel('Sample E.xlsx', index=False)
    print("Sample E.xlsx 생성 완료: 일상 대화 번역 (영어 -> 중국어)")
    return df

def create_sample_f():
    """Sample F: 여행 가이드 번역 예제 (영어 -> 포르투갈어)"""
    
    data = {
        'SOURCE-EN': [
            "Welcome to our beautiful city!",
            "The museum is open from 9 AM to 5 PM.",
            "Please keep your ticket until the end of your visit.",
            "Photography is not allowed in this area.",
            "The next guided tour starts in 30 minutes.",
            "Emergency exits are located on both sides of the building.",
        ],
        'TARGET-PT': [
            "Bem-vindos à nossa bela cidade!",
            "O museu está aberto das 9h às 17h.",
            "Por favor, mantenha seu ingresso até o final da visita.",
            "Não é permitido fotografar nesta área.",
            "A próxima visita guiada começa em 30 minutos.",
            "As saídas de emergência estão localizadas em ambos os lados do edifício.",
        ],
        'OVERALL': [None] * 6,
        'ACCURACY': [None] * 6,
        'OMISSION/ADDITION': [None] * 6,
        'COMPLIANCE': [None] * 6,
        'FLUENCY': [None] * 6
    }
    
    df = pd.DataFrame(data)
    df.to_excel('Sample F.xlsx', index=False)
    print("Sample F.xlsx 생성 완료: 여행 가이드 번역 (영어 -> 포르투갈어)")
    return df

def main():
    """모든 샘플 파일 생성"""
    print("=== MT-Eval Pro 샘플 데이터 생성 ===")
    print()
    
    samples = [
        create_sample_a,
        create_sample_c, 
        create_sample_d,
        create_sample_e,
        create_sample_f
    ]
    
    created_files = []
    
    for create_func in samples:
        try:
            df = create_func()
            created_files.append(create_func.__name__.replace('create_', '').upper() + '.xlsx')
            print()
        except Exception as e:
            print(f"오류: {create_func.__name__} 생성 실패 - {e}")
            print()
    
    print("=== 생성 완료 ===")
    print(f"생성된 파일: {len(created_files)}개")
    for file in created_files:
        if os.path.exists(file):
            size = os.path.getsize(file) / 1024
            print(f"  - {file} ({size:.1f} KB)")
    
    print()
    print("각 파일의 특징:")
    print("- Sample A.xlsx: 기술 문서 (영어 -> 일본어)")
    print("- Sample C.xlsx: 마케팅 문구 (영어 -> 프랑스어)")
    print("- Sample D.xlsx: 법률 문서 (영어 -> 아랍어)")
    print("- Sample E.xlsx: 일상 대화 (영어 -> 중국어)")
    print("- Sample F.xlsx: 여행 가이드 (영어 -> 포르투갈어)")
    print()
    print("각 파일은 SOURCE-EN, TARGET-XX, OVERALL, ACCURACY, OMISSION/ADDITION, COMPLIANCE, FLUENCY 구조를 가집니다.")
    print("점수 칼럼들은 비어있으며, AI가 평가하여 채우게 됩니다.")

if __name__ == "__main__":
    main() 