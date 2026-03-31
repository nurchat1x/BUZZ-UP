"""
Демонстрационный скрипт для показа возможностей приложения детекции сонливости.
"""

import os
import sys

def print_banner():
    """Выводит красивый баннер"""
    print("=" * 60)
    print("ДЕТЕКЦИЯ СОНЛИВОСТИ - STREAMLIT ПРИЛОЖЕНИЕ")
    print("=" * 60)
    print()

def check_files():
    """Проверяет наличие необходимых файлов"""
    print("Проверка файлов...")
    
    files_to_check = {
        'app.py': 'Основное Streamlit приложение',
        'run_app.py': 'Скрипт запуска',
        'train.py': 'Скрипт обучения модели',
        'infer.py': 'Скрипт инференса',
        'requirements.txt': 'Зависимости Python',
        'README_STREAMLIT.md': 'Документация'
    }
    
    model_files = {
        'eye_classifier.pkl': 'Обученная модель',
        'eye_scaler.pkl': 'Нормализатор данных',
        'class_mapping.pkl': 'Маппинг классов',
        'lol.xml': 'Каскадный классификатор глаз'
    }
    
    all_good = True
    
    print("\nОсновные файлы:")
    for file, description in files_to_check.items():
        if os.path.exists(file):
            print(f"  [OK] {file} - {description}")
        else:
            print(f"  [X] {file} - {description} (ОТСУТСТВУЕТ)")
            all_good = False
    
    print("\nФайлы модели:")
    for file, description in model_files.items():
        if os.path.exists(file):
            print(f"  [OK] {file} - {description}")
        else:
            print(f"  [!] {file} - {description} (НУЖНО ОБУЧИТЬ МОДЕЛЬ)")
    
    return all_good

def show_instructions():
    """Показывает инструкции по использованию"""
    print("\nИНСТРУКЦИИ ПО ИСПОЛЬЗОВАНИЮ:")
    print("-" * 40)
    
    print("\n1. Установка зависимостей:")
    print("   pip install -r requirements.txt")
    
    print("\n2. Обучение модели (если еще не обучена):")
    print("   python train.py")
    
    print("\n3. Запуск Streamlit приложения:")
    print("   python run_app.py")
    print("   или")
    print("   streamlit run app.py")
    
    print("\n4. Открыть браузер:")
    print("   http://localhost:8501")
    
    print("\n5. Использование:")
    print("   - Нажать 'Запустить камеру'")
    print("   - Смотреть прямо в камеру")
    print("   - Следить за статусом: 'Спит' или 'Не Спит'")

def show_features():
    """Показывает возможности приложения"""
    print("\nВОЗМОЖНОСТИ ПРИЛОЖЕНИЯ:")
    print("-" * 40)
    
    features = [
        "Детекция в реальном времени через веб-камеру",
        "Автоматическое обнаружение глаз",
        "Классификация состояния: Спит / Не Спит",
        "Отображение уверенности модели",
        "Цветные рамки вокруг глаз",
        "Современный веб-интерфейс",
        "Быстрая обработка (оптимизированные алгоритмы)",
        "Адаптивные параметры под разрешение камеры"
    ]
    
    for i, feature in enumerate(features, 1):
        print(f"  {i}. {feature}")

def main():
    """Основная функция демонстрации"""
    print_banner()
    
    # Проверяем файлы
    files_ok = check_files()
    
    # Показываем возможности
    show_features()
    
    # Показываем инструкции
    show_instructions()
    
    print("\n" + "=" * 60)
    print("ГОТОВО К ЗАПУСКУ!")
    print("=" * 60)
    
    if not files_ok:
        print("\nВНИМАНИЕ: Некоторые файлы отсутствуют!")
        print("Убедитесь, что все файлы находятся в правильной папке.")
    
    print("\nСовет: Начните с обучения модели, затем запустите приложение!")
    print("\nНажмите Enter для выхода...")
    input()

if __name__ == "__main__":
    main()