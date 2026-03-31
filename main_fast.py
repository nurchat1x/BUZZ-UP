"""
Универсальный запуск системы классификации состояния глаз.

Использование:
    python main_fast.py --mode train    # Обучение модели
    python main_fast.py --mode infer    # Инференс с веб-камеры
    python main_fast.py                 # Интерактивный выбор режима

Этот файл служит точкой входа для выбора между обучением и инференсом.
"""

import argparse
import sys
import os

def run_training():
    """
    Запускает обучение модели
    """
    print("🚀 Запуск обучения модели...")
    try:
        import train
        train.main()
    except ImportError:
        print("❌ Ошибка: Не удалось импортировать модуль train.py")
        print("Убедитесь, что файл train.py существует в текущей директории")
    except Exception as e:
        print(f"❌ Ошибка при обучении: {e}")


def run_inference():
    """
    Запускает инференс с веб-камеры
    """
    print("🎥 Запуск инференса с веб-камеры...")
    try:
        import infer
        # Запускаем инференс с веб-камеры
        infer.run_webcam_inference(*infer.load_model())
    except ImportError:
        print("❌ Ошибка: Не удалось импортировать модуль infer.py")
        print("Убедитесь, что файл infer.py существует в текущей директории")
    except Exception as e:
        print(f"❌ Ошибка при инференсе: {e}")


def interactive_mode():
    """
    Интерактивный выбор режима работы
    """
    print("\n=== СИСТЕМА КЛАССИФИКАЦИИ СОСТОЯНИЯ ГЛАЗ ===")
    print("Выберите режим работы:")
    print("1. 🎓 Обучение модели")
    print("2. 🎥 Инференс с веб-камеры")
    print("3. 📸 Анализ изображения")
    print("4. 🧪 Тестовый пример")
    print("5. ❌ Выход")
    
    while True:
        choice = input("\nВведите номер (1-5): ").strip()
        
        if choice == '1':
            run_training()
            break
        elif choice == '2':
            run_inference()
            break
        elif choice == '3':
            image_path = input("Введите путь к изображению: ").strip()
            if os.path.exists(image_path):
                try:
                    import infer
                    classifier, scaler, class_mapping = infer.load_model()
                    if classifier is not None:
                        results = infer.predict_image(image_path, classifier, scaler, class_mapping)
                        if results:
                            print("\n📊 Результаты:")
                            for result in results:
                                print(f"Глаз {result['eye_id']}: {result['state_name']} "
                                      f"(уверенность: {result['confidence']:.3f})")
                except Exception as e:
                    print(f"❌ Ошибка: {e}")
            else:
                print("❌ Файл не найден!")
            break
        elif choice == '4':
            try:
                import infer
                classifier, scaler, class_mapping = infer.load_model()
                if classifier is not None:
                    # Ищем тестовое изображение
                    test_images = []
                    if os.path.exists('images'):
                        for root, dirs, files in os.walk('images'):
                            for file in files:
                                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                                    test_images.append(os.path.join(root, file))
                                    if len(test_images) >= 1:
                                        break
                            if len(test_images) >= 1:
                                break
                    
                    if test_images:
                        print(f"\n🧪 Тестовый пример с изображением: {test_images[0]}")
                        results = infer.predict_image(test_images[0], classifier, scaler, class_mapping)
                        if results:
                            print("\n📊 Результаты:")
                            for result in results:
                                print(f"Глаз {result['eye_id']}: {result['state_name']} "
                                      f"(уверенность: {result['confidence']:.3f})")
                    else:
                        print("❌ Тестовые изображения не найдены в папке images")
            except Exception as e:
                print(f"❌ Ошибка: {e}")
            break
        elif choice == '5':
            print("👋 До свидания!")
            break
        else:
            print("❌ Неверный выбор. Введите число от 1 до 5.")

def main():
    """
    Основная функция - точка входа в программу
    """
    parser = argparse.ArgumentParser(
        description='Система классификации состояния глаз',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python main_fast.py --mode train    # Обучение модели
  python main_fast.py --mode infer   # Инференс с веб-камеры
  python main_fast.py                # Интерактивный выбор режима
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['train', 'infer'], 
        help='Режим работы: train (обучение) или infer (инференс)'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        run_training()
    elif args.mode == 'infer':
        run_inference()
    else:
        interactive_mode()

if __name__ == "__main__":
    main() 