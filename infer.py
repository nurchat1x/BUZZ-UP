"""
Файл для инференса (предсказания) с использованием обученной модели.

Использование:
    python infer.py

Этот файл загружает уже обученные .pkl файлы и делает предсказания
без переобучения модели.
"""

import cv2
import numpy as np
import joblib
import os
import argparse


def detect_eyes(gray_img, eye_cascade):
    """
    Детекция глаз с оптимизированными параметрами
    """
    height, width = gray_img.shape
    
    if width > 1920:
        scale_factor = 1.05
        min_neighbors = 3
        min_size = (40, 40)
    elif width > 1280:
        scale_factor = 1.1
        min_neighbors = 4
        min_size = (35, 35)
    else:
        scale_factor = 1.1
        min_neighbors = 5
        min_size = (30, 30)
    
    eyes = eye_cascade.detectMultiScale(
        gray_img, 
        scaleFactor=scale_factor, 
        minNeighbors=min_neighbors, 
        minSize=min_size,
        maxSize=(width//4, height//4)
    )
    
    return eyes


def classify_eye_state_fast(eye_img, classifier, scaler):
    """
    Быстрая классификация состояния глаза для любого количества классов
    """
    if classifier is None or scaler is None:
        return None
    
    # Изменяем размер до 32x32 (как при обучении)
    eye_resized = cv2.resize(eye_img, (32, 32))
    eye_flattened = eye_resized.flatten().reshape(1, -1)
    
    # Нормализация и предсказание
    eye_scaled = scaler.transform(eye_flattened)
    prediction = classifier.predict(eye_scaled)[0]
    probability = classifier.predict_proba(eye_scaled)[0]
    
    return prediction, probability


def load_model():
    """
    Загружает обученную модель и связанные файлы
    """
    try:
        classifier = joblib.load('eye_classifier.pkl')
        scaler = joblib.load('eye_scaler.pkl')
        class_mapping = joblib.load('class_mapping.pkl')
        print("✅ Модель успешно загружена!")
        return classifier, scaler, class_mapping
    except FileNotFoundError as e:
        print(f"❌ Файл модели не найден: {e}")
        print("Сначала запустите обучение: python train.py")
        return None, None, None
    except Exception as e:
        print(f"❌ Ошибка при загрузке модели: {e}")
        return None, None, None


def predict_image(image_path, classifier, scaler, class_mapping):
    """
    Предсказывает состояние глаз на изображении
    """
    # Загружаем изображение
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"❌ Не удалось загрузить изображение: {image_path}")
        return None
    
    # Загружаем каскадный классификатор
    eye_cascade = cv2.CascadeClassifier('lol.xml')
    if eye_cascade.empty():
        print("❌ Не удалось загрузить каскадный классификатор")
        return None
    
    # Детектируем глаза
    eyes = detect_eyes(img, eye_cascade)
    
    if len(eyes) == 0:
        print("❌ Глаза не обнаружены на изображении")
        return None
    
    print(f"✅ Обнаружено {len(eyes)} глаз(а)")
    
    # Классифицируем каждое найденное глаз
    results = []
    for i, (x, y, w, h) in enumerate(eyes):
        eye_roi = img[y:y+h, x:x+w]
        
        result = classify_eye_state_fast(eye_roi, classifier, scaler)
        if result is not None:
            eye_state, probability = result
            
            # Находим название класса
            class_name = "Unknown"
            for name, class_id in class_mapping.items():
                if class_id == eye_state:
                    class_name = name
                    break
            
            # Простая классификация для 2 классов
            if eye_state == 0:  # Закрытые глаза
                eye_state_name = "Close Eyes"
            elif eye_state == 1:  # Сонные глаза
                eye_state_name = "Drowsy"
            elif eye_state == 2:  # Открытые глаза
                eye_state_name = "Open Eyes"
            else:
                eye_state_name = "Unknown"
            
            confidence = probability[eye_state]
            results.append({
                'eye_id': i + 1,
                'state': eye_state,
                'state_name': eye_state_name,
                'class_name': class_name,
                'confidence': confidence,
                'bbox': (x, y, w, h)
            })
            
            print(f"Глаз {i + 1}: {eye_state_name} (уверенность: {confidence:.3f})")
    
    return results


def run_webcam_inference(classifier, scaler, class_mapping):
    """
    Запускает инференс в реальном времени с веб-камеры
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Ошибка: Не удалось открыть веб-камеру")
        return
    
    eye_cascade = cv2.CascadeClassifier('lol.xml')
    if eye_cascade.empty():
        print("❌ Ошибка: Не удалось загрузить каскадный классификатор")
        return
    
    print("🎥 Запуск веб-камеры...")
    print("Нажмите 'q' для выхода")
    
    while True:
        ret, img = cap.read()
        if not ret:
            break
        
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        eyes = detect_eyes(gray, eye_cascade)
        
        for (x, y, w, h) in eyes:
            eye_roi = gray[y:y+h, x:x+w]
            
            result = classify_eye_state_fast(eye_roi, classifier, scaler)
            if result is not None:
                eye_state, probability = result
                
                # Находим название класса
                class_name = "Unknown"
                for name, class_id in class_mapping.items():
                    if class_id == eye_state:
                        class_name = name
                        break
                
                # Простая классификация для 2 классов
                if eye_state == 0:  # Закрытые глаза
                    eye_state_name = "Close Eyes"
                    color = (0, 0, 255)  # Красный
                elif eye_state == 1:  # Сонные глаза
                    eye_state_name = "Drowsy"
                    color = (0, 165, 255)  # Оранжевый
                elif eye_state == 2:  # Открытые глаза
                    eye_state_name = "Open Eyes"
                    color = (0, 255, 0)  # Зеленый
                else:
                    eye_state_name = "Unknown"
                    color = (128, 128, 128)  # Серый
                
                label = f"{eye_state_name} ({probability[eye_state]:.2f})"
                
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        cv2.putText(img, "Press 'q' to quit", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('Eye State Detection', img)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


def main():
    """
    Основная функция инференса
    """
    parser = argparse.ArgumentParser(description='Инференс модели классификации состояния глаз')
    parser.add_argument('--image', type=str, help='Путь к изображению для предсказания')
    parser.add_argument('--webcam', action='store_true', help='Запустить инференс с веб-камеры')
    
    args = parser.parse_args()
    
    print("=== ИНФЕРЕНС МОДЕЛИ КЛАССИФИКАЦИИ СОСТОЯНИЯ ГЛАЗ ===")
    
    # Загружаем модель
    classifier, scaler, class_mapping = load_model()
    
    if classifier is None:
        return
    
    print(f"Загружена модель для {len(class_mapping)} классов:")
    for name, class_id in class_mapping.items():
        print(f"  - {name}: класс {class_id}")
    
    # Выбираем режим работы
    if args.image:
        # Предсказание на изображении
        print(f"\n📸 Анализ изображения: {args.image}")
        results = predict_image(args.image, classifier, scaler, class_mapping)
        
        if results:
            print("\n📊 Результаты:")
            for result in results:
                print(f"Глаз {result['eye_id']}: {result['state_name']} "
                      f"(уверенность: {result['confidence']:.3f})")
    
    elif args.webcam:
        # Инференс с веб-камеры
        run_webcam_inference(classifier, scaler, class_mapping)
    
    else:
        # Интерактивный режим
        print("\nВыберите режим:")
        print("1. Анализ изображения")
        print("2. Инференс с веб-камеры")
        print("3. Тестовый пример")
        
        choice = input("Введите номер (1-3): ").strip()
        
        if choice == '1':
            image_path = input("Введите путь к изображению: ").strip()
            if os.path.exists(image_path):
                results = predict_image(image_path, classifier, scaler, class_mapping)
                if results:
                    print("\n📊 Результаты:")
                    for result in results:
                        print(f"Глаз {result['eye_id']}: {result['state_name']} "
                              f"(уверенность: {result['confidence']:.3f})")
            else:
                print("❌ Файл не найден!")
        
        elif choice == '2':
            run_webcam_inference(classifier, scaler, class_mapping)
        
        elif choice == '3':
            # Тестовый пример с изображением из папки images
            test_images = []
            if os.path.exists('images'):
                for root, dirs, files in os.walk('images'):
                    for file in files:
                        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                            test_images.append(os.path.join(root, file))
                            if len(test_images) >= 3:  # Берем первые 3 изображения
                                break
                    if len(test_images) >= 3:
                        break
            
            if test_images:
                print(f"\n🧪 Тестовый пример с изображением: {test_images[0]}")
                results = predict_image(test_images[0], classifier, scaler, class_mapping)
                if results:
                    print("\n📊 Результаты:")
                    for result in results:
                        print(f"Глаз {result['eye_id']}: {result['state_name']} "
                              f"(уверенность: {result['confidence']:.3f})")
            else:
                print("❌ Тестовые изображения не найдены в папке images")


if __name__ == "__main__":
    main()
