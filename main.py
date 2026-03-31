import cv2
import numpy as np
import os
import zipfile
import tempfile
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

def load_dataset_from_zip(data_path):
    """
    Загружает датасет из .zip файлов для 4-классовой классификации
    """
    X = []
    y = []
    
    # Ищем .zip файлы в папке images
    zip_files = [f for f in os.listdir(data_path) if f.endswith('.zip')]
    
    if not zip_files:
        print("Не найдено .zip файлов в папке images")
        return np.array(X), np.array(y)
    
    print(f"Найдено {len(zip_files)} .zip файлов")
    
    for zip_file in zip_files:
        zip_path = os.path.join(data_path, zip_file)
        print(f"Обрабатываю {zip_file}...")
        
        # Создаем временную папку для распаковки
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Распаковываем архив
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                # Ищем папки с разными состояниями глаз
                for root, dirs, files in os.walk(temp_dir):
                    for dir_name in dirs:
                        dir_lower = dir_name.lower()
                        
                        # Non Drowsy (открытые глаза) - класс 0
                        if 'non' in dir_lower and 'drowsy' in dir_lower:
                            print(f"  Найдена папка: {dir_name} -> класс 0 (Non Drowsy)")
                            open_path = os.path.join(root, dir_name)
                            for filename in os.listdir(open_path):
                                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                                    img_path = os.path.join(open_path, filename)
                                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                                    if img is not None:
                                        img_resized = cv2.resize(img, (64, 64))
                                        X.append(img_resized.flatten())
                                        y.append(0)
                        
                        # Drowsy (сонные глаза) - класс 1
                        elif 'drowsy' in dir_lower and 'non' not in dir_lower:
                            print(f"  Найдена папка: {dir_name} -> класс 1 (Drowsy)")
                            drowsy_path = os.path.join(root, dir_name)
                            for filename in os.listdir(drowsy_path):
                                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                                    img_path = os.path.join(drowsy_path, filename)
                                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                                    if img is not None:
                                        img_resized = cv2.resize(img, (64, 64))
                                        X.append(img_resized.flatten())
                                        y.append(1)
                        
                        # Eyesclose (просто закрытые глаза) - класс 2
                        elif 'eyesclose' in dir_lower or 'eyes_close' in dir_lower:
                            print(f"  Найдена папка: {dir_name} -> класс 2 (Eyesclose)")
                            eyesclose_path = os.path.join(root, dir_name)
                            for filename in os.listdir(eyesclose_path):
                                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                                    img_path = os.path.join(eyesclose_path, filename)
                                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                                    if img is not None:
                                        img_resized = cv2.resize(img, (64, 64))
                                        X.append(img_resized.flatten())
                                        y.append(2)
                        
                        # Neutral (нейтральные глаза) - класс 3
                        elif 'neutral' in dir_lower:
                            print(f"  Найдена папка: {dir_name} -> класс 3 (Neutral)")
                            neutral_path = os.path.join(root, dir_name)
                            for filename in os.listdir(neutral_path):
                                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                                    img_path = os.path.join(neutral_path, filename)
                                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                                    if img is not None:
                                        img_resized = cv2.resize(img, (64, 64))
                                        X.append(img_resized.flatten())
                                        y.append(3)
                
                print(f"Из {zip_file} загружено изображений: {len([label for label in y if label in [0,1,2,3]])}")
                
            except Exception as e:
                print(f"Ошибка при обработке {zip_file}: {e}")
                continue
    
    # Статистика по классам
    class_names = ['Non Drowsy', 'Drowsy', 'Eyesclose', 'Neutral']
    for i, name in enumerate(class_names):
        count = sum(1 for label in y if label == i)
        print(f"Класс {i} ({name}): {count} изображений")
    
    print(f"Всего загружено изображений: {len(X)}")
    return np.array(X), np.array(y)

def train_classifier(X, y, max_epochs=5):
    """
    Обучает SVM классификатор на датасете
    """
    if len(X) == 0:
        print("Датасет пуст! Поместите .zip файлы с папками 'Non Drowsy', 'Drowsy', 'Eyesclose' и 'Neutral' в папку images.")
        return None, None
    
    # Разделяем данные на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Нормализуем данные
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Создаем и обучаем SVM классификатор
    classifier = SVC(kernel='rbf', random_state=42, probability=True)
    
    print(f"Обучаем классификатор на {len(X_train)} изображениях...")
    for epoch in range(max_epochs):
        classifier.fit(X_train_scaled, y_train)
        train_score = classifier.score(X_train_scaled, y_train)
        test_score = classifier.score(X_test_scaled, y_test)
        print(f"Эпоха {epoch + 1}/{max_epochs}: Точность на обучающей выборке: {train_score:.3f}, на тестовой: {test_score:.3f}")
    
    return classifier, scaler

def detect_eyes(gray_img, eye_cascade):
    """
    Детектирует глаза на изображении с помощью Haar Cascade
    Оптимизировано для изображений высокого разрешения
    """
    # Адаптивные параметры для разных размеров изображения
    height, width = gray_img.shape
    
    if width > 1920:  # Для очень больших изображений
        scale_factor = 1.05
        min_neighbors = 3
        min_size = (40, 40)
    elif width > 1280:  # Для больших изображений
        scale_factor = 1.1
        min_neighbors = 4
        min_size = (35, 35)
    else:  # Для стандартных изображений
        scale_factor = 1.1
        min_neighbors = 5
        min_size = (30, 30)
    
    eyes = eye_cascade.detectMultiScale(
        gray_img, 
        scaleFactor=scale_factor, 
        minNeighbors=min_neighbors, 
        minSize=min_size,
        maxSize=(width//4, height//4)  # Максимальный размер глаза
    )
    
    return eyes

def classify_eye_state(eye_img, classifier, scaler):
    """
    Классифицирует состояние глаза (4 класса)
    """
    if classifier is None or scaler is None:
        return None
    
    # Предобработка изображения глаза
    eye_resized = cv2.resize(eye_img, (64, 64))
    eye_flattened = eye_resized.flatten().reshape(1, -1)
    
    # Нормализация
    eye_scaled = scaler.transform(eye_flattened)
    
    # Предсказание с вероятностью для 4 классов
    prediction = classifier.predict(eye_scaled)[0]
    probability = classifier.predict_proba(eye_scaled)[0]
    
    return prediction, probability

def main():
    """
    Основная функция программы
    """
    # Инициализация веб-камеры
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Ошибка: Не удалось открыть веб-камеру")
        return
    
    # Загружаем каскадный классификатор для глаз
    eye_cascade = cv2.CascadeClassifier('lol.xml')
    if eye_cascade.empty():
        print("Ошибка: Не удалось загрузить каскадный классификатор")
        return
    
    # Загружаем датасет и обучаем классификатор
    print("Загружаем датасет...")
    X, y = load_dataset_from_zip('images')
    
    if len(X) > 0:
        print(f"Загружено {len(X)} изображений")
        classifier, scaler = train_classifier(X, y, max_epochs=5)
        
        # Сохраняем обученную модель
        if classifier is not None:
            joblib.dump(classifier, 'eye_classifier.pkl')
            joblib.dump(scaler, 'eye_scaler.pkl')
            print("Модель сохранена в файлы 'eye_classifier.pkl' и 'eye_scaler.pkl'")
    else:
        print("Датасет не найден. Попробуем загрузить сохраненную модель...")
        try:
            classifier = joblib.load('eye_classifier.pkl')
            scaler = joblib.load('eye_scaler.pkl')
            print("Модель загружена из файлов")
        except:
            print("Сохраненная модель не найдена. Классификация состояния глаз недоступна.")
            classifier, scaler = None, None
    
    print("Нажмите 'q' для выхода")
    
    while True:
        ret, img = cap.read()
        if not ret:
            break
        
        # Отзеркаливаем изображение
        img = cv2.flip(img, 1)
        
        # Конвертируем в оттенки серого
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Детектируем глаза
        eyes = detect_eyes(gray, eye_cascade)
        
        # Обрабатываем каждый обнаруженный глаз
        for (x, y, w, h) in eyes:
            # Вырезаем область глаза
            eye_roi = gray[y:y+h, x:x+w]
            
            # Классифицируем состояние глаза
            if classifier is not None and scaler is not None:
                result = classify_eye_state(eye_roi, classifier, scaler)
                if result is not None:
                    eye_state, probability = result
                    
                    # Определяем цвет и текст в зависимости от состояния
                    if eye_state == 0:  # Non Drowsy
                        color = (0, 255, 0)  # Зеленый
                        label = f"Open ({probability[1]:.2f})"
                    elif eye_state == 1:  # Drowsy
                        color = (0, 0, 255)  # Красный
                        label = f"Closed ({probability[0]:.2f})"
                    elif eye_state == 2:  # Eyesclose
                        color = (255, 0, 0)  # Синий
                        label = f"Eyesclose ({probability[0]:.2f})"
                    elif eye_state == 3:  # Neutral
                        color = (0, 255, 255)  # Желтый
                        label = f"Neutral ({probability[0]:.2f})"
                    
                    # Рисуем прямоугольник и текст
                    cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                # Если классификатор недоступен, рисуем синий прямоугольник
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Показываем результат
        cv2.imshow('Eye Detection', img)
        
        # Выход по нажатию 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Освобождаем ресурсы
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()