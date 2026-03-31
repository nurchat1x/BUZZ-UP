"""
Файл для обучения модели классификации состояния глаз.

Использование:
    python train.py

Этот файл содержит всю логику загрузки датасета, препроцессинга и обучения модели.
После обучения модель и scaler сохраняются в файлы .pkl.
"""

import cv2
import numpy as np
import os
import zipfile
import tempfile
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from tqdm import tqdm
import time


def _try_load_image_file(path):
    """Попытаться загрузить файл как изображение. Вернуть список (0 или 1) загруженных изображений (flatten)."""
    imgs = []
    try:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img_resized = cv2.resize(img, (32, 32))
            imgs.append(img_resized.flatten())
    except Exception:
        pass
    return imgs

def _collect_images_from_dir(root_path):
    """Собрать все изображения из директории (рекурсивно). Возвращает список flatten-изображений."""
    collected = []
    for r, d, files in os.walk(root_path):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                p = os.path.join(r, f)
                collected.extend(_try_load_image_file(p))
    return collected

def _collect_images_from_zip(zip_path):
    """Распаковать zip во временную папку и собрать изображения рекурсивно."""
    collected = []
    try:
        if zipfile.is_zipfile(zip_path):
            with tempfile.TemporaryDirectory() as tmp:
                with zipfile.ZipFile(zip_path, 'r') as z:
                    z.extractall(tmp)
                collected.extend(_collect_images_from_dir(tmp))
    except Exception:
        pass
    return collected

def _collect_images_from_txt(txt_path, base_root):
    """Прочитать .txt и попытаться загрузить каждую строку как путь (относительный к base_root или абсолютный)."""
    collected = []
    try:
        with open(txt_path, 'r', encoding='utf-8', errors='ignore') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                # относительный сначала
                candidate = os.path.join(base_root, line) if not os.path.isabs(line) else line
                if os.path.exists(candidate):
                    if os.path.isdir(candidate):
                        collected.extend(_collect_images_from_dir(candidate))
                    elif zipfile.is_zipfile(candidate):
                        collected.extend(_collect_images_from_zip(candidate))
                    else:
                        collected.extend(_try_load_image_file(candidate))
    except Exception:
        pass
    return collected

def load_dataset_fast(data_path):
    """
    Упрощенная загрузка из 2 папок: Eyesclose (закрытые) и Neutral (открытые)
    """
    X = []
    y = []
    class_mapping = {}
    
    # Ищем .zip файлы в папке images
    zip_files = [f for f in os.listdir(data_path) if f.endswith('.zip')]
    
    if not zip_files:
        print("Не найдено .zip файлов в папке images")
        return np.array(X), np.array(y), class_mapping
    
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
                
                print(f"  Содержимое {zip_file}:")
                # Показываем структуру распакованного архива
                for root, dirs, files in os.walk(temp_dir):
                    level = root.replace(temp_dir, '').count(os.sep)
                    indent = ' ' * 2 * level
                    print(f"{indent}{os.path.basename(root)}/")
                    subindent = ' ' * 2 * (level + 1)
                    for d in dirs:
                        print(f"{subindent}{d}/")
                    for f in files[:5]:  # Показываем первые 5 файлов
                        print(f"{subindent}{f}")
                    if len(files) > 5:
                        print(f"{subindent}... и еще {len(files) - 5} файлов")
                
                # Ищем только 2 нужные папки
                for root, dirs, files in os.walk(temp_dir):
                    for dir_name in dirs:
                        dir_lower = dir_name.lower()
                        print(f"  Проверяю папку: {dir_name}")
                        
                        # Eyesclose (закрытые глаза) - класс 0
                        if 'eyesclose' in dir_lower or 'eyes_close' in dir_lower or 'close' in dir_lower:
                            if 'eyesclose' not in class_mapping:
                                class_mapping['eyesclose'] = 0
                                print(f"  ✅ Найдена папка закрытых глаз: {dir_name} -> класс 0 (Close Eyes)")
                            
                            current_path = os.path.join(root, dir_name)
                            image_files = [f for f in os.listdir(current_path) 
                                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                            
                            if image_files:
                                print(f"    📁 Загружаю {len(image_files)} изображений закрытых глаз...")
                                for filename in tqdm(image_files, desc="Закрытые глаза"):
                                    img_path = os.path.join(current_path, filename)
                                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                                    if img is not None:
                                        img_resized = cv2.resize(img, (32, 32))
                                        X.append(img_resized.flatten())
                                        y.append(0)  # 0 для закрытых глаз
                            else:
                                print(f"    ⚠️  В папке {dir_name} нет изображений!")
                        
                        # Drowsy (сонные глаза) - класс 1
                        elif 'drowsy' in dir_lower and 'non' not in dir_lower:
                            if 'drowsy' not in class_mapping:
                                class_mapping['drowsy'] = 1
                                print(f"  ✅ Найдена папка сонных глаз: {dir_name} -> класс 1 (Drowsy)")
                            
                            current_path = os.path.join(root, dir_name)
                            image_files = [f for f in os.listdir(current_path) 
                                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                            
                            if image_files:
                                print(f"    📁 Загружаю {len(image_files)} изображений сонных глаз...")
                                for filename in tqdm(image_files, desc="Сонные глаза"):
                                    img_path = os.path.join(current_path, filename)
                                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                                    if img is not None:
                                        img_resized = cv2.resize(img, (32, 32))
                                        X.append(img_resized.flatten())
                                        y.append(1)  # 1 для сонных глаз
                            else:
                                print(f"    ⚠️  В папке {dir_name} нет изображений!")
                        
                        # Neutral (открытые глаза) - класс 2
                        elif 'neutral' in dir_lower or 'non' in dir_lower or 'open' in dir_lower:
                            if 'neutral' not in class_mapping:
                                class_mapping['neutral'] = 2
                                print(f"  ✅ Найдена папка открытых глаз: {dir_name} -> класс 2 (Open Eyes)")
                            
                            current_path = os.path.join(root, dir_name)
                            image_files = [f for f in os.listdir(current_path) 
                                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                            
                            if image_files:
                                print(f"    📁 Загружаю {len(image_files)} изображений открытых глаз...")
                                for filename in tqdm(image_files, desc="Открытые глаза"):
                                    img_path = os.path.join(current_path, filename)
                                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                                    if img is not None:
                                        img_resized = cv2.resize(img, (32, 32))
                                        X.append(img_resized.flatten())
                                        y.append(2)  # 2 для открытых глаз
                            else:
                                print(f"    ⚠️  В папке {dir_name} нет изображений!")
                        
                        else:
                            print(f"  ⚠️  Пропускаю папку: {dir_name} (не подходит под критерии)")
                    
                    # --- Новая часть: также обрабатываем отдельные файлы внутри текущей папки
                    for filename in files:
                        fname_lower = filename.lower()
                        file_path = os.path.join(root, filename)
                        
                        # Ищем файлы, явно указывающие на "open eyes" (например "open eyes", "open_eyes" и т.п.)
                        if 'open' in fname_lower and 'eye' in fname_lower:
                            print(f"  🔎 Найден файл-перечислитель/архив/изображение для Open Eyes: {os.path.join(root, filename)}")
                            # Если это zip — распаковать и собрать
                            if zipfile.is_zipfile(file_path):
                                imgs = _collect_images_from_zip(file_path)
                                if imgs:
                                    if 'neutral' not in class_mapping:
                                        class_mapping['neutral'] = 2
                                    for im in imgs:
                                        X.append(im); y.append(2)
                                    print(f"    📦 Загружено {len(imgs)} изображений из архива {filename}")
                                else:
                                    print(f"    ⚠️ Не удалось найти изображения в архиве {filename}")
                            
                            # Если это текстовый файл со списком путей
                            elif filename.lower().endswith('.txt'):
                                imgs = _collect_images_from_txt(file_path, root)
                                if imgs:
                                    if 'neutral' not in class_mapping:
                                        class_mapping['neutral'] = 2
                                    for im in imgs:
                                        X.append(im); y.append(2)
                                    print(f"    📝 Загружено {len(imgs)} изображений по списку в {filename}")
                                else:
                                    print(f"    ⚠️ Пустой или некорректный список в {filename}")
                            
                            else:
                                # Попытаться загрузить файл как одиночное изображение
                                imgs = _try_load_image_file(file_path)
                                if imgs:
                                    if 'neutral' not in class_mapping:
                                        class_mapping['neutral'] = 2
                                    for im in imgs:
                                        X.append(im); y.append(2)
                                    print(f"    🖼️ Загружено {len(imgs)} изображение(й) из файла {filename}")
                                else:
                                    # Возможно это директория без расширения или другой контейнер — попытаться пройти как папку
                                    if os.path.isdir(file_path):
                                        imgs = _collect_images_from_dir(file_path)
                                        if imgs:
                                            if 'neutral' not in class_mapping:
                                                class_mapping['neutral'] = 2
                                            for im in imgs:
                                                X.append(im); y.append(2)
                                            print(f"    📁 Загружено {len(imgs)} изображений из папки {filename}")
                                        else:
                                            print(f"    ⚠️ Не удалось найти изображения в {filename}")
                                    else:
                                        print(f"    ⚠️ Файл {filename} не распознан как изображение, zip или список")
                    
                print(f"Из {zip_file} загружено изображений: {len([label for label in y if label in [0, 1, 2]])}")
                
            except Exception as e:
                print(f"Ошибка при обработке {zip_file}: {e}")
                continue
    
    # Статистика по классам
    print("\n=== СТАТИСТИКА ПО КЛАССАМ ===")
    if 0 in y:
        count_close = sum(1 for label in y if label == 0)
        print(f"Класс 0 (Close Eyes - закрытые глаза): {count_close} изображений")
    if 1 in y:
        count_drowsy = sum(1 for label in y if label == 1)
        print(f"Класс 1 (Drowsy - сонные глаза): {count_drowsy} изображений")
    if 2 in y:
        count_open = sum(1 for label in y if label == 2)
        print(f"Класс 2 (Open Eyes - открытые глаза): {count_open} изображений")
    
    print(f"\nВсего загружено изображений: {len(X)}")
    print(f"Всего классов: {len(class_mapping)}")
    
    return np.array(X), np.array(y), class_mapping


def train_classifier_fast(X, y, max_epochs=15):
    """
    Быстрое обучение с LogisticRegression для любого количества классов
    """
    if len(X) == 0:
        print("Датасет пуст!")
        return None, None
    
    # Разделяем данные
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Нормализация
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Используем LogisticRegression с лучшими параметрами
    classifier = LogisticRegression(
        random_state=42, 
        max_iter=2000,  # Увеличил количество итераций
        solver='lbfgs', 
        multi_class='multinomial',
        C=1.0,  # Регуляризация
        penalty='l2'
    )
    
    print(f"Быстрое обучение на {len(X_train)} изображениях для {len(set(y))} классов...")
    print(f"Количество эпох: {max_epochs}")
    start_time = time.time()
    
    best_score = 0
    best_classifier = None
    
    for epoch in range(max_epochs):
        epoch_start = time.time()
        classifier.fit(X_train_scaled, y_train)
        epoch_time = time.time() - epoch_start
        
        train_score = classifier.score(X_train_scaled, y_train)
        test_score = classifier.score(X_test_scaled, y_test)
        
        print(f"Эпоха {epoch + 1}/{max_epochs}: Точность {train_score:.3f}/{test_score:.3f} (время: {epoch_time:.1f}с)")
        
        # Сохраняем лучшую модель
        if test_score > best_score:
            best_score = test_score
            best_classifier = classifier
            print(f"  🎯 Новая лучшая модель! Точность: {best_score:.3f}")
    
    total_time = time.time() - start_time
    print(f"Обучение завершено за {total_time:.1f} секунд!")
    print(f"Лучшая точность на тестовой выборке: {best_score:.3f}")
    
    return best_classifier, scaler


def inference_on_image(image_path, classifier, scaler, class_mapping):
    """Load single image, preprocess, predict and print result."""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"❌ Не удалось загрузить изображение: {image_path}")
            return
        img_resized = cv2.resize(img, (32, 32)).flatten().reshape(1, -1)
        img_scaled = scaler.transform(img_resized)
        pred = classifier.predict(img_scaled)[0]
        probs = None
        try:
            probs = classifier.predict_proba(img_scaled)[0]
        except Exception:
            pass
        inv_map = {v: k for k, v in class_mapping.items()}
        class_name = inv_map.get(pred, str(pred))
        print(f"Результат инференса для {image_path}: {class_name} (label={pred})")
        if probs is not None:
            prob_str = ", ".join(f"{inv_map.get(i,i)}:{p:.3f}" for i, p in enumerate(probs))
            print(f"Вероятности: {prob_str}")
    except Exception as e:
        print(f"Ошибка при инференсе: {e}")

def run_inference_demo(classifier, scaler, class_mapping):
    """Try a few common locations for a sample image and run inference if found."""
    candidates = ['sample.jpg', os.path.join('images', 'sample.jpg'), 'test.jpg', os.path.join('images', 'test.jpg')]
    for c in candidates:
        if os.path.exists(c):
            print(f"🔎 Запускаю демо на {c} ...")
            inference_on_image(c, classifier, scaler, class_mapping)
            return
    print("⚠️  Не найден sample image. Поместите 'sample.jpg' в корень проекта или в папку 'images/' и повторите.")

def main():
    """
    Основная функция обучения модели
    """
    print("=== ОБУЧЕНИЕ МОДЕЛИ КЛАССИФИКАЦИИ СОСТОЯНИЯ ГЛАЗ ===")
    
    # Проверяем существование файлов модели
    model_files = ['eye_classifier.pkl', 'eye_scaler.pkl', 'class_mapping.pkl']
    existing_files = [f for f in model_files if os.path.exists(f)]
    
    if existing_files:
        print(f"Найдены существующие файлы модели: {existing_files}")
        response = input("Перезаписать существующие файлы? (y/n): ").strip().lower()
        if response != 'y':
            print("Обучение отменено.")
            return
    
    # Загружаем датасет
    print("Загружаем изображения из папки images...")
    X, y, class_mapping = load_dataset_fast('images')
    
    if len(X) == 0:
        print("Ошибка: Датасет пуст!")
        return
    
    print(f"Загружено {len(X)} изображений")
    
    # Обучаем модель
    classifier, scaler = train_classifier_fast(X, y, max_epochs=15)
    
    if classifier is not None and scaler is not None:
        # Сохраняем модель и маппинг классов
        joblib.dump(classifier, 'eye_classifier.pkl')
        joblib.dump(scaler, 'eye_scaler.pkl')
        joblib.dump(class_mapping, 'class_mapping.pkl')
        print("\n✅ Модель успешно обучена и сохранена!")
        print("Файлы:")
        print("  - eye_classifier.pkl (модель)")
        print("  - eye_scaler.pkl (нормализатор)")
        print("  - class_mapping.pkl (маппинг классов)")

        # Новая часть: предложить быстрое демо инференса
        try:
            resp = input("Запустить быстрое демо на sample.jpg? (y/n): ").strip().lower()
        except Exception:
            resp = 'n'
        if resp == 'y':
            # используем только что обученную модель/скейлер в памяти
            run_inference_demo(classifier, scaler, class_mapping)
    else:
        print("❌ Ошибка при обучении модели!")


if __name__ == "__main__":
    main()
