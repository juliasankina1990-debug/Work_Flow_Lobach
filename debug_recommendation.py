#!/usr/bin/env python3
"""
Детальная диагностика рекомендательной системы
"""

import sys
import traceback
from lightfm.data import Dataset
from lightfm import LightFM
import numpy as np

def debug_lightfm_installation():
    """Проверка установки LightFM"""
    print("\n" + "="*50)
    print("🔍 ДИАГНОСТИКА LightFM")
    print("="*50)
    
    try:
        import lightfm
        print(f"✅ LightFM импортирован")
        print(f"   Версия: {lightfm.__version__}")
        print(f"   Путь: {lightfm.__file__}")
    except ImportError as e:
        print(f"❌ Ошибка импорта: {e}")
        return False
    
    return True

def debug_dataset_creation():
    """Детальная проверка создания Dataset"""
    print("\n" + "="*50)
    print("🔍 ТЕСТ СОЗДАНИЯ DATASET")
    print("="*50)
    
    try:
        dataset = Dataset()
        print("✅ Dataset создан")
        
        # Тест 1: Простейший fit
        print("\n📌 Тест 1: fit без признаков")
        try:
            dataset.fit(['u1'], ['i1'])
            print("   ✅ fit без признаков - OK")
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
            print(traceback.format_exc())
        
        # Тест 2: fit с признаками в разных форматах
        print("\n📌 Тест 2: fit с признаками (список кортежей)")
        try:
            user_features = [('u1', ['dept_it'])]
            item_features = [('i1', ['type_news'])]
            dataset.fit(['u1'], ['i1'], 
                       user_features=user_features,
                       item_features=item_features)
            print("   ✅ fit с признаками (кортежи) - OK")
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
            print(f"   Тип ошибки: {type(e).__name__}")
            print(f"   Формат данных: {type(user_features)} -> {type(user_features[0])}")
        
        # Тест 3: fit с признаками (список списков)
        print("\n📌 Тест 3: fit с признаками (список списков)")
        try:
            user_features = [['dept_it']]
            item_features = [['type_news']]
            dataset.fit(['u1'], ['i1'], 
                       user_features=user_features,
                       item_features=item_features)
            print("   ✅ fit с признаками (списки) - OK")
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
            print(f"   Тип ошибки: {type(e).__name__}")
        
        return dataset
        
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        traceback.print_exc()
        return None

def debug_feature_building():
    """Проверка построения матриц признаков"""
    print("\n" + "="*50)
    print("🔍 ТЕСТ ПОСТРОЕНИЯ МАТРИЦ")
    print("="*50)
    
    try:
        dataset = Dataset()
        dataset.fit(['u1', 'u2'], ['i1', 'i2'])
        
        # Тест с разными форматами
        test_cases = [
            (
                "Правильный формат (кортежи)",
                [('u1', ['dept_it']), ('u2', ['dept_hr'])],
                [('i1', ['type_news']), ('i2', ['type_article'])]
            ),
            (
                "Неправильный формат (списки)",
                [['dept_it'], ['dept_hr']],
                [['type_news'], ['type_article']]
            ),
            (
                "Смешанный формат",
                [('u1', ['dept_it']), ['dept_hr']],
                [('i1', ['type_news']), ['type_article']]
            )
        ]
        
        for test_name, user_feats, item_feats in test_cases:
            print(f"\n📌 Тест: {test_name}")
            try:
                user_matrix = dataset.build_user_features(user_feats)
                item_matrix = dataset.build_item_features(item_feats)
                print(f"   ✅ Успешно")
                print(f"   user_matrix тип: {type(user_matrix)}")
                print(f"   user_matrix[0] тип: {type(user_matrix[0]) if user_matrix else 'None'}")
            except Exception as e:
                print(f"   ❌ Ошибка: {e}")
                print(f"   Тип ошибки: {type(e).__name__}")
                print(f"   Формат user_features: {type(user_feats)} -> {type(user_feats[0]) if user_feats else 'None'}")
                if user_feats and not isinstance(user_feats[0], tuple):
                    print(f"   Содержимое: {user_feats[0]}")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        traceback.print_exc()

def debug_model_training():
    """Проверка обучения модели"""
    print("\n" + "="*50)
    print("🔍 ТЕСТ ОБУЧЕНИЯ МОДЕЛИ")
    print("="*50)
    
    try:
        # Создаем минимальные данные
        dataset = Dataset()
        dataset.fit(['u1'], ['i1'])
        
        # Создаем взаимодействие
        interactions = dataset.build_interactions([('u1', 'i1', 1.0)])[0]
        print(f"✅ Матрица взаимодействий создана: {interactions.shape}")
        
        # Пробуем обучить модель
        model = LightFM(no_components=5, random_state=42)
        
        try:
            model.fit(interactions, epochs=1)
            print("✅ Модель обучена без признаков")
        except Exception as e:
            print(f"❌ Ошибка обучения без признаков: {e}")
        
        # Добавляем признаки
        user_features = dataset.build_user_features([('u1', ['dept_it'])])[0]
        item_features = dataset.build_item_features([('i1', ['type_news'])])[0]
        
        try:
            model = LightFM(no_components=5, random_state=42)
            model.fit(interactions, 
                     user_features=user_features,
                     item_features=item_features,
                     epochs=1)
            print("✅ Модель обучена с признаками")
        except Exception as e:
            print(f"❌ Ошибка обучения с признаками: {e}")
            traceback.print_exc()
            
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        traceback.print_exc()

def debug_with_real_data():
    """Проверка с реальными данными из БД"""
    print("\n" + "="*50)
    print("🔍 ТЕСТ С РЕАЛЬНЫМИ ДАННЫМИ")
    print("="*50)
    
    try:
        from app import app
        from models import User, Item
        
        with app.app_context():
            users = User.query.all()
            items = Item.query.all()
            
            print(f"📊 Данные из БД:")
            print(f"   Пользователей: {len(users)}")
            print(f"   Материалов: {len(items)}")
            
            # Показываем первые несколько записей
            print("\n📋 Примеры пользователей:")
            for user in users[:3]:
                print(f"   User {user.id}: {user.name} | {user.department} | {user.position}")
            
            print("\n📋 Примеры материалов:")
            for item in items[:3]:
                tags = item.get_tags_list() if item.tags else []
                print(f"   Item {item.id}: {item.title[:30]}... | {item.type} | {tags}")
            
            # Создаем признаки в правильном формате
            print("\n🔄 Создание признаков...")
            
            user_features = []
            for user in users:
                feats = []
                if user.department:
                    feats.append(f"dept_{user.department}")
                if user.position:
                    feats.append(f"pos_{user.position}")
                user_features.append((f"user_{user.id}", feats))
                print(f"   user_{user.id}: {feats}")
            
            item_features = []
            for item in items:
                feats = []
                if item.type:
                    feats.append(f"type_{item.type}")
                if item.department:
                    feats.append(f"dept_{item.department}")
                if item.tags:
                    for tag in item.get_tags_list():
                        if tag and tag.strip():
                            feats.append(f"tag_{tag.strip()}")
                item_features.append((f"item_{item.id}", feats))
                print(f"   item_{item.id}: {feats}")
            
            # Пробуем создать Dataset
            print("\n🔄 Создание Dataset...")
            dataset = Dataset()
            
            try:
                user_ids = [f"user_{u.id}" for u in users]
                item_ids = [f"item_{i.id}" for i in items]
                
                print(f"   user_ids: {user_ids[:3]}...")
                print(f"   item_ids: {item_ids[:3]}...")
                
                dataset.fit(user_ids, item_ids, 
                           user_features=user_features,
                           item_features=item_features)
                print("✅ Dataset.fit успешно выполнен")
                
                # Пробуем построить матрицы
                print("\n🔄 Построение матриц признаков...")
                user_matrix = dataset.build_user_features(user_features)[0]
                item_matrix = dataset.build_item_features(item_features)[0]
                print(f"✅ Матрицы созданы: user={user_matrix.shape}, item={item_matrix.shape}")
                
            except Exception as e:
                print(f"❌ Ошибка: {e}")
                print(f"   Тип: {type(e).__name__}")
                print("\n📝 Детальная информация об ошибке:")
                traceback.print_exc()
                
                # Анализируем данные
                print("\n🔍 Анализ данных:")
                print(f"   user_features тип: {type(user_features)}")
                print(f"   user_features[0] тип: {type(user_features[0])}")
                print(f"   user_features[0][0] тип: {type(user_features[0][0])}")
                print(f"   user_features[0][1] тип: {type(user_features[0][1])}")
                print(f"   user_features[0][1] содержимое: {user_features[0][1]}")
                
    except Exception as e:
        print(f"❌ Ошибка доступа к БД: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    print("\n" + "🚀"*10)
    print("🚀 ЗАПУСК ДЕТАЛЬНОЙ ДИАГНОСТИКИ")
    print("🚀"*10 + "\n")
    
    if debug_lightfm_installation():
        debug_dataset_creation()
        debug_feature_building()
        debug_model_training()
        debug_with_real_data()
    
    print("\n" + "="*50)
    print("✅ ДИАГНОСТИКА ЗАВЕРШЕНА")
    print("="*50)
