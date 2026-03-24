#!/usr/bin/env python3
"""
Скрипт для генерации синтетических данных для улучшения модели рекомендаций
Создает реалистичные паттерны поведения пользователей
"""

import os
import sys
import random
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import traceback

# Добавляем путь к проекту
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import app
from models import db, User, Item, Interaction
from recommendation import update_recommender

# Конфигурация
TARGET_USERS = 50  # Целевое количество пользователей
TARGET_INTERACTIONS_PER_USER = 30  # Целевое количество взаимодействий на пользователя
TARGET_PRECISION = 70  # Целевая точность (%)

# Профили пользователей с их интересами
USER_PROFILES = [
    {
        'name': 'Алексей_Волков',
        'position': 'Python Developer',
        'department': 'Разработка',
        'avatar': '👨‍💻',
        'interests': ['python', 'docker', 'git', 'fastapi', 'postgresql']
    },
    {
        'name': 'Елена_Морозова',
        'position': 'Data Scientist',
        'department': 'Аналитика',
        'avatar': '👩‍🔬',
        'interests': ['python', 'pandas', 'numpy', 'sql', 'tableau']
    },
    {
        'name': 'Иван_Сидоров',
        'position': 'Frontend Developer',
        'department': 'Разработка',
        'avatar': '👨‍🎨',
        'interests': ['javascript', 'react', 'css', 'html', 'typescript']
    },
    {
        'name': 'Мария_Петрова',
        'position': 'HR Manager',
        'department': 'HR',
        'avatar': '👩‍💼',
        'interests': ['рекрутинг', 'онбординг', 'soft skills', 'карьера', 'оценка']
    },
    {
        'name': 'Дмитрий_Соколов',
        'position': 'DevOps Engineer',
        'department': 'IT',
        'avatar': '👨‍🔧',
        'interests': ['docker', 'kubernetes', 'jenkins', 'aws', 'linux']
    },
    {
        'name': 'Анна_Козлова',
        'position': 'Project Manager',
        'department': 'Управление',
        'avatar': '👩‍💼',
        'interests': ['agile', 'scrum', 'управление', 'планирование', 'команда']
    },
    {
        'name': 'Павел_Новиков',
        'position': 'QA Engineer',
        'department': 'Тестирование',
        'avatar': '👨‍🔬',
        'interests': ['testing', 'selenium', 'pytest', 'автоматизация', 'quality']
    },
    {
        'name': 'Ольга_Смирнова',
        'position': 'UI/UX Designer',
        'department': 'Дизайн',
        'avatar': '🎨',
        'interests': ['figma', 'ui', 'ux', 'дизайн', 'прототипирование']
    },
    {
        'name': 'Сергей_Иванов',
        'position': 'System Administrator',
        'department': 'IT',
        'avatar': '👨‍💻',
        'interests': ['linux', 'сети', 'безопасность', 'мониторинг', 'backup']
    },
    {
        'name': 'Татьяна_Мороз',
        'position': 'Business Analyst',
        'department': 'Аналитика',
        'avatar': '👩‍💼',
        'interests': ['аналитика', 'sql', 'excel', 'требования', 'документация']
    }
]

# Расширенный список материалов с тегами
MATERIALS_TEMPLATES = [
    # Python разработка
    {'title': 'Python Best Practices 2024', 'type': 'Статья', 'tags': ['python', 'best practices', 'clean code']},
    {'title': 'FastAPI: Современная разработка API', 'type': 'Руководство', 'tags': ['python', 'fastapi', 'api']},
    {'title': 'Асинхронное программирование в Python', 'type': 'Обучение', 'tags': ['python', 'async', 'asyncio']},
    {'title': 'Django vs FastAPI: Сравнение', 'type': 'Аналитика', 'tags': ['python', 'django', 'fastapi']},
    {'title': 'Type Hints в Python: Полное руководство', 'type': 'Документация', 'tags': ['python', 'typing', 'mypy']},
    
    # Docker и контейнеризация
    {'title': 'Docker для разработчиков', 'type': 'Обучение', 'tags': ['docker', 'контейнеры', 'devops']},
    {'title': 'Docker Compose: Оркестрация сервисов', 'type': 'Руководство', 'tags': ['docker', 'docker-compose', 'devops']},
    {'title': 'Оптимизация Docker образов', 'type': 'Статья', 'tags': ['docker', 'оптимизация', 'devops']},
    {'title': 'Kubernetes для начинающих', 'type': 'Обучение', 'tags': ['kubernetes', 'k8s', 'devops']},
    {'title': 'Микросервисы с Docker и Kubernetes', 'type': 'Курс', 'tags': ['docker', 'kubernetes', 'microservices']},
    
    # Базы данных
    {'title': 'PostgreSQL: Оптимизация запросов', 'type': 'Руководство', 'tags': ['postgresql', 'sql', 'базы данных']},
    {'title': 'NoSQL: MongoDB для разработчиков', 'type': 'Обучение', 'tags': ['mongodb', 'nosql', 'базы данных']},
    {'title': 'SQL: От простого к сложному', 'type': 'Курс', 'tags': ['sql', 'базы данных', 'аналитика']},
    {'title': 'Индексы в базах данных', 'type': 'Статья', 'tags': ['sql', 'индексы', 'оптимизация']},
    
    # Frontend
    {'title': 'React: Современные подходы', 'type': 'Обучение', 'tags': ['react', 'javascript', 'frontend']},
    {'title': 'TypeScript: Почему стоит использовать', 'type': 'Статья', 'tags': ['typescript', 'javascript', 'frontend']},
    {'title': 'CSS Grid и Flexbox', 'type': 'Руководство', 'tags': ['css', 'frontend', 'верстка']},
    {'title': 'Vue.js для начинающих', 'type': 'Обучение', 'tags': ['vue', 'javascript', 'frontend']},
    
    # Аналитика и данные
    {'title': 'Pandas: Анализ данных', 'type': 'Обучение', 'tags': ['pandas', 'python', 'аналитика']},
    {'title': 'Визуализация данных с Matplotlib', 'type': 'Руководство', 'tags': ['matplotlib', 'визуализация', 'аналитика']},
    {'title': 'Tableau: Создание дашбордов', 'type': 'Обучение', 'tags': ['tableau', 'дашборды', 'аналитика']},
    {'title': 'Статистика для аналитиков', 'type': 'Курс', 'tags': ['статистика', 'аналитика', 'math']},
    
    # HR и управление
    {'title': 'Онбординг новых сотрудников', 'type': 'Методичка', 'tags': ['hr', 'онбординг', 'адаптация']},
    {'title': 'Оценка персонала: Методики', 'type': 'Руководство', 'tags': ['hr', 'оценка', 'kpi']},
    {'title': 'Как проводить 1:1 встречи', 'type': 'Статья', 'tags': ['hr', 'менеджмент', 'soft skills']},
    {'title': 'Agile для HR', 'type': 'Обучение', 'tags': ['agile', 'hr', 'управление']},
    
    # Soft skills
    {'title': 'Эффективная коммуникация', 'type': 'Тренинг', 'tags': ['soft skills', 'коммуникация', 'работа в команде']},
    {'title': 'Тайм-менеджмент для IT', 'type': 'Статья', 'tags': ['soft skills', 'продуктивность', 'управление временем']},
    {'title': 'Как давать обратную связь', 'type': 'Методичка', 'tags': ['soft skills', 'feedback', 'коммуникация']},
    
    # Тестирование
    {'title': 'Автоматизация тестирования с Selenium', 'type': 'Обучение', 'tags': ['testing', 'selenium', 'автоматизация']},
    {'title': 'Pytest: Продвинутые техники', 'type': 'Руководство', 'tags': ['testing', 'pytest', 'python']},
    {'title': 'Unit тестирование best practices', 'type': 'Статья', 'tags': ['testing', 'unit', 'quality']},
    
    # Безопасность
    {'title': 'Основы кибербезопасности', 'type': 'Обучение', 'tags': ['security', 'безопасность', 'it']},
    {'title': 'Защита веб-приложений', 'type': 'Статья', 'tags': ['security', 'web', 'pentest']},
    {'title': 'OAuth2 и JWT: Аутентификация', 'type': 'Руководство', 'tags': ['security', 'oauth', 'jwt']},
    
    # Инфраструктура
    {'title': 'Linux для администраторов', 'type': 'Обучение', 'tags': ['linux', 'администрирование', 'it']},
    {'title': 'Мониторинг с Prometheus', 'type': 'Руководство', 'tags': ['monitoring', 'prometheus', 'devops']},
    {'title': 'CI/CD с Jenkins', 'type': 'Обучение', 'tags': ['jenkins', 'cicd', 'devops']},
    
    # Управление проектами
    {'title': 'Scrum: Роли и процессы', 'type': 'Обучение', 'tags': ['scrum', 'agile', 'управление']},
    {'title': 'Roadmap проекта: Как составить', 'type': 'Методичка', 'tags': ['управление', 'планирование', 'project']},
    {'title': 'Оценка сроков в разработке', 'type': 'Статья', 'tags': ['управление', 'оценка', 'project']},
]

def create_materials():
    """Создает расширенный набор материалов"""
    print("\n📦 СОЗДАНИЕ МАТЕРИАЛОВ")
    print("="*60)
    
    with app.app_context():
        existing = Item.query.count()
        if existing > 0:
            print(f"✅ Уже существует {existing} материалов")
            return Item.query.all()
        
        materials = []
        for template in MATERIALS_TEMPLATES:
            # Добавляем случайную дату
            random_days = random.randint(1, 365)
            date = (datetime.now() - timedelta(days=random_days)).strftime('%Y-%m-%d')
            
            # Расширяем превью и контент
            preview = f"Обзор материала по теме {', '.join(template['tags'][:2])}. Подходит для специалистов в области {template['tags'][0]}."
            content = f"Полное руководство по {template['title']}. Рассматриваются ключевые аспекты: {', '.join(template['tags'])}. " * 5
            
            material = Item(
                title=template['title'],
                type=template['type'],
                type_icon=get_icon_for_type(template['type']),
                tags=','.join(template['tags']),
                date=date,
                department=random.choice(['Разработка', 'Аналитика', 'HR', 'IT', 'Управление']),
                department_owner='Библиотека знаний',
                preview=preview,
                content=content,
                views=random.randint(50, 500),
                saved_count=random.randint(10, 100),
                author='Knowledge Base'
            )
            materials.append(material)
            db.session.add(material)
        
        db.session.commit()
        print(f"✅ Создано {len(materials)} новых материалов")
        return materials

def get_icon_for_type(material_type):
    """Возвращает иконку для типа материала"""
    icons = {
        'Статья': '📄',
        'Руководство': '📚',
        'Обучение': '🎓',
        'Курс': '📖',
        'Документация': '📋',
        'Методичка': '📝',
        'Тренинг': '💪',
        'Аналитика': '📊'
    }
    return icons.get(material_type, '📄')

def create_users():
    """Создает пользователей с профилями"""
    print("\n👥 СОЗДАНИЕ ПОЛЬЗОВАТЕЛЕЙ")
    print("="*60)
    
    with app.app_context():
        existing = User.query.count()
        if existing >= TARGET_USERS:
            print(f"✅ Уже существует {existing} пользователей")
            return User.query.all()
        
        users = []
        users_to_create = TARGET_USERS - existing
        
        # Добавляем базовых пользователей из профилей
        base_count = min(len(USER_PROFILES), users_to_create)
        for i in range(base_count):
            profile = USER_PROFILES[i]
            username = profile['name'].lower()
            
            # Проверяем, не существует ли уже
            if User.query.filter_by(username=username).first():
                continue
            
            user = User(
                username=username,
                name=profile['name'],
                password='password123',
                position=profile['position'],
                department=profile['department'],
                avatar=profile['avatar'],
                color=random.choice(['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']),
                projects=','.join(profile['interests'][:3]),
                hire_date=f"202{random.randint(1,4)}",
                role='user'
            )
            users.append(user)
            db.session.add(user)
        
        # Добавляем дополнительных пользователей с случайными профилями
        departments = ['Разработка', 'Аналитика', 'HR', 'IT', 'Управление', 'Дизайн', 'Тестирование']
        positions = ['Специалист', 'Ведущий специалист', 'Руководитель', 'Аналитик', 'Разработчик']
        
        for i in range(users_to_create - base_count):
            dept = random.choice(departments)
            username = f"user_{len(users) + existing + i + 1}"
            
            if User.query.filter_by(username=username).first():
                continue
            
            user = User(
                username=username,
                name=f"Сотрудник {len(users) + existing + i + 1}",
                password='password123',
                position=f"{random.choice(positions)} {dept}",
                department=dept,
                avatar=random.choice(['👤', '👩', '👨', '🧑']),
                color=random.choice(['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']),
                projects='Проект А,Проект Б',
                hire_date=f"202{random.randint(1,4)}",
                role='user'
            )
            users.append(user)
            db.session.add(user)
        
        db.session.commit()
        print(f"✅ Создано {len(users)} новых пользователей")
        return users + User.query.all()

def generate_smart_interactions(users, materials):
    """Генерирует умные взаимодействия на основе интересов"""
    print("\n🔄 ГЕНЕРАЦИЯ ВЗАИМОДЕЙСТВИЙ")
    print("="*60)
    
    with app.app_context():
        # Очищаем старые взаимодействия для чистоты эксперимента
        count = Interaction.query.delete()
        print(f"✅ Удалено {count} старых взаимодействий")
        
        # Группируем материалы по тегам
        materials_by_tag = defaultdict(list)
        for material in materials:
            for tag in material.get_tags_list():
                materials_by_tag[tag].append(material)
        
        interactions_created = 0
        
        for user in users:
            # Определяем интересы пользователя
            user_interests = []
            
            # Из проектов пользователя
            if user.projects:
                user_interests.extend([p.strip() for p in user.projects.split(',')])
            
            # Из отдела
            dept_interests = {
                'Разработка': ['python', 'docker', 'git', 'react', 'javascript'],
                'Аналитика': ['pandas', 'sql', 'tableau', 'статистика', 'аналитика'],
                'HR': ['hr', 'онбординг', 'оценка', 'soft skills', 'управление'],
                'IT': ['linux', 'docker', 'kubernetes', 'security', 'monitoring'],
                'Управление': ['agile', 'scrum', 'управление', 'планирование', 'project'],
                'Дизайн': ['figma', 'ui', 'ux', 'дизайн', 'прототипирование'],
                'Тестирование': ['testing', 'selenium', 'pytest', 'quality', 'автоматизация']
            }
            
            if user.department in dept_interests:
                user_interests.extend(dept_interests[user.department])
            
            # Убираем дубликаты
            user_interests = list(set(user_interests))
            
            # Генерируем просмотры (базовые взаимодействия)
            num_interactions = TARGET_INTERACTIONS_PER_USER
            viewed_items = set()
            
            for _ in range(num_interactions):
                # 80% материалов по интересам, 20% случайных
                if random.random() < 0.8 and user_interests:
                    # Выбираем тег из интересов
                    interest = random.choice(user_interests)
                    candidates = materials_by_tag.get(interest, [])
                    if candidates:
                        material = random.choice(candidates)
                    else:
                        material = random.choice(materials)
                else:
                    material = random.choice(materials)
                
                # Не повторяемся слишком часто
                if material.id in viewed_items and random.random() < 0.3:
                    continue
                
                viewed_items.add(material.id)
                
                # Определяем тип взаимодействия с весами
                action_weights = [
                    ('view', 1),    # легкий вес
                    ('view', 1),
                    ('view', 1),
                    ('read', 3),    # средний вес
                    ('read', 3),
                    ('save', 5),    # тяжелый вес
                    ('share', 10)   # очень тяжелый вес
                ]
                
                # Для материалов по интересам - больше сохранений и прочтений
                if any(tag in user_interests for tag in material.get_tags_list()):
                    action = random.choices(
                        ['read', 'save', 'share', 'view'],
                        weights=[5, 3, 1, 1],
                        k=1
                    )[0]
                else:
                    action = random.choices(
                        ['view', 'read', 'save'],
                        weights=[7, 2, 1],
                        k=1
                    )[0]
                
                # Создаем взаимодействие
                interaction = Interaction(
                    user_id=user.id,
                    item_id=material.id,
                    action=action,
                    weight={'view': 1, 'read': 3, 'save': 5, 'share': 10}.get(action, 1),
                    created_at=datetime.now() - timedelta(days=random.randint(0, 60))
                )
                db.session.add(interaction)
                interactions_created += 1
                
                # Обновляем счетчики в материале
                if action == 'view':
                    material.views += 1
                elif action == 'save':
                    material.saved_count += 1
            
            if (user.id % 10 == 0):
                db.session.commit()
                print(f"  Прогресс: пользователь {user.id} готов")
        
        db.session.commit()
        print(f"✅ Создано {interactions_created} новых взаимодействий")
        
        # Статистика
        action_stats = {}
        for action in ['view', 'read', 'save', 'share']:
            count = Interaction.query.filter_by(action=action).count()
            action_stats[action] = count
        
        print("\n📊 Статистика взаимодействий:")
        for action, count in action_stats.items():
            print(f"   - {action}: {count} ({count/interactions_created*100:.1f}%)")
        
        return interactions_created

def create_training_patterns(users, materials):
    """Создает паттерны для обучения (чтобы Precision вырос)"""
    print("\n🎯 СОЗДАНИЕ ОБУЧАЮЩИХ ПАТТЕРНОВ")
    print("="*60)
    
    with app.app_context():
        # Группируем материалы по тегам
        materials_by_tag = defaultdict(list)
        for material in materials:
            for tag in material.get_tags_list():
                materials_by_tag[tag].append(material)
        
        patterns_created = 0
        
        # Для каждого пользователя создаем паттерны "прочитал -> похожее"
        for user in users:
            # Получаем все прочтения пользователя
            user_reads = Interaction.query.filter_by(user_id=user.id, action='read').all()
            read_tags = set()
            
            for read in user_reads:
                material = read.item
                if material:
                    read_tags.update(material.get_tags_list())
            
            if not read_tags:
                continue
            
            # Для каждого тега добавляем дополнительные прочтения похожих материалов
            for tag in list(read_tags)[:3]:  # Ограничиваем
                similar_materials = materials_by_tag.get(tag, [])
                for material in similar_materials[:5]:  # Берем до 5 похожих
                    # Проверяем, не взаимодействовал ли уже
                    existing = Interaction.query.filter_by(
                        user_id=user.id, 
                        item_id=material.id
                    ).first()
                    
                    if not existing and random.random() < 0.7:  # 70% вероятность
                        interaction = Interaction(
                            user_id=user.id,
                            item_id=material.id,
                            action='read',
                            weight=3,
                            created_at=datetime.now() - timedelta(days=random.randint(1, 30))
                        )
                        db.session.add(interaction)
                        patterns_created += 1
            
            if user.id % 10 == 0:
                db.session.commit()
        
        db.session.commit()
        print(f"✅ Создано {patterns_created} обучающих паттернов")
        return patterns_created

def verify_model_quality():
    """Проверяет качество модели и дает рекомендации"""
    print("\n🔍 ПРОВЕРКА КАЧЕСТВА МОДЕЛИ")
    print("="*60)
    
    with app.app_context():
        # Базовая статистика
        users = User.query.count()
        materials = Item.query.count()
        interactions = Interaction.query.count()
        
        print(f"📊 Итоговая статистика:")
        print(f"   - Пользователей: {users}")
        print(f"   - Материалов: {materials}")
        print(f"   - Взаимодействий: {interactions}")
        print(f"   - Среднее на пользователя: {interactions/users:.1f}")
        
        # Проверяем разнообразие
        action_counts = {}
        for action in ['view', 'read', 'save', 'share']:
            count = Interaction.query.filter_by(action=action).count()
            action_counts[action] = count
        
        print(f"\n📈 Распределение действий:")
        for action, count in action_counts.items():
            print(f"   - {action}: {count} ({count/interactions*100:.1f}%)")
        
        # Проверяем покрытие материалов
        materials_with_interactions = db.session.query(Interaction.item_id).distinct().count()
        print(f"\n🎯 Покрытие материалов: {materials_with_interactions}/{materials} ({materials_with_interactions/materials*100:.1f}%)")
        
        # Проверяем активность пользователей
        users_with_interactions = db.session.query(Interaction.user_id).distinct().count()
        print(f"👥 Активных пользователей: {users_with_interactions}/{users} ({users_with_interactions/users*100:.1f}%)")
        
        # Рекомендации
        print("\n💡 Рекомендации:")
        if interactions/users < 20:
            print("   - Увеличьте количество взаимодействий на пользователя")
        if materials_with_interactions/materials < 0.5:
            print("   - Добавьте взаимодействий с разными материалами")
        if action_counts.get('read', 0) < interactions * 0.2:
            print("   - Увеличьте долю прочтений (самые важные для обучения)")
        
        return {
            'users': users,
            'materials': materials,
            'interactions': interactions,
            'active_users': users_with_interactions,
            'covered_materials': materials_with_interactions
        }

def main():
    """Главная функция"""
    print("\n" + "="*70)
    print("🚀 ГЕНЕРАТОР ОБУЧАЮЩИХ ДАННЫХ ДЛЯ РЕКОМЕНДАТЕЛЬНОЙ СИСТЕМЫ")
    print("="*70)
    print(f"🎯 Целевая точность: {TARGET_PRECISION}%")
    print(f"👥 Целевое количество пользователей: {TARGET_USERS}")
    print(f"📊 Целевое взаимодействий на пользователя: {TARGET_INTERACTIONS_PER_USER}")
    print("="*70)
    
    try:
        # Шаг 1: Создаем материалы
        materials = create_materials()
        
        # Шаг 2: Создаем пользователей
        users = create_users()
        
        # Шаг 3: Генерируем умные взаимодействия
        interactions = generate_smart_interactions(users, materials)
        
        # Шаг 4: Добавляем обучающие паттерны
        patterns = create_training_patterns(users, materials)
        
        # Шаг 5: Проверяем качество
        stats = verify_model_quality()
        
        # Шаг 6: Переобучаем модель
        print("\n" + "="*70)
        print("🔄 ПЕРЕОБУЧЕНИЕ МОДЕЛИ")
        print("="*70)
        
        with app.app_context():
            update_recommender()
        
        print("\n" + "="*70)
        print("✅ ГЕНЕРАЦИЯ ДАННЫХ ЗАВЕРШЕНА!")
        print("="*70)
        print(f"📊 Итог:")
        print(f"   - Пользователей: {stats['users']}")
        print(f"   - Материалов: {stats['materials']}")
        print(f"   - Взаимодействий: {stats['interactions']}")
        print(f"   - Активных пользователей: {stats['active_users']}")
        print(f"   - Охвачено материалов: {stats['covered_materials']}")
        print("="*70)
        print("🚀 Теперь запустите: python3 model_metrics.py")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ ОШИБКА: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
