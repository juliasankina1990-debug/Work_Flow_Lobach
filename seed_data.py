#!/usr/bin/env python3
"""
Скрипт для наполнения базы тестовыми данными.
"""
import random
from datetime import datetime, timedelta
from app import app
from models import db, User, Item, Interaction
from recommendation import update_recommender

# Тестовые пользователи
users_data = [
    {
        'username': 'ivan_dev',
        'name': 'Иван Петров',
        'position': 'Senior Python Developer',
        'department': 'Разработка',
        'avatar': '👨‍💻',
        'color': '#3498db',
        'projects': 'WorkFlow,AuthService',
        'hire_date': '2022',
        'role': 'admin',
        'password': 'admin123'
    },
    {
        'username': 'maria_hr',
        'name': 'Мария Иванова',
        'position': 'HR Business Partner',
        'department': 'Управление персоналом',
        'avatar': '👩‍💼',
        'color': '#e74c3c',
        'projects': 'Recruiting,Onboarding',
        'hire_date': '2023',
        'role': 'user',
        'password': 'user123'
    },
    {
        'username': 'alex_analyst',
        'name': 'Алексей Смирнов',
        'position': 'Data Analyst',
        'department': 'Аналитика',
        'avatar': '👨‍🔬',
        'color': '#2ecc71',
        'projects': 'Analytics Dashboard,Reports',
        'hire_date': '2023',
        'role': 'user',
        'password': 'user123'
    }
]

# Список материалов для добавления
materials_data = [
    {
        'title': 'Руководство по стилю кода Python (PEP 8)',
        'type': 'Документация',
        'type_icon': '📄',
        'tags': ['python', 'pep8', 'стиль'],
        'department': 'Разработка',
        'department_owner': 'Разработка',
        'preview': 'Основные правила оформления кода на Python, принятые в компании.',
        'content': 'Полное руководство по PEP 8 с примерами и исключениями...',
        'author': 'Отдел разработки'
    },
    {
        'title': 'Введение в Docker для разработчиков',
        'type': 'Обучение',
        'type_icon': '🐳',
        'tags': ['docker', 'контейнеры', 'devops'],
        'department': 'Разработка',
        'department_owner': 'IT',
        'preview': 'Базовые concepts Docker, создание образов, docker-compose.',
        'content': 'Материалы внутреннего воркшопа по Docker...',
        'author': 'DevOps команда'
    },
    {
        'title': 'План развития для Junior Python разработчика',
        'type': 'Карьера',
        'type_icon': '📈',
        'tags': ['карьера', 'обучение', 'junior'],
        'department': 'HR',
        'department_owner': 'HR',
        'preview': 'Индивидуальный план обучения и развития для начинающих разработчиков.',
        'content': 'Список тем, проектов и сроков для перехода на уровень Middle...',
        'author': 'HR отдел'
    },
    {
        'title': 'Анализ продаж за Q1 2024',
        'type': 'Отчёт',
        'type_icon': '📊',
        'tags': ['аналитика', 'продажи', 'отчёт'],
        'department': 'Аналитика',
        'department_owner': 'Аналитика',
        'preview': 'Ключевые метрики и выводы по итогам первого квартала.',
        'content': 'Подробный отчёт с графиками и рекомендациями...',
        'author': 'Аналитический отдел'
    },
    {
        'title': 'Вакансия: Middle Python Developer',
        'type': 'Вакансия',
        'type_icon': '💼',
        'tags': ['вакансия', 'python', 'middle'],
        'department': 'Разработка',
        'department_owner': 'HR',
        'preview': 'Ищем разработчика в команду бэкенда для работы над новым проектом.',
        'content': 'Требования: опыт от 3 лет, знание Django/FastAPI, SQL...',
        'author': 'HR отдел'
    },
    {
        'title': 'Как провести эффективное совещание',
        'type': 'Статья',
        'type_icon': '📝',
        'tags': ['soft skills', 'совещания', 'продуктивность'],
        'department': 'Общие',
        'department_owner': 'HR',
        'preview': 'Практические советы по организации и проведению встреч.',
        'content': 'Чек-лист подготовки, роли участников, работа с возражениями...',
        'author': 'Тренинг-центр'
    },
    {
        'title': 'Основы Git: от коммита до merge request',
        'type': 'Обучение',
        'type_icon': '📚',
        'tags': ['git', 'version control', 'обучение'],
        'department': 'Разработка',
        'department_owner': 'Разработка',
        'preview': 'Базовый курс по Git для новых сотрудников.',
        'content': 'Видео-уроки и практические задания по работе с Git...',
        'author': 'Отдел разработки'
    },
    {
        'title': 'Новости компании: открытие офиса в Казани',
        'type': 'Новость',
        'type_icon': '📰',
        'tags': ['новость', 'офис', 'казань'],
        'department': 'Общие',
        'department_owner': 'PR',
        'preview': 'Мы расширяемся! В августе открывается новый офис в Иннополисе.',
        'content': 'Подробности переезда, условия работы и как подать заявку...',
        'author': 'PR отдел'
    },
    {
        'title': 'Шаблон технической документации проекта',
        'type': 'Шаблон',
        'type_icon': '📋',
        'tags': ['документация', 'шаблон', 'arch'],
        'department': 'Разработка',
        'department_owner': 'Архитектура',
        'preview': 'Единый стандарт оформления тех. документации.',
        'content': 'Markdown шаблон с разделами: архитектура, API, развёртывание...',
        'author': 'Архитектурный комитет'
    },
    {
        'title': 'Книга «Чистый Python» (рекомендации)',
        'type': 'Книга',
        'type_icon': '📕',
        'tags': ['python', 'книга', 'best practices'],
        'department': 'Разработка',
        'department_owner': 'Разработка',
        'preview': 'Конспект ключевых идей из книги для внутреннего использования.',
        'content': 'Выдержки и примеры кода из книги «Clean Python»...',
        'author': 'Библиотека знаний'
    },
    {
        'title': 'Дашборды в Tableau: базовый курс',
        'type': 'Обучение',
        'type_icon': '📊',
        'tags': ['tableau', 'дашборды', 'аналитика'],
        'department': 'Аналитика',
        'department_owner': 'Аналитика',
        'preview': 'Видео-уроки по созданию интерактивных дашбордов.',
        'content': 'Доступ к записям вебинаров и материалам...',
        'author': 'Аналитический отдел'
    },
    {
        'title': 'Политика информационной безопасности',
        'type': 'Документ',
        'type_icon': '🔒',
        'tags': ['безопасность', 'политика', 'it'],
        'department': 'IT',
        'department_owner': 'IT',
        'preview': 'Основные правила и требования по кибербезопасности.',
        'content': 'Пароли, двухфакторная аутентификация, работа с VPN...',
        'author': 'Отдел ИБ'
    },
    {
        'title': 'Инструменты для удалённой работы: обзор',
        'type': 'Статья',
        'type_icon': '🛠️',
        'tags': ['remote', 'инструменты', 'collaboration'],
        'department': 'Общие',
        'department_owner': 'HR',
        'preview': 'Сравнение Zoom, Teams, Slack, Miro и других.',
        'content': 'Плюсы и минусы, рекомендации по выбору...',
        'author': 'HR отдел'
    },
    {
        'title': 'Регулярные выражения для начинающих',
        'type': 'Обучение',
        'type_icon': '🔍',
        'tags': ['regex', 'обучение', 'python'],
        'department': 'Разработка',
        'department_owner': 'Разработка',
        'preview': 'Шпаргалка и практические упражнения по regex.',
        'content': 'Примеры использования в Python и командной строке...',
        'author': 'Отдел разработки'
    },
    {
        'title': 'Итоги хакатона 2024',
        'type': 'Новость',
        'type_icon': '🏆',
        'tags': ['хакатон', 'мероприятие', 'итоги'],
        'department': 'Общие',
        'department_owner': 'HR',
        'preview': 'Победители, проекты и фотоотчёт с мероприятия.',
        'content': 'Список команд, призы, планы на следующий хакатон...',
        'author': 'Оргкомитет'
    }
]

def seed_users():
    """Создает тестовых пользователей"""
    with app.app_context():
        existing = User.query.count()
        if existing > 0:
            print(f"В базе уже есть {existing} пользователей. Пропускаем создание.")
            return False
        
        print("Создаем пользователей...")
        for data in users_data:
            password = data.pop('password')
            user = User(**data)
            user.set_password(password)
            db.session.add(user)
        
        db.session.commit()
        print(f"Создано {len(users_data)} пользователей.")
        return True

def seed_materials():
    """Добавляет материалы в БД"""
    with app.app_context():
        existing = Item.query.count()
        if existing > 0:
            print(f"В базе уже есть {existing} материалов. Пропускаем добавление.")
            return False
        
        print("Добавляем материалы...")
        for data in materials_data:
            item = Item(**data)
            db.session.add(item)
        db.session.commit()
        print(f"Добавлено {len(materials_data)} материалов.")
        return True

def seed_interactions():
    """Создаёт случайные взаимодействия"""
    with app.app_context():
        users = User.query.all()
        items = Item.query.all()
        if not users or not items:
            print("Нет пользователей или материалов для создания взаимодействий.")
            return
        
        actions = ['view', 'read', 'save', 'share']
        weights = {'view': 1, 'read': 3, 'save': 5, 'share': 10
