import random
from datetime import datetime, timedelta
from faker import Faker
from app import app, db
from models import User, Item, Interaction

fake = Faker('ru_RU')

# ПОЛНЫЙ список пользователей
# ПОЛНЫЙ список пользователей (должен быть таким)
USERS_DATA = [
    {
        "name": "Алексей Волков",
        "position": "Senior Python Developer",
        "department": "Разработка",
        "avatar": "👨‍💻",
        "color": "#3498DB",
        "projects": "Внутренний портал, API интеграции",
        "hire_date": "15.06.2022",
        "role": "user"
    },
    {
        "name": "Елена Морозова",
        "position": "HR Business Partner",
        "department": "Управление персоналом",
        "avatar": "👩‍💼",
        "color": "#27AE60",
        "projects": "Онбординг, Оценка персонала",
        "hire_date": "10.03.2021",
        "role": "user"
    },
    {
        "name": "Дмитрий Соколов",
        "position": "Системный аналитик",
        "department": "Аналитика",
        "avatar": "👨‍🔬",
        "color": "#9B59B6",
        "projects": "Аналитика данных, ТЗ для разработки",
        "hire_date": "22.11.2022",
        "role": "user"
    },
    {
        "name": "Мария Крылова",
        "position": "Маркетолог",
        "department": "Маркетинг",
        "avatar": "👩‍🎨",
        "color": "#E67E22",
        "projects": "SMM, Контент-маркетинг",
        "hire_date": "05.04.2023",
        "role": "user"
    },
    {
        "name": "Иван Новиков",
        "position": "Новый сотрудник (стажер)",
        "department": "Разработка",
        "avatar": "🧑‍🎓",
        "color": "#95A5A6",
        "projects": "Внутренний портал",
        "hire_date": "01.03.2024",
        "role": "user"
    },
    {
        "name": "Демонстратор",
        "position": "Администратор",
        "department": "IT",
        "avatar": "👨‍💻",
        "color": "#E74C3C",
        "projects": "Все проекты",
        "hire_date": "01.01.2020",
        "role": "admin"
    }
]

# ПОЛНЫЙ список материалов
ITEMS_DATA = [
    {
        "title": "Хакатон 2024: итоги и победители",
        "type": "Новость",
        "type_icon": "📄",
        "tags": "мероприятие,тимбилдинг",
        "department": "Общие",
        "department_owner": "HR",
        "preview": "В минувшие выходные прошел ежегодный хакатон, в котором приняли участие 15 команд. Победители получили ценные призы и возможность реализовать свой проект.",
        "author": "Отдел мероприятий"
    },
    {
        "title": "Руководство для новых сотрудников",
        "type": "Документация",
        "type_icon": "📚",
        "tags": "онбординг,правила",
        "department": "HR",
        "department_owner": "HR",
        "preview": "Все, что нужно знать в первый рабочий день: от настройки рабочего места до знакомства с коллегами.",
        "author": "HR отдел"
    },
    {
        "title": "Открыта вакансия: Middle Python Developer",
        "type": "Вакансия",
        "type_icon": "💼",
        "tags": "найм,middle",
        "department": "Разработка",
        "department_owner": "Разработка",
        "preview": "Ищем разработчика в команду бэкенда для работы над высоконагруженными проектами.",
        "author": "Отдел разработки"
    },
    {
        "title": "Квартальный отчет по продуктам",
        "type": "Отчет",
        "type_icon": "📊",
        "tags": "аналитика,дашборды",
        "department": "Аналитика",
        "department_owner": "Аналитика",
        "preview": "Ключевые метрики и выводы за Q1 2024 года.",
        "author": "Аналитический отдел"
    },
    {
        "title": "Чистая архитектура на Python",
        "type": "Документация",
        "type_icon": "📄",
        "tags": "python,бэкенд,архитектура",
        "department": "Разработка",
        "department_owner": "Разработка",
        "preview": "Почему важно разделять слои приложения и как это правильно сделать.",
        "author": "Отдел разработки"
    },
    {
        "title": "Банды четырех: паттерны проектирования",
        "type": "Книга",
        "type_icon": "📚",
        "tags": "паттерны,ООП",
        "department": "Разработка",
        "department_owner": "Разработка",
        "preview": "Классика, которую должен знать каждый разработчик.",
        "author": "Технический отдел"
    },
    {
        "title": "Бесплатный доступ к курсам Stepik",
        "type": "Обучение",
        "type_icon": "🎓",
        "tags": "курсы,повышение_квалификации",
        "department": "Общие",
        "department_owner": "HR",
        "preview": "Компания оплачивает обучение по Python, SQL и алгоритмам.",
        "author": "HR отдел"
    },
    {
        "title": "Настройка CI/CD для своих проектов",
        "type": "Инструменты",
        "type_icon": "🔧",
        "tags": "git,ci/cd",
        "department": "Разработка",
        "department_owner": "Разработка",
        "preview": "Гайд по GitLab CI для начинающих: от простого pipeline до полного деплоя.",
        "author": "DevOps команда"
    }
]

def generate_data():
    with app.app_context():
        # Сначала создаем все таблицы
        print("Создаем таблицы в базе данных...")
        db.create_all()
        print("Таблицы созданы успешно")
        
        # Очищаем существующие данные
        print("Очищаем существующие данные...")
        db.session.query(Interaction).delete()
        db.session.query(Item).delete()
        db.session.query(User).delete()
        db.session.commit()
        
        print("Создаем пользователей...")
        users = []
        for user_data in USERS_DATA:
            user = User(**user_data)
            db.session.add(user)
            users.append(user)
        db.session.commit()
        print(f"Создано {len(users)} пользователей")
        
        print("Создаем материалы...")
        items = []
        for item_data in ITEMS_DATA:
            # Генерируем контент через faker
            item = Item(
                title=item_data['title'],
                type=item_data['type'],
                type_icon=item_data['type_icon'],
                tags=item_data['tags'],
                date=(datetime.now() - timedelta(days=random.randint(1, 30))).strftime('%Y-%m-%d'),
                department=item_data['department'],
                department_owner=item_data['department_owner'],
                preview=item_data['preview'],
                content=fake.text(max_nb_chars=2000),
                views=random.randint(10, 500),
                saved_count=random.randint(5, 100),
                author=item_data['author']
            )
            db.session.add(item)
            items.append(item)
        db.session.commit()
        print(f"Создано {len(items)} материалов")
        
        print("Создаем взаимодействия...")
        interactions = []
        actions = ['read', 'save', 'view']
        
        for user in users[:-1]:  # Для всех кроме админа
            # Каждый пользователь взаимодействует с 3-6 материалами
            num_interactions = random.randint(3, 6)
            selected_items = random.sample(items, min(num_interactions, len(items)))
            
            for item in selected_items:
                # Несколько действий с одним материалом
                num_actions = random.randint(1, 3)
                selected_actions = random.sample(actions, min(num_actions, len(actions)))
                
                for action in selected_actions:
                    interaction = Interaction(
                        user_id=user.id,
                        item_id=item.id,
                        action=action,
                        created_at=fake.date_time_between(start_date='-14d', end_date='now')
                    )
                    db.session.add(interaction)
                    interactions.append(interaction)
        
        db.session.commit()
        print(f"Создано {len(interactions)} взаимодействий")
        
        print("\n=== ИТОГИ ГЕНЕРАЦИИ ===")
        print(f"Пользователей: {User.query.count()}")
        print(f"Материалов: {Item.query.count()}")
        print(f"Взаимодействий: {Interaction.query.count()}")

if __name__ == "__main__":
    generate_data()
