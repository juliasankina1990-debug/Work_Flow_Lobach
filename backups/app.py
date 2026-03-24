from recommendation import recommender, init_recommender, update_recommender
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from config import Config
from models import db, User, Item, Interaction
from datetime import datetime
import random
import threading

app = Flask(__name__)
app.config.from_object(Config)

# Инициализация базы данных
db.init_app(app)

# Глобальные переменные для рекомендательной системы
new_interactions_count = 0
RETRAIN_THRESHOLD = 20
training_lock = threading.Lock()

# Создание таблиц и инициализация рекомендательной системы при запуске
with app.app_context():
    db.create_all()
    # Инициализируем рекомендательную систему внутри контекста
    init_recommender()

def update_recommender_with_context():
    """Обёртка для переобучения с блокировкой"""
    global new_interactions_count, training_lock
    try:
        with app.app_context():
            update_recommender()
    finally:
        training_lock.release()

# Маршруты
@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login_page'))
    user = db.session.get(User, session['user_id'])
    if not user:
        return redirect(url_for('login_page'))
    return render_template('index.html', user=user)

@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if request.method == 'POST':
        data = request.get_json()
        if data and 'user_id' in data:
            session['user_id'] = data['user_id']
            return jsonify({'status': 'ok', 'redirect': url_for('index')})

    users = User.query.all()
    return render_template('login.html', users={u.id: u.to_dict() for u in users})

@app.route('/logout')
def logout_page():
    session.pop('user_id', None)
    return redirect(url_for('login_page'))

@app.route('/material/<int:id>')
def material_page(id):
    if 'user_id' not in session:
        return redirect(url_for('login_page'))

    user = db.session.get(User, session['user_id'])
    material = db.session.get(Item, id)

    if not material:
        return "Материал не найден", 404

    # Увеличиваем счетчик просмотров
    material.views += 1
    db.session.commit()

    # Записываем взаимодействие
    interaction = Interaction(
        user_id=user.id,
        item_id=material.id,
        action='view'
    )
    db.session.add(interaction)
    db.session.commit()

    return render_template('material.html', user=user, material=material)

@app.route('/profile')
def profile_page():
    if 'user_id' not in session:
        return redirect(url_for('login_page'))

    user = db.session.get(User, session['user_id'])

    # Получаем теги из прочитанных материалов
    interactions = Interaction.query.filter_by(
        user_id=user.id,
        action='read'
    ).all()

    tags = {}
    for interaction in interactions:
        item = db.session.get(Item, interaction.item_id)
        if item and item.tags:
            for tag in item.get_tags_list():
                tags[tag] = tags.get(tag, 0) + 1

    # Получаем последние действия
    actions = Interaction.query.filter_by(user_id=user.id)\
        .order_by(Interaction.created_at.desc())\
        .limit(10)\
        .all()

    actions_list = []
    for action in actions:
        item = db.session.get(Item, action.item_id)
        if item:
            actions_list.append({
                'action': 'Прочитал' if action.action == 'read' else
                         'Сохранил' if action.action == 'save' else 'Просмотрел',
                'material': item.title,
                'date': action.created_at.strftime('%d.%m.%Y %H:%M')
            })

    return render_template('profile.html', user=user, tags=tags, actions=actions_list)

@app.route('/profile/<int:user_id>')
def profile_other_page(user_id):
    if 'user_id' not in session:
        return redirect(url_for('login_page'))

    user = db.session.get(User, user_id)
    if not user:
        return "Пользователь не найден", 404

    # Получаем теги из прочитанных материалов
    interactions = Interaction.query.filter_by(
        user_id=user.id,
        action='read'
    ).all()

    tags = {}
    for interaction in interactions:
        item = db.session.get(Item, interaction.item_id)
        if item and item.tags:
            for tag in item.get_tags_list():
                tags[tag] = tags.get(tag, 0) + 1

    return render_template('profile.html', user=user, tags=tags, actions=[], viewing_other=True)

@app.route('/demo-panel')
def demo_panel_page():
    if 'user_id' not in session:
        return redirect(url_for('login_page'))

    user = db.session.get(User, session['user_id'])
    if not user:
        session.pop('user_id', None)
        return redirect(url_for('login_page'))

    if user.role != 'admin':
        return "Доступ запрещен. Только для администраторов.", 403

    users = User.query.all()
    materials = Item.query.all()

    return render_template('demo_panel.html',
                         users=users,
                         materials=materials,
                         current_user=user)

# API эндпоинты
@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.get_json()
    if data and 'user_id' in data:
        session['user_id'] = data['user_id']
        return jsonify({'status': 'ok'})
    return jsonify({'status': 'error', 'message': 'No user_id provided'}), 400

@app.route('/api/materials')
def api_materials():
    materials = Item.query.order_by(Item.date.desc()).all()
    return jsonify({
        'materials': [m.to_dict() for m in materials]
    })

@app.route('/api/interaction', methods=['POST'])
def api_interaction():
    """Сохранить взаимодействие пользователя с материалом"""
    global new_interactions_count, training_lock

    data = request.json
    user_id = data.get('user_id')
    item_id = data.get('item_id')
    action = data.get('action')

    if not all([user_id, item_id, action]):
        return jsonify({'status': 'error', 'message': 'Missing data'}), 400

    # Веса действий
    weights = {'view': 1, 'save': 5, 'share': 10}

    interaction = Interaction(
        user_id=user_id,
        item_id=item_id,
        action=action,
        weight=weights.get(action, 1)
    )

    db.session.add(interaction)

    # Увеличиваем счетчик просмотров
    if action == 'view':
        item = Item.query.get(item_id)
        if item:
            item.views += 1

    db.session.commit()

    # Счетчик для автоматического переобучения
    new_interactions_count += 1

    # Если набрали порог - переобучаем в фоне
    if new_interactions_count >= RETRAIN_THRESHOLD:
        new_interactions_count = 0
        # Запускаем переобучение, только если не обучается прямо сейчас
        if training_lock.acquire(blocking=False):
            try:
                threading.Thread(target=update_recommender_with_context).start()
                print("🔄 Запущено фоновое переобучение модели")
            except:
                training_lock.release()
                print("❌ Не удалось запустить поток переобучения")
        else:
            print("⚠️ Переобучение уже запущено, пропускаем")

    return jsonify({'status': 'ok'})

@app.route('/api/user/<int:user_id>/interests')
def api_user_interests(user_id):
    # Получаем теги из прочитанных материалов
    interactions = Interaction.query.filter_by(
        user_id=user_id,
        action='read'
    ).all()

    tags = {}
    for interaction in interactions:
        item = db.session.get(Item, interaction.item_id)
        if item and item.tags:
            for tag in item.get_tags_list():
                tags[tag] = tags.get(tag, 0) + 1

    return jsonify({
        'user_id': user_id,
        'tags': tags,
        'total_reads': len(interactions)
    })

# CRUD операции для демо-панели
@app.route('/api/items', methods=['POST'])
def api_create_item():
    if 'user_id' not in session:
        return jsonify({'status': 'error', 'message': 'Not logged in'}), 401

    user = db.session.get(User, session['user_id'])
    if user.role != 'admin':
        return jsonify({'status': 'error', 'message': 'Access denied'}), 403

    data = request.get_json()

    item = Item(
        title=data.get('title'),
        type=data.get('type'),
        type_icon=data.get('type_icon', '📄'),
        tags=','.join(data.get('tags', [])),
        date=datetime.now().strftime('%Y-%m-%d'),
        department=data.get('department'),
        department_owner=data.get('department_owner'),
        preview=data.get('preview', '')[:200],
        content=data.get('content'),
        views=0,
        saved_count=0,
        author=data.get('author', user.name)
    )

    db.session.add(item)
    db.session.commit()

    return jsonify({'status': 'ok', 'id': item.id})

@app.route('/api/items/<int:item_id>', methods=['GET'])
def api_get_item(item_id):
    """Получить один материал (для редактирования)"""
    item = db.session.get(Item, item_id)
    if not item:
        return jsonify({'error': 'Not found'}), 404
    return jsonify(item.to_dict())

@app.route('/api/items/<int:item_id>', methods=['PUT'])
def api_update_item(item_id):
    if 'user_id' not in session:
        return jsonify({'status': 'error', 'message': 'Not logged in'}), 401

    user = db.session.get(User, session['user_id'])
    if user.role != 'admin':
        return jsonify({'status': 'error', 'message': 'Access denied'}), 403

    item = db.session.get(Item, item_id)
    if not item:
        return jsonify({'status': 'error', 'message': 'Item not found'}), 404

    data = request.get_json()

    item.title = data.get('title', item.title)
    item.type = data.get('type', item.type)
    item.type_icon = data.get('type_icon', item.type_icon)
    if 'tags' in data:
        item.tags = ','.join(data['tags'])
    item.department = data.get('department', item.department)
    item.department_owner = data.get('department_owner', item.department_owner)
    item.preview = data.get('preview', item.preview)
    item.content = data.get('content', item.content)
    item.author = data.get('author', item.author)

    db.session.commit()

    return jsonify({'status': 'ok'})

@app.route('/api/items/<int:item_id>', methods=['DELETE'])
def api_delete_item(item_id):
    if 'user_id' not in session:
        return jsonify({'status': 'error', 'message': 'Not logged in'}), 401

    user = db.session.get(User, session['user_id'])
    if user.role != 'admin':
        return jsonify({'status': 'error', 'message': 'Access denied'}), 403

    item = db.session.get(Item, item_id)
    if not item:
        return jsonify({'status': 'error', 'message': 'Item not found'}), 404

    # Удаляем связанные взаимодействия
    Interaction.query.filter_by(item_id=item_id).delete()

    db.session.delete(item)
    db.session.commit()

    return jsonify({'status': 'ok'})

@app.route('/api/recommendations/<int:user_id>')
def api_recommendations(user_id):
    """
    Получить рекомендации для пользователя
    ТЕПЕРЬ С РЕАЛЬНОЙ МОДЕЛЬЮ
    """
    try:
        # Получаем рекомендации из модели
        recommended_items = recommender.get_recommendations(user_id, n=5)

        # Возвращаем в формате, совместимом с фронтендом
        return jsonify({
            'user_id': user_id,
            'recommendations': [item.id for item in recommended_items],  # список ID
            'materials': [item.to_dict() for item in recommended_items], # список объектов
            'method': 'hybrid_lightfm'
        })
    except Exception as e:
        print(f"Ошибка получения рекомендаций: {e}")
        # В случае ошибки возвращаем популярное
        popular_items = Item.query.order_by(Item.views.desc()).limit(5).all()
        return jsonify({
            'user_id': user_id,
            'recommendations': [item.id for item in popular_items],
            'materials': [item.to_dict() for item in popular_items],
            'method': 'popular_fallback'
        })

@app.route('/api/recommendations/<int:user_id>/explain/<int:item_id>')
def api_explain_recommendation(user_id, item_id):
    """
    Получить объяснение, почему материал рекомендован
    """
    explanation = recommender.explain_recommendation(user_id, item_id)
    return jsonify({
        'user_id': user_id,
        'item_id': item_id,
        'explanation': explanation
    })

@app.route('/api/recommender/retrain', methods=['POST'])
def api_retrain_recommender():
    """
    Переобучить модель (только для admin)
    """
    # Проверка прав администратора
    user_id = session.get('user_id')
    user = User.query.get(user_id)

    if user and user.role == 'admin':
        # Запускаем переобучение в фоне
        if training_lock.acquire(blocking=False):
            try:
                threading.Thread(target=update_recommender_with_context).start()
                return jsonify({'status': 'ok', 'message': 'Модель переобучается в фоне'})
            except:
                training_lock.release()
                return jsonify({'status': 'error', 'message': 'Ошибка запуска переобучения'}), 500
        else:
            return jsonify({'status': 'ok', 'message': 'Переобучение уже запущено'})
    else:
        return jsonify({'status': 'error', 'message': 'Недостаточно прав'}), 403

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
