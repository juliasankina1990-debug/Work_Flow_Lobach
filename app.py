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
    init_recommender()

def update_recommender_with_context():
    global new_interactions_count, training_lock
    try:
        with app.app_context():
            update_recommender()
    finally:
        training_lock.release()

# Вспомогательная функция для обогащения материалов статусами пользователя
def enrich_with_user_status(materials, user_id):
    """Добавляет к каждому материалу поля is_read, is_saved, is_viewed"""
    if not user_id:
        for mat in materials:
            mat_dict = mat.to_dict()
            mat_dict['is_read'] = False
            mat_dict['is_saved'] = False
            mat_dict['is_viewed'] = False
            yield mat_dict
        return
    
    interactions = Interaction.query.filter_by(user_id=user_id).all()
    status_map = {}
    for inter in interactions:
        status_map.setdefault(inter.item_id, {'read': False, 'save': False, 'view': False})
        status_map[inter.item_id][inter.action] = True
    for mat in materials:
        mat_dict = mat.to_dict()
        status = status_map.get(mat.id, {})
        mat_dict['is_read'] = status.get('read', False)
        mat_dict['is_saved'] = status.get('save', False)
        mat_dict['is_viewed'] = status.get('view', False)
        yield mat_dict

# Декоратор для проверки авторизации
def login_required(f):
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

def admin_required(f):
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login_page'))
        user = db.session.get(User, session['user_id'])
        if not user or user.role != 'admin':
            return "Доступ запрещен", 403
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

# Маршруты
@app.route('/')
@login_required
def index():
    user = db.session.get(User, session['user_id'])
    if not user:
        return redirect(url_for('login_page'))
    return render_template('index.html', user=user)

@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if request.method == 'POST':
        data = request.get_json()
        if data:
            username = data.get('username')
            password = data.get('password')
            # Простая проверка пароля без хеширования
            user = User.query.filter_by(username=username, password=password).first()
            
            if user:
                session['user_id'] = user.id
                return jsonify({'status': 'ok', 'redirect': url_for('index')})
            else:
                return jsonify({'status': 'error', 'message': 'Неверное имя пользователя или пароль'}), 401
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register_page():
    if request.method == 'POST':
        data = request.get_json()
        if data:
            username = data.get('username')
            password = data.get('password')
            password_confirm = data.get('password_confirm')
            name = data.get('name', username)
            
            # Проверяем совпадение паролей
            if password != password_confirm:
                return jsonify({'status': 'error', 'message': 'Пароли не совпадают'}), 400
            
            # Проверяем, существует ли пользователь
            if User.query.filter_by(username=username).first():
                return jsonify({'status': 'error', 'message': 'Имя пользователя уже занято'}), 400
            
            # Создаем нового пользователя с паролем в открытом виде
            user = User(
                username=username,
                name=name,
                password=password,  # Пароль сохраняется как есть
                position=data.get('position', 'Сотрудник'),
                department=data.get('department', 'Общий отдел'),
                avatar=data.get('avatar', '👤'),
                role='user'
            )
            
            db.session.add(user)
            db.session.commit()
            
            # Сразу авторизуем
            session['user_id'] = user.id
            return jsonify({'status': 'ok', 'redirect': url_for('index')})
    
    return render_template('register.html')

@app.route('/logout')
def logout_page():
    session.pop('user_id', None)
    return redirect(url_for('login_page'))

@app.route('/material/<int:id>')
@login_required
def material_page(id):
    user = db.session.get(User, session['user_id'])
    material = db.session.get(Item, id)
    if not material:
        return "Материал не найден", 404
    material.views += 1
    db.session.commit()
    interaction = Interaction(user_id=user.id, item_id=material.id, action='view')
    db.session.add(interaction)
    db.session.commit()
    return render_template('material.html', user=user, material=material)

@app.route('/profile')
@login_required
def profile_page():
    user = db.session.get(User, session['user_id'])
    interactions = Interaction.query.filter_by(user_id=user.id, action='read').all()
    tags = {}
    for interaction in interactions:
        item = db.session.get(Item, interaction.item_id)
        if item and item.tags:
            for tag in item.get_tags_list():
                tags[tag] = tags.get(tag, 0) + 1
    actions = Interaction.query.filter_by(user_id=user.id)\
        .order_by(Interaction.created_at.desc()).limit(10).all()
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
@login_required
def profile_other_page(user_id):
    current_user = db.session.get(User, session['user_id'])
    user = db.session.get(User, user_id)
    if not user:
        return "Пользователь не найден", 404
    interactions = Interaction.query.filter_by(user_id=user.id, action='read').all()
    tags = {}
    for interaction in interactions:
        item = db.session.get(Item, interaction.item_id)
        if item and item.tags:
            for tag in item.get_tags_list():
                tags[tag] = tags.get(tag, 0) + 1
    return render_template('profile.html', user=user, tags=tags, actions=[], viewing_other=True)

@app.route('/articles')
@login_required
def articles_page():
    user = db.session.get(User, session['user_id'])
    return render_template('articles.html', user=user)

@app.route('/favorites')
@login_required
def favorites_page():
    user = db.session.get(User, session['user_id'])
    return render_template('favorites.html', user=user)

@app.route('/history')
@login_required
def history_page():
    user = db.session.get(User, session['user_id'])
    return render_template('history.html', user=user)

@app.route('/demo-panel')
@admin_required
def demo_panel_page():
    current_user = db.session.get(User, session['user_id'])
    users = User.query.all()
    materials = Item.query.all()
    return render_template('demo_panel.html', users=users, materials=materials, current_user=current_user)

# API эндпоинты
@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.get_json()
    if data:
        username = data.get('username')
        password = data.get('password')
        # Простая проверка пароля без хеширования
        user = User.query.filter_by(username=username, password=password).first()
        
        if user:
            session['user_id'] = user.id
            return jsonify({'status': 'ok'})
    
    return jsonify({'status': 'error', 'message': 'Неверные учетные данные'}), 401

@app.route('/api/register', methods=['POST'])
def api_register():
    data = request.get_json()
    if data:
        username = data.get('username')
        password = data.get('password')
        password_confirm = data.get('password_confirm')
        
        if password != password_confirm:
            return jsonify({'status': 'error', 'message': 'Пароли не совпадают'}), 400
        
        if User.query.filter_by(username=username).first():
            return jsonify({'status': 'error', 'message': 'Имя пользователя уже занято'}), 400
        
        user = User(
            username=username,
            name=data.get('name', username),
            password=password,  # Пароль сохраняется как есть
            position=data.get('position', 'Сотрудник'),
            department=data.get('department', 'Общий отдел'),
            avatar=data.get('avatar', '👤'),
            role='user'
        )
        
        db.session.add(user)
        db.session.commit()
        
        session['user_id'] = user.id
        return jsonify({'status': 'ok', 'redirect': url_for('index')})
    
    return jsonify({'status': 'error', 'message': 'Нет данных'}), 400

@app.route('/api/materials')
@login_required
def api_materials():
    materials = Item.query.order_by(Item.date.desc()).all()
    user_id = session.get('user_id')
    return jsonify({'materials': list(enrich_with_user_status(materials, user_id))})

@app.route('/api/user/<int:user_id>/interests')
@login_required
def api_user_interests(user_id):
    current_user = db.session.get(User, session['user_id'])
    if current_user.id != user_id and current_user.role != 'admin':
        return jsonify({'error': 'Access denied'}), 403
    
    interactions = Interaction.query.filter_by(user_id=user_id, action='read').all()
    tags = {}
    for interaction in interactions:
        item = db.session.get(Item, interaction.item_id)
        if item and item.tags:
            for tag in item.get_tags_list():
                tags[tag] = tags.get(tag, 0) + 1
    return jsonify({'user_id': user_id, 'tags': tags, 'total_reads': len(interactions)})

@app.route('/api/user/<int:user_id>/stats')
@login_required
def api_user_stats(user_id):
    current_user = db.session.get(User, session['user_id'])
    if current_user.id != user_id and current_user.role != 'admin':
        return jsonify({'error': 'Access denied'}), 403
    
    user = db.session.get(User, user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    total_read = Interaction.query.filter_by(user_id=user_id, action='read').count()
    total_saved = Interaction.query.filter_by(user_id=user_id, action='save').count()
    total_viewed = Interaction.query.filter_by(user_id=user_id, action='view').count()
    tag_counter = {}
    for inter in Interaction.query.filter_by(user_id=user_id, action='read').all():
        item = inter.item
        if item and item.tags:
            for tag in item.get_tags_list():
                tag_counter[tag] = tag_counter.get(tag, 0) + 1
    top_tags = sorted(tag_counter.items(), key=lambda x: x[1], reverse=True)[:5]
    return jsonify({
        'total_read': total_read,
        'total_saved': total_saved,
        'total_viewed': total_viewed,
        'top_tags': [{'tag': t, 'count': c} for t, c in top_tags]
    })

@app.route('/api/user/<int:user_id>/history')
@login_required
def api_user_history(user_id):
    if user_id != session.get('user_id'):
        return jsonify({'error': 'Access denied'}), 403
    interactions = Interaction.query.filter_by(user_id=user_id)\
        .order_by(Interaction.created_at.desc()).limit(50).all()
    history = []
    for inter in interactions:
        item = inter.item
        if item:
            history.append({
                'id': inter.id,
                'item_id': item.id,
                'title': item.title,
                'action': inter.action,
                'date': inter.created_at.strftime('%d.%m.%Y %H:%M'),
                'weight': inter.weight
            })
    return jsonify(history)

@app.route('/api/user/<int:user_id>/favorites')
@login_required
def api_user_favorites(user_id):
    if user_id != session.get('user_id'):
        return jsonify({'error': 'Access denied'}), 403
    saved_interactions = Interaction.query.filter_by(user_id=user_id, action='save').all()
    items = [inter.item for inter in saved_interactions if inter.item]
    enriched = list(enrich_with_user_status(items, user_id))
    return jsonify({'materials': enriched})

@app.route('/api/interaction', methods=['POST'])
@login_required
def api_interaction():
    global new_interactions_count, training_lock
    data = request.json
    user_id = data.get('user_id')
    item_id = data.get('item_id')
    action = data.get('action')
    
    if user_id != session.get('user_id'):
        return jsonify({'status': 'error', 'message': 'Access denied'}), 403
    
    if not all([user_id, item_id, action]):
        return jsonify({'status': 'error', 'message': 'Missing data'}), 400
    
    weights = {'view': 1, 'read': 3, 'save': 5, 'share': 10}
    interaction = Interaction(user_id=user_id, item_id=item_id, action=action, weight=weights.get(action, 1))
    db.session.add(interaction)
    if action == 'view':
        item = db.session.get(Item, item_id)
        if item:
            item.views += 1
    db.session.commit()
    
    new_interactions_count += 1
    if new_interactions_count >= RETRAIN_THRESHOLD:
        new_interactions_count = 0
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

# CRUD для материалов
@app.route('/api/items', methods=['POST'])
@admin_required
def api_create_item():
    data = request.get_json()
    user = db.session.get(User, session['user_id'])
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
@login_required
def api_get_item(item_id):
    item = db.session.get(Item, item_id)
    if not item:
        return jsonify({'error': 'Not found'}), 404
    return jsonify(item.to_dict())

@app.route('/api/items/<int:item_id>', methods=['PUT'])
@admin_required
def api_update_item(item_id):
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
@admin_required
def api_delete_item(item_id):
    item = db.session.get(Item, item_id)
    if not item:
        return jsonify({'status': 'error', 'message': 'Item not found'}), 404
    Interaction.query.filter_by(item_id=item_id).delete()
    db.session.delete(item)
    db.session.commit()
    return jsonify({'status': 'ok'})

@app.route('/api/items/<int:item_id>/interactions', methods=['DELETE'])
@admin_required
def api_reset_item_interactions(item_id):
    count = Interaction.query.filter_by(item_id=item_id).delete()
    db.session.commit()
    return jsonify({'status': 'ok', 'deleted': count})

# CRUD для пользователей (только для админов)
@app.route('/api/users', methods=['GET'])
@admin_required
def api_get_users():
    users = User.query.all()
    return jsonify({'users': [user.to_dict_admin() for user in users]})

@app.route('/api/users/<int:user_id>', methods=['GET'])
@admin_required
def api_get_user(user_id):
    user = db.session.get(User, user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    return jsonify(user.to_dict_admin())

@app.route('/api/users', methods=['POST'])
@admin_required
def api_create_user():
    data = request.get_json()
    
    if User.query.filter_by(username=data.get('username')).first():
        return jsonify({'status': 'error', 'message': 'Имя пользователя уже занято'}), 400
    
    user = User(
        username=data.get('username'),
        name=data.get('name'),
        password=data.get('password', 'default123'),  # Пароль в открытом виде
        position=data.get('position', 'Сотрудник'),
        department=data.get('department', 'Общий отдел'),
        avatar=data.get('avatar', '👤'),
        color=data.get('color', '#3498db'),
        projects=','.join(data.get('projects', [])),
        hire_date=data.get('hire_date'),
        role=data.get('role', 'user')
    )
    
    db.session.add(user)
    db.session.commit()
    return jsonify({'status': 'ok', 'id': user.id})

@app.route('/api/users/<int:user_id>', methods=['PUT'])
@admin_required
def api_update_user(user_id):
    user = db.session.get(User, user_id)
    if not user:
        return jsonify({'status': 'error', 'message': 'User not found'}), 404
    
    current_user = db.session.get(User, session['user_id'])
    if user.id == current_user.id and request.get_json().get('role') != 'admin':
        return jsonify({'status': 'error', 'message': 'Нельзя изменить свою роль'}), 403
    
    data = request.get_json()
    
    new_username = data.get('username')
    if new_username and new_username != user.username:
        if User.query.filter_by(username=new_username).first():
            return jsonify({'status': 'error', 'message': 'Имя пользователя уже занято'}), 400
        user.username = new_username
    
    user.name = data.get('name', user.name)
    user.position = data.get('position', user.position)
    user.department = data.get('department', user.department)
    user.avatar = data.get('avatar', user.avatar)
    user.color = data.get('color', user.color)
    if 'projects' in data:
        user.projects = ','.join(data['projects'])
    user.hire_date = data.get('hire_date', user.hire_date)
    user.role = data.get('role', user.role)
    
    # Если передан новый пароль
    if data.get('password'):
        user.password = data['password']  # Просто сохраняем новый пароль
    
    db.session.commit()
    return jsonify({'status': 'ok'})

@app.route('/api/users/<int:user_id>', methods=['DELETE'])
@admin_required
def api_delete_user(user_id):
    user = db.session.get(User, user_id)
    if not user:
        return jsonify({'status': 'error', 'message': 'User not found'}), 404
    
    current_user = db.session.get(User, session['user_id'])
    if user.id == current_user.id:
        return jsonify({'status': 'error', 'message': 'Нельзя удалить самого себя'}), 403
    
    if user.role == 'admin' and current_user.role == 'admin':
        admin_count = User.query.filter_by(role='admin').count()
        if admin_count <= 1:
            return jsonify({'status': 'error', 'message': 'Нельзя удалить последнего администратора'}), 403
    
    Interaction.query.filter_by(user_id=user_id).delete()
    db.session.delete(user)
    db.session.commit()
    return jsonify({'status': 'ok'})

@app.route('/api/recommendations/<int:user_id>')
@login_required
def api_recommendations(user_id):
    try:
        recommended_items = recommender.get_recommendations(user_id, n=5)
        user_id_session = session.get('user_id')
        materials_with_status = list(enrich_with_user_status(recommended_items, user_id_session))
        return jsonify({
            'user_id': user_id,
            'recommendations': [item.id for item in recommended_items],
            'materials': materials_with_status,
            'method': 'hybrid_lightfm'
        })
    except Exception as e:
        print(f"Ошибка получения рекомендаций: {e}")
        popular_items = Item.query.order_by(Item.views.desc()).limit(5).all()
        materials_with_status = list(enrich_with_user_status(popular_items, session.get('user_id')))
        return jsonify({
            'user_id': user_id,
            'recommendations': [item.id for item in popular_items],
            'materials': materials_with_status,
            'method': 'popular_fallback'
        })

@app.route('/api/recommendations/<int:user_id>/explain/<int:item_id>')
@login_required
def api_explain_recommendation(user_id, item_id):
    explanation = recommender.explain_recommendation(user_id, item_id)
    return jsonify({'user_id': user_id, 'item_id': item_id, 'explanation': explanation})

@app.route('/api/recommender/retrain', methods=['POST'])
@admin_required
def api_retrain_recommender():
    if training_lock.acquire(blocking=False):
        try:
            threading.Thread(target=update_recommender_with_context).start()
            return jsonify({'status': 'ok', 'message': 'Модель переобучается в фоне'})
        except:
            training_lock.release()
            return jsonify({'status': 'error', 'message': 'Ошибка запуска переобучения'}), 500
    else:
        return jsonify({'status': 'ok', 'message': 'Переобучение уже запущено'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
