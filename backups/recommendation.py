"""
Модуль рекомендаций на основе LightFM (коллаборативная версия с объяснениями по метаданным)
"""

import numpy as np
from lightfm import LightFM
from lightfm.data import Dataset
from models import db, User, Item, Interaction
import pickle
import os
import traceback

class RecommenderSystem:
    def __init__(self):
        self.model = None
        self.dataset = Dataset()
        self.interactions = None
        self.item_id_map = {}
        self.user_id_map = {}
        # Для объяснений храним признаки в виде списков
        self.user_features_list = {}   # user_key -> [признаки]
        self.item_features_list = {}   # item_key -> [признаки]

        self.learning_rate = 0.05
        self.epochs = 30
        self.no_components = 20

    def _get_user_features(self, user):
        """Собирает признаки пользователя для объяснений"""
        feats = []
        if user.department and user.department.strip():
            feats.append(f"dept_{user.department}")
        if user.position and user.position.strip():
            feats.append(f"pos_{user.position}")
        if user.projects and user.projects.strip():
            for proj in user.projects.split(','):
                proj = proj.strip()
                if proj:
                    feats.append(f"proj_{proj}")
        return feats

    def _get_item_features(self, item):
        """Собирает признаки материала для объяснений"""
        feats = []
        if item.type and item.type.strip():
            feats.append(f"type_{item.type}")
        if item.department and item.department.strip():
            feats.append(f"dept_{item.department}")
        if item.tags and item.tags.strip():
            for tag in item.get_tags_list():
                tag = tag.strip()
                if tag:
                    feats.append(f"tag_{tag}")
        return feats

    def prepare_data(self):
        print("📥 Загрузка данных из БД...")
        try:
            users = User.query.all()
            items = Item.query.all()
            print(f"   Найдено пользователей: {len(users)}")
            print(f"   Найдено материалов: {len(items)}")

            user_ids = [f"user_{u.id}" for u in users]
            item_ids = [f"item_{i.id}" for i in items]

            self.user_id_map = {u.id: f"user_{u.id}" for u in users}
            self.item_id_map = {i.id: f"item_{i.id}" for i in items}

            # Сохраняем признаки для объяснений
            self.user_features_list.clear()
            for user in users:
                feats = self._get_user_features(user)
                if feats:
                    self.user_features_list[f"user_{user.id}"] = feats

            self.item_features_list.clear()
            for item in items:
                feats = self._get_item_features(item)
                if feats:
                    self.item_features_list[f"item_{item.id}"] = feats

            print(f"   Признаков пользователей: {len(self.user_features_list)}")
            print(f"   Признаков материалов: {len(self.item_features_list)}")

            # Фитим датасет БЕЗ ПРИЗНАКОВ (избегаем ошибок библиотеки)
            self.dataset.fit(user_ids, item_ids)
            print("   ✅ Датасет подготовлен (без признаков)")

            print("✅ Данные подготовлены")
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            traceback.print_exc()
            db.session.rollback()
            raise

    def build_interaction_matrix(self):
        print("🔧 Построение матрицы взаимодействий...")
        try:
            weights = {'view': 1, 'read': 3, 'save': 5, 'share': 10}
            interactions_data = []
            for inter in Interaction.query.all():
                user_key = f"user_{inter.user_id}"
                item_key = f"item_{inter.item_id}"
                if user_key in self.dataset.mapping()[0] and item_key in self.dataset.mapping()[2]:
                    interactions_data.append((user_key, item_key, weights.get(inter.action, 1)))
            print(f"   Взаимодействий: {len(interactions_data)}")
            if interactions_data:
                self.interactions = self.dataset.build_interactions(interactions_data)[0]
                print(f"   Матрица: {self.interactions.shape}")
            else:
                self.interactions = None
            print("✅ Матрица готова")
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            traceback.print_exc()
            raise

    def train(self):
        print("🧠 Обучение модели...")
        try:
            if self.interactions is None or self.interactions.nnz == 0:
                print("   ⚠️ Нет данных")
                self.model = None
                return
            self.model = LightFM(
                learning_rate=self.learning_rate,
                loss='warp',
                no_components=self.no_components,
                random_state=42
            )
            self.model.fit(self.interactions, epochs=self.epochs, verbose=True)
            print("✅ Модель обучена (коллаборативная)")
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            traceback.print_exc()
            self.model = None

    def get_recommendations(self, user_id, n=5):
        try:
            if self.model is None:
                return self.get_popular_items(n)
            user_key = f"user_{user_id}"
            if user_key not in self.dataset.mapping()[0]:
                return self.get_popular_items(n)
            user_idx = self.dataset.mapping()[0][user_key]
            item_ids = list(self.item_id_map.keys())
            item_idxs = []
            valid_ids = []
            for item_id in item_ids:
                item_key = f"item_{item_id}"
                if item_key in self.dataset.mapping()[2]:
                    item_idxs.append(self.dataset.mapping()[2][item_key])
                    valid_ids.append(item_id)
            if not item_idxs:
                return self.get_popular_items(n)
            scores = self.model.predict(user_idx, item_idxs)
            top_idx = np.argsort(-scores)[:n*2]
            top_ids = [valid_ids[i] for i in top_idx if i < len(valid_ids)]
            read = [i.item_id for i in Interaction.query.filter_by(user_id=user_id).all()]
            final = [iid for iid in top_ids if iid not in read]
            if len(final) < n:
                popular = self.get_popular_items(n - len(final))
                final.extend([p.id for p in popular])
            if final:
                return Item.query.filter(Item.id.in_(final[:n])).all()
            return self.get_popular_items(n)
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            traceback.print_exc()
            return self.get_popular_items(n)

    def get_popular_items(self, n=5):
        try:
            return Item.query.order_by(Item.views.desc()).limit(n).all()
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            return []

    def explain_recommendation(self, user_id, item_id):
        try:
            user = User.query.get(user_id)
            item = Item.query.get(item_id)
            if not user or not item:
                return "Нет данных"

            explanations = []

            # Используем сохранённые списки признаков
            user_feats = self.user_features_list.get(f"user_{user_id}", [])
            item_feats = self.item_features_list.get(f"item_{item_id}", [])

            # Ищем совпадения
            common = set(user_feats) & set(item_feats)
            if common:
                readable = []
                for feat in common:
                    if feat.startswith('dept_'):
                        readable.append(f"отдел {feat[5:]}")
                    elif feat.startswith('pos_'):
                        readable.append(f"должность {feat[4:]}")
                    elif feat.startswith('proj_'):
                        readable.append(f"проект {feat[5:]}")
                    elif feat.startswith('tag_'):
                        readable.append(f"тег #{feat[4:]}")
                    elif feat.startswith('type_'):
                        readable.append(f"тип {feat[5:]}")
                    else:
                        readable.append(feat)
                explanations.append("совпадают: " + ", ".join(readable[:3]))

            # По отделу (если не попал в совпадения)
            if not common and user.department and item.department and user.department == item.department:
                explanations.append(f"отдел {user.department}")

            # Популярность
            if item.views > 10:
                explanations.append(f"популярно ({item.views} просмотров)")

            if explanations:
                return " + ".join(explanations)
            return "На основе ваших предпочтений"
        except Exception as e:
            print(f"❌ Ошибка explain: {e}")
            return "На основе ваших интересов"

    def save_model(self, path='model.pkl'):
        try:
            with open(path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'dataset': self.dataset,
                    'user_id_map': self.user_id_map,
                    'item_id_map': self.item_id_map,
                    'user_features_list': self.user_features_list,
                    'item_features_list': self.item_features_list
                }, f)
            print(f"💾 Модель сохранена")
        except Exception as e:
            print(f"❌ Ошибка сохранения: {e}")

    def load_model(self, path='model.pkl'):
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                    self.model = data['model']
                    self.dataset = data['dataset']
                    self.user_id_map = data['user_id_map']
                    self.item_id_map = data['item_id_map']
                    self.user_features_list = data.get('user_features_list', {})
                    self.item_features_list = data.get('item_features_list', {})
                print(f"📂 Модель загружена")
                return True
            except Exception as e:
                print(f"❌ Ошибка загрузки: {e}")
                traceback.print_exc()
        return False

recommender = RecommenderSystem()

def init_recommender():
    print("🚀 Инициализация...")
    try:
        if not recommender.load_model():
            print("🆕 Создание новой модели...")
            recommender.prepare_data()
            recommender.build_interaction_matrix()
            recommender.train()
            recommender.save_model()
        print("✅ Система готова")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        traceback.print_exc()
    return recommender

def update_recommender():
    print("🔄 Переобучение...")
    try:
        recommender.prepare_data()
        recommender.build_interaction_matrix()
        recommender.train()
        recommender.save_model()
        print("✅ Переобучено")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        traceback.print_exc()
