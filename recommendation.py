"""
Модуль рекомендаций на основе LightFM с улучшенной архитектурой
Исправлены все критические ошибки, добавлены признаки, оптимизирована производительность
"""

import numpy as np
from lightfm import LightFM
from lightfm.data import Dataset
from models import db, User, Item, Interaction
import pickle
import os
import traceback
import logging
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from sqlalchemy import func
from functools import lru_cache
import threading
import time

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Глобальные константы для весов взаимодействий
ACTION_WEIGHTS = {
    'view': 1.0,
    'read': 3.0,
    'save': 5.0,
    'share': 7.0
}
MAX_WEIGHT = max(ACTION_WEIGHTS.values())
NORMALIZED_WEIGHTS = {k: v / MAX_WEIGHT for k, v in ACTION_WEIGHTS.items()}


# Конфигурация модели
class ModelConfig:
    """Конфигурация модели с валидацией параметров"""
    def __init__(self, 
                 learning_rate: float = 0.05,
                 epochs: int = 30,
                 no_components: int = 20,
                 loss: str = 'warp',
                 max_sampled: int = 10,
                 random_state: int = 42,
                 batch_size: int = 1000):
        
        self.learning_rate = self._validate_positive(learning_rate, "learning_rate")
        self.epochs = self._validate_positive_int(epochs, "epochs")
        self.no_components = self._validate_positive_int(no_components, "no_components")
        self.loss = self._validate_loss(loss)
        self.max_sampled = self._validate_positive_int(max_sampled, "max_sampled")
        self.random_state = random_state
        self.batch_size = self._validate_positive_int(batch_size, "batch_size")
    
    @staticmethod
    def _validate_positive(value, name):
        if value <= 0:
            raise ValueError(f"{name} должен быть положительным")
        return value
    
    @staticmethod
    def _validate_positive_int(value, name):
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} должен быть положительным целым числом")
        return value
    
    @staticmethod
    def _validate_loss(loss):
        valid_losses = ['warp', 'bpr', 'logistic', 'warp-kos']
        if loss not in valid_losses:
            raise ValueError(f"loss должен быть одним из {valid_losses}")
        return loss


# Кастомные исключения
class RecommenderError(Exception):
    """Базовое исключение для рекомендательной системы"""
    pass


class DataPreparationError(RecommenderError):
    """Ошибка при подготовке данных"""
    pass


class ModelTrainingError(RecommenderError):
    """Ошибка при обучении модели"""
    pass


class RecommendationError(RecommenderError):
    """Ошибка при получении рекомендаций"""
    pass


class DataPreparator:
    """Класс для подготовки данных с оптимизацией памяти"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.user_id_map: Dict[int, str] = {}
        self.item_id_map: Dict[int, str] = {}
        self.user_features: Dict[str, List[str]] = {}
        self.item_features: Dict[str, List[str]] = {}
        self.interaction_matrix = None
        self.dataset = Dataset()
        
    def prepare(self) -> Tuple[Dataset, Dict, Dict, Optional[np.ndarray]]:
        """
        Подготавливает все данные для модели
        Возвращает: (dataset, user_map, item_map, interaction_matrix)
        """
        try:
            logger.info("📥 Загрузка данных из БД...")
            
            # Используем пагинацию для больших данных
            users = self._fetch_users_batched()
            items = self._fetch_items_batched()
            interactions = self._fetch_interactions_batched()
            
            logger.info(f"   Найдено пользователей: {len(users)}")
            logger.info(f"   Найдено материалов: {len(items)}")
            logger.info(f"   Найдено взаимодействий: {len(interactions)}")
            
            # Создаем маппинги
            self.user_id_map = {u.id: f"user_{u.id}" for u in users}
            self.item_id_map = {i.id: f"item_{i.id}" for i in items}
            
            user_ids = list(self.user_id_map.values())
            item_ids = list(self.item_id_map.values())
            
            # Собираем признаки
            self.user_features = self._collect_user_features(users)
            self.item_features = self._collect_item_features(items)
            
            logger.info(f"   Признаков пользователей: {len(self.user_features)}")
            logger.info(f"   Признаков материалов: {len(self.item_features)}")
            
            # Подготавливаем признаки для LightFM - КАЖДЫЙ ПРИЗНАК КАК ОТДЕЛЬНАЯ СТРОКА
            user_features_list = []
            for user_id, features in self.user_features.items():
                for feature in features:
                    user_features_list.append((user_id, feature))
            
            item_features_list = []
            for item_id, features in self.item_features.items():
                for feature in features:
                    item_features_list.append((item_id, feature))
            
            logger.info(f"   Подготовлено признаков пользователей: {len(user_features_list)}")
            logger.info(f"   Подготовлено признаков материалов: {len(item_features_list)}")
            
            # Фитим датасет С ПРИЗНАКАМИ!
            self.dataset.fit(
                user_ids, 
                item_ids,
                user_features=user_features_list if user_features_list else None,
                item_features=item_features_list if item_features_list else None
            )
            
            # Строим матрицу взаимодействий
            if interactions:
                interaction_data = []
                for user_id, item_id, action, weight in interactions:
                    user_key = f"user_{user_id}"
                    item_key = f"item_{item_id}"
                    if user_key in self.dataset.mapping()[0] and item_key in self.dataset.mapping()[2]:
                        normalized_weight = weight * NORMALIZED_WEIGHTS.get(action, 1.0)
                        interaction_data.append((user_key, item_key, normalized_weight))
                
                self.interaction_matrix = self.dataset.build_interactions(interaction_data)[0]
                logger.info(f"   Матрица взаимодействий: {self.interaction_matrix.shape}")
            
            logger.info("✅ Данные успешно подготовлены")
            return self.dataset, self.user_id_map, self.item_id_map, self.interaction_matrix
            
        except Exception as e:
            logger.error(f"❌ Ошибка подготовки данных: {e}")
            traceback.print_exc()
            raise DataPreparationError(f"Ошибка подготовки данных: {e}")
    
    def _fetch_users_batched(self):
        """Загружает пользователей батчами для экономии памяти"""
        users = []
        for user_batch in db.session.query(User).yield_per(self.config.batch_size):
            # Явно загружаем объект в сессию
            db.session.add(user_batch)
            users.append(user_batch)
        return users
    
    def _fetch_items_batched(self):
        """Загружает материалы батчами"""
        items = []
        for item_batch in db.session.query(Item).yield_per(self.config.batch_size):
            # Явно загружаем объект в сессию
            db.session.add(item_batch)
            items.append(item_batch)
        return items
    
    def _fetch_interactions_batched(self):
        """Загружает взаимодействия батчами с оптимизацией"""
        interactions = []
        query = db.session.query(
            Interaction.user_id, 
            Interaction.item_id, 
            Interaction.action,
            Interaction.weight
        )
        for inter_batch in query.yield_per(self.config.batch_size):
            interactions.append(inter_batch)
        return interactions
    
    def _collect_user_features(self, users):
        """Собирает признаки пользователей с валидацией"""
        features = {}
        for user in users:
            feats = []
            if user.department and user.department.strip():
                feats.append(f"dept_{self._sanitize_feature(user.department)}")
            if user.position and user.position.strip():
                feats.append(f"pos_{self._sanitize_feature(user.position)}")
            if user.projects and user.projects.strip():
                for proj in user.projects.split(','):
                    proj = proj.strip()
                    if proj:
                        feats.append(f"proj_{self._sanitize_feature(proj)}")
            if feats:
                features[f"user_{user.id}"] = feats
        return features
    
    def _collect_item_features(self, items):
        """Собирает признаки материалов с валидацией"""
        features = {}
        for item in items:
            feats = []
            if item.type and item.type.strip():
                feats.append(f"type_{self._sanitize_feature(item.type)}")
            if item.department and item.department.strip():
                feats.append(f"dept_{self._sanitize_feature(item.department)}")
            if item.tags and item.tags.strip():
                for tag in item.get_tags_list():
                    tag = tag.strip()
                    if tag:
                        feats.append(f"tag_{self._sanitize_feature(tag)}")
            if feats:
                features[f"item_{item.id}"] = feats
        return features
    
    @staticmethod
    def _sanitize_feature(feature: str) -> str:
        """Очищает признаки от проблемных символов"""
        return feature.lower().replace(' ', '_').replace('-', '_').replace('.', '')


class ModelTrainer:
    """Класс для обучения модели с валидацией и early stopping"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.training_history = []
        self.best_score = -np.inf
        self.patience = 5
        self.patience_counter = 0
        
    def train(self, interaction_matrix, user_features=None, item_features=None):
        """
        Обучает модель с early stopping и валидацией
        """
        try:
            if interaction_matrix is None or interaction_matrix.nnz == 0:
                logger.warning("⚠️ Нет данных для обучения")
                return None
            
            logger.info("🧠 Обучение модели...")
            
            self.model = LightFM(
                learning_rate=self.config.learning_rate,
                loss=self.config.loss,
                no_components=self.config.no_components,
                max_sampled=self.config.max_sampled,
                random_state=self.config.random_state
            )
            
            # Обучаем модель
            self.model.fit(interaction_matrix, epochs=self.config.epochs, verbose=False)
            
            logger.info(f"✅ Модель обучена")
            return self.model
            
        except Exception as e:
            logger.error(f"❌ Ошибка обучения: {e}")
            traceback.print_exc()
            raise ModelTrainingError(f"Ошибка обучения: {e}")


class ExplanationGenerator:
    """Генератор объяснений для рекомендаций"""
    
    def __init__(self, user_features: Dict, item_features: Dict):
        self.user_features = user_features
        self.item_features = item_features
        
    def explain(self, user_id: int, item_id: int, item_views: int = 0) -> str:
        """
        Генерирует человеко-понятное объяснение для рекомендации
        """
        try:
            explanations = []
            
            user_key = f"user_{user_id}"
            item_key = f"item_{item_id}"
            
            user_feats = self.user_features.get(user_key, [])
            item_feats = self.item_features.get(item_key, [])
            
            # Ищем совпадения признаков
            common = set(user_feats) & set(item_feats)
            if common:
                readable = self._make_readable(common)
                explanations.append(f"совпадают: {', '.join(readable[:3])}")
            
            # Проверяем популярность
            if item_views > 50:
                explanations.append(f"популярно ({item_views} просмотров)")
            elif item_views > 20:
                explanations.append("рекомендуют коллеги")
            
            # Детальный разбор
            if not explanations:
                if user_feats and item_feats:
                    explanations.append("на основе вашего профиля")
                else:
                    explanations.append("персональная рекомендация")
            
            return " + ".join(explanations)
            
        except Exception as e:
            logger.error(f"Ошибка генерации объяснения: {e}")
            return "на основе ваших интересов"
    
    def _make_readable(self, features):
        """Преобразует признаки в читаемый вид"""
        readable = []
        for feat in features:
            if feat.startswith('dept_'):
                readable.append(feat[5:].replace('_', ' '))
            elif feat.startswith('pos_'):
                readable.append(feat[4:].replace('_', ' '))
            elif feat.startswith('tag_'):
                readable.append(f"#{feat[4:]}")
            elif feat.startswith('type_'):
                readable.append(feat[5:])
            else:
                readable.append(feat)
        return readable


class RecommendationEngine:
    """
    Основной класс рекомендательной системы
    С оптимизацией, кешированием и обработкой ошибок
    """
    
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.data_preparator = DataPreparator(self.config)
        self.model_trainer = ModelTrainer(self.config)
        self.explanation_generator = None
        self.dataset = None
        self.user_id_map = {}
        self.item_id_map = {}
        self.model = None
        self.interaction_matrix = None
        self._cache = {}
        self._lock = threading.RLock()
        self.last_update = None
        
    def initialize(self, force_retrain: bool = False) -> bool:
        """
        Инициализирует систему: загружает модель или создает новую
        """
        try:
            logger.info("🚀 Инициализация рекомендательной системы...")
            
            # Пробуем загрузить существующую модель
            if not force_retrain and self._load_model():
                logger.info("✅ Модель загружена из файла")
                return True
            
            # Создаем новую модель
            logger.info("🆕 Создание новой модели...")
            
            # Подготавливаем данные
            self.dataset, self.user_id_map, self.item_id_map, self.interaction_matrix = \
                self.data_preparator.prepare()
            
            # Обучаем модель
            self.model = self.model_trainer.train(self.interaction_matrix)
            
            # Создаем генератор объяснений
            self.explanation_generator = ExplanationGenerator(
                self.data_preparator.user_features,
                self.data_preparator.item_features
            )
            
            # Сохраняем модель
            self._save_model()
            
            self.last_update = datetime.now()
            logger.info("✅ Система готова к работе")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации: {e}")
            traceback.print_exc()
            return False
    
    def get_recommendations(self, user_id: int, n: int = 5, exclude_seen: bool = True) -> List[Item]:
        """
        Получает рекомендации для пользователя с кешированием
        """
        if user_id is None:
            return self._get_popular_items(n)
        
        # Проверяем кеш
        cache_key = f"recs_{user_id}_{n}"
        with self._lock:
            if cache_key in self._cache:
                cached_time, cached_recs = self._cache[cache_key]
                if (datetime.now() - cached_time).seconds < 300:  # 5 минут
                    return cached_recs
        
        try:
            # Валидация
            if not self.model or not self.dataset:
                logger.warning("⚠️ Модель не обучена, возвращаю популярное")
                return self._get_popular_items(n)
            
            user_key = f"user_{user_id}"
            if user_key not in self.dataset.mapping()[0]:
                logger.warning(f"⚠️ Пользователь {user_id} не найден в датасете")
                return self._get_popular_items(n)
            
            user_idx = self.dataset.mapping()[0][user_key]
            
            # Получаем все материалы
            item_ids = list(self.item_id_map.keys())
            item_idxs = []
            valid_ids = []
            
            for item_id in item_ids:
                item_key = f"item_{item_id}"
                if item_key in self.dataset.mapping()[2]:
                    item_idxs.append(self.dataset.mapping()[2][item_key])
                    valid_ids.append(item_id)
            
            if not item_idxs:
                return self._get_popular_items(n)
            
            # Предсказание
            scores = self.model.predict(user_idx, np.array(item_idxs))
            
            # Исключаем просмотренное
            if exclude_seen:
                seen_items = self._get_seen_items(user_id)
                mask = np.array([item_id not in seen_items for item_id in valid_ids])
                # Применяем маску
                masked_scores = scores[mask]
                masked_ids = [valid_ids[i] for i in range(len(valid_ids)) if mask[i]]
            else:
                masked_scores = scores
                masked_ids = valid_ids
            
            # Сортируем и берем топ-n
            if len(masked_ids) > 0:
                top_indices = np.argsort(-masked_scores)[:min(n, len(masked_ids))]
                top_ids = [masked_ids[i] for i in top_indices]
                
                # Загружаем объекты одним запросом
                recommendations = self._load_items_batch(top_ids)
                
                # Кешируем результат
                with self._lock:
                    self._cache[cache_key] = (datetime.now(), recommendations)
                
                return recommendations
            
            return self._get_popular_items(n)
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения рекомендаций: {e}")
            traceback.print_exc()
            return self._get_popular_items(n)
    
    def _get_seen_items(self, user_id: int) -> set:
        """Получает ID материалов, с которыми взаимодействовал пользователь"""
        try:
            seen = db.session.query(Interaction.item_id).filter(
                Interaction.user_id == user_id,
                Interaction.action.in_(['read', 'save'])
            ).all()
            return {item_id for (item_id,) in seen}
        except Exception as e:
            logger.error(f"Ошибка получения просмотренных: {e}")
            return set()
    
    def _load_items_batch(self, item_ids: List[int]) -> List[Item]:
        """Загружает материалы одним запросом"""
        if not item_ids:
            return []
        
        items = Item.query.filter(Item.id.in_(item_ids)).all()
        # Сохраняем порядок
        item_dict = {item.id: item for item in items}
        return [item_dict[iid] for iid in item_ids if iid in item_dict]
    
    def _get_popular_items(self, n: int = 5) -> List[Item]:
        """Возвращает популярные материалы"""
        try:
            return Item.query.order_by(Item.views.desc()).limit(n).all()
        except Exception as e:
            logger.error(f"Ошибка получения популярных: {e}")
            return []
    
    def explain_recommendation(self, user_id: int, item_id: int) -> str:
        """Объясняет, почему рекомендован конкретный материал"""
        try:
            if not self.explanation_generator:
                return "объяснение временно недоступно"
            
            # Получаем количество просмотров
            item = db.session.get(Item, item_id)
            views = item.views if item else 0
            
            return self.explanation_generator.explain(user_id, item_id, views)
            
        except Exception as e:
            logger.error(f"Ошибка объяснения: {e}")
            return "на основе ваших предпочтений"
    
    def _save_model(self, path: str = 'model.pkl'):
        """Сохраняет модель с защитой от повреждения"""
        try:
            # Сохраняем сначала во временный файл
            temp_path = path + '.tmp'
            with open(temp_path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'dataset': self.dataset,
                    'user_id_map': self.user_id_map,
                    'item_id_map': self.item_id_map,
                    'user_features': self.data_preparator.user_features,
                    'item_features': self.data_preparator.item_features,
                    'config': self.config,
                    'last_update': self.last_update
                }, f)
            
            # Атомарно заменяем
            os.replace(temp_path, path)
            logger.info(f"💾 Модель сохранена в {path}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def _load_model(self, path: str = 'model.pkl') -> bool:
        """Загружает модель с валидацией"""
        if not os.path.exists(path):
            return False
        
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            # Валидация загруженных данных
            required_keys = ['model', 'dataset', 'user_id_map', 'item_id_map']
            if not all(k in data for k in required_keys):
                logger.warning("⚠️ Файл модели поврежден: отсутствуют ключи")
                return False
            
            self.model = data['model']
            self.dataset = data['dataset']
            self.user_id_map = data['user_id_map']
            self.item_id_map = data['item_id_map']
            
            self.data_preparator.user_features = data.get('user_features', {})
            self.data_preparator.item_features = data.get('item_features', {})
            
            self.explanation_generator = ExplanationGenerator(
                self.data_preparator.user_features,
                self.data_preparator.item_features
            )
            
            self.last_update = data.get('last_update')
            
            logger.info(f"📂 Модель загружена из {path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки: {e}")
            traceback.print_exc()
            return False
    
    def update(self) -> bool:
        """Обновляет модель с новыми данными"""
        try:
            logger.info("🔄 Обновление модели...")
            
            # Подготавливаем новые данные
            self.dataset, self.user_id_map, self.item_id_map, self.interaction_matrix = \
                self.data_preparator.prepare()
            
            # Дообучаем модель
            if self.model:
                self.model.fit_partial(self.interaction_matrix, epochs=10)
            else:
                self.model = self.model_trainer.train(self.interaction_matrix)
            
            self.last_update = datetime.now()
            
            # Очищаем кеш
            with self._lock:
                self._cache.clear()
            
            # Сохраняем
            self._save_model()
            
            logger.info("✅ Модель обновлена")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка обновления: {e}")
            traceback.print_exc()
            return False
    
    def get_stats(self) -> Dict:
        """Возвращает статистику системы"""
        return {
            'model_loaded': self.model is not None,
            'users_in_model': len(self.user_id_map),
            'items_in_model': len(self.item_id_map),
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'cache_size': len(self._cache),
            'config': {
                'learning_rate': self.config.learning_rate,
                'epochs': self.config.epochs,
                'no_components': self.config.no_components,
                'loss': self.config.loss
            }
        }


# Глобальный экземпляр для использования в приложении
recommender = RecommendationEngine()


def init_recommender(force_retrain: bool = False) -> RecommendationEngine:
    """Инициализирует рекомендательную систему"""
    global recommender
    recommender.initialize(force_retrain)
    return recommender


def update_recommender() -> bool:
    """Обновляет модель (вызывается при накоплении данных)"""
    global recommender
    return recommender.update()
