// Загрузка профиля
document.addEventListener('DOMContentLoaded', async function() {
    console.log('Profile page loaded');
    
    // Получаем ID пользователя из URL или meta
    const pathParts = window.location.pathname.split('/');
    let userId = null;
    
    if (pathParts[1] === 'profile' && pathParts[2]) {
        userId = parseInt(pathParts[2]);
    } else {
        // Свой профиль - получаем из meta
        const userElement = document.querySelector('meta[name="user-id"]');
        if (userElement) {
            userId = parseInt(userElement.content);
        }
    }
    
    if (userId) {
        await loadProfileData(userId);
    }
});

// Загрузка данных профиля
async function loadProfileData(userId) {
    try {
        // Загружаем теги
        const response = await fetch(`/api/user/${userId}/interests`);
        const data = await response.json();
        
        // Обновляем облако тегов
        updateTagsCloud(data.tags);
    } catch (error) {
        console.error('Error loading profile data:', error);
    }
}

// Обновление облака тегов
function updateTagsCloud(tags) {
    const tagsContainer = document.getElementById('tags-cloud');
    if (!tagsContainer) return;
    
    tagsContainer.innerHTML = '';
    
    const maxCount = Math.max(...Object.values(tags), 1);
    
    for (const [tag, count] of Object.entries(tags)) {
        const size = 0.8 + (count / maxCount) * 1.2;
        const span = document.createElement('span');
        span.className = 'badge bg-primary m-1';
        span.style.fontSize = `${size}em`;
        span.textContent = `#${tag} (${count})`;
        tagsContainer.appendChild(span);
    }
    
    // Если нет тегов
    if (Object.keys(tags).length === 0) {
        tagsContainer.innerHTML = '<p class="text-muted">Пока нет прочитанных материалов</p>';
    }
}
