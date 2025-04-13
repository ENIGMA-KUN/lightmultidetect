from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "worker",
    backend=settings.CELERY_RESULT_BACKEND,
    broker=settings.CELERY_BROKER_URL
)

celery_app.conf.task_routes = {
    "app.tasks.image_tasks.*": {"queue": "image_queue"},
    "app.tasks.audio_tasks.*": {"queue": "audio_queue"},
    "app.tasks.video_tasks.*": {"queue": "video_queue"},
    "app.tasks.result_tasks.*": {"queue": "result_queue"},
}

celery_app.conf.result_expires = 60 * 60 * 24  # 1 day