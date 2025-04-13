from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional

from app.db.session import get_db
from app.db.models import DetectionResult
from app.schemas.detection import DetectionRequest, DetectionResponse, DetectionResult as DetectionResultSchema


router = APIRouter()


@router.get("/status/{task_id}", response_model=dict)
async def get_task_status(task_id: str):
    """Get the status of a detection task."""
    from app.tasks.celery_app import celery_app
    
    task = celery_app.AsyncResult(task_id)
    
    if task.state == 'PENDING':
        response = {
            'status': 'pending',
            'progress': 0,
            'message': 'Task is pending'
        }
    elif task.state == 'PROGRESS':
        response = {
            'status': 'progress',
            'progress': task.info.get('progress', 0),
            'message': task.info.get('message', '')
        }
    elif task.state == 'SUCCESS':
        response = {
            'status': 'success',
            'progress': 100,
            'result_id': task.result,
            'message': 'Task completed successfully'
        }
    elif task.state == 'FAILURE':
        response = {
            'status': 'failure',
            'progress': 0,
            'message': str(task.info)
        }
    else:
        response = {
            'status': task.state,
            'progress': 0,
            'message': str(task.info)
        }
    
    return response


@router.get("/results", response_model=List[DetectionResultSchema])
async def get_recent_results(
    limit: Optional[int] = 10,
    db: Session = Depends(get_db)
):
    """Get recent detection results."""
    results = db.query(DetectionResult).order_by(DetectionResult.created_at.desc()).limit(limit).all()
    return results


@router.get("/results/{result_id}", response_model=DetectionResultSchema)
async def get_result(
    result_id: str,
    db: Session = Depends(get_db)
):
    """Get a specific detection result."""
    result = db.query(DetectionResult).filter(DetectionResult.id == result_id).first()
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")
    return result