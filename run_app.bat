@echo off
echo =========================================================
echo  DeepFake Detection Platform - Starting Services
echo =========================================================

REM Create properly encoded __init__.py files
echo Creating properly encoded __init__.py files...
echo """DeepFake Detection Platform Main Module""" > backend\app\__init__.py
echo """API Module""" > backend\app\api\__init__.py
echo """API Endpoints Module""" > backend\app\api\endpoints\__init__.py
echo """Core Module""" > backend\app\core\__init__.py
echo """Database Module""" > backend\app\db\__init__.py
echo """ML Models Module""" > backend\app\models\__init__.py
echo """Schemas Module""" > backend\app\schemas\__init__.py
echo """Services Module""" > backend\app\services\__init__.py
echo """Async Tasks Module""" > backend\app\tasks\__init__.py
echo """Utility Functions Module""" > backend\app\utils\__init__.py

REM Start Docker Compose
echo Starting Docker services...
docker-compose down
docker-compose up -d

echo =========================================================
echo  DeepFake Detection Platform is now running!
echo  Frontend: http://localhost:3000
echo  API: http://localhost:8000
echo  API Docs: http://localhost:8000/docs
echo =========================================================
echo.
echo Press Ctrl+C to stop viewing logs, or close this window.
echo.

REM Show container logs
docker-compose logs -f