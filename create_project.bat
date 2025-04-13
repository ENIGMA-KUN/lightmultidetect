@echo off
echo Creating DeepFake Detection Project Structure...

REM Create main directories
mkdir deepfake-detection
cd deepfake-detection
mkdir uploads
mkdir visualizations
mkdir scripts

REM Create backend structure
mkdir backend
mkdir backend\app
mkdir backend\app\api
mkdir backend\app\api\endpoints
mkdir backend\app\core
mkdir backend\app\db
mkdir backend\app\models
mkdir backend\app\models\weights
mkdir backend\app\schemas
mkdir backend\app\services
mkdir backend\app\tasks
mkdir backend\app\utils

REM Create frontend structure
mkdir frontend
mkdir frontend\public
mkdir frontend\public\assets
mkdir frontend\public\assets\images
mkdir frontend\src
mkdir frontend\src\components
mkdir frontend\src\components\common
mkdir frontend\src\components\upload
mkdir frontend\src\components\analysis
mkdir frontend\src\components\results
mkdir frontend\src\contexts
mkdir frontend\src\hooks
mkdir frontend\src\pages
mkdir frontend\src\services
mkdir frontend\src\styles
mkdir frontend\src\types

REM Create backend files
echo // Add code here > backend\app\api\__init__.py
echo // Add code here > backend\app\api\endpoints\__init__.py
echo // Add code here > backend\app\api\endpoints\detection.py
echo // Add code here > backend\app\api\endpoints\results.py
echo // Add code here > backend\app\api\endpoints\upload.py
echo // Add code here > backend\app\api\router.py
echo // Add code here > backend\app\core\__init__.py
echo // Add code here > backend\app\core\config.py
echo // Add code here > backend\app\core\events.py
echo // Add code here > backend\app\db\__init__.py
echo // Add code here > backend\app\db\models.py
echo // Add code here > backend\app\db\session.py
echo // Add code here > backend\app\models\__init__.py
echo // Add code here > backend\app\models\audio_models.py
echo // Add code here > backend\app\models\ensemble.py
echo // Add code here > backend\app\models\image_models.py
echo // Add code here > backend\app\models\video_models.py
echo // Add code here > backend\app\schemas\__init__.py
echo // Add code here > backend\app\schemas\detection.py
echo // Add code here > backend\app\schemas\results.py
echo // Add code here > backend\app\services\__init__.py
echo // Add code here > backend\app\services\detection_service.py
echo // Add code here > backend\app\services\result_service.py
echo // Add code here > backend\app\tasks\__init__.py
echo // Add code here > backend\app\tasks\audio_tasks.py
echo // Add code here > backend\app\tasks\celery_app.py
echo // Add code here > backend\app\tasks\image_tasks.py
echo // Add code here > backend\app\tasks\result_tasks.py
echo // Add code here > backend\app\tasks\video_tasks.py
echo // Add code here > backend\app\utils\__init__.py
echo // Add code here > backend\app\utils\audio_utils.py
echo // Add code here > backend\app\utils\image_utils.py
echo // Add code here > backend\app\utils\video_utils.py
echo // Add code here > backend\app\utils\visualization.py
echo // Add code here > backend\app\__init__.py
echo // Add code here > backend\app\main.py
echo // Add code here > backend\Dockerfile
echo // Add code here > backend\requirements.txt

REM Create frontend files
echo // Add code here > frontend\public\index.html
echo // Add code here > frontend\public\favicon.ico
echo // Add code here > frontend\src\components\common\Button.tsx
echo // Add code here > frontend\src\components\common\ConfidenceGauge.tsx
echo // Add code here > frontend\src\components\common\Footer.tsx
echo // Add code here > frontend\src\components\common\Navbar.tsx
echo // Add code here > frontend\src\components\common\ProgressIndicator.tsx
echo // Add code here > frontend\src\components\upload\MediaPreview.tsx
echo // Add code here > frontend\src\components\upload\UploadArea.tsx
echo // Add code here > frontend\src\components\upload\UploadOptions.tsx
echo // Add code here > frontend\src\components\analysis\AnalysisOptions.tsx
echo // Add code here > frontend\src\components\analysis\AnalysisProgress.tsx
echo // Add code here > frontend\src\components\analysis\AnalysisSummary.tsx
echo // Add code here > frontend\src\components\results\ConfidenceScore.tsx
echo // Add code here > frontend\src\components\results\ExplanationPanel.tsx
echo // Add code here > frontend\src\components\results\FrequencyAnalysis.tsx
echo // Add code here > frontend\src\components\results\HeatmapView.tsx
echo // Add code here > frontend\src\components\results\ResultsPage.tsx
echo // Add code here > frontend\src\components\results\TemporalAnalysis.tsx
echo // Add code here > frontend\src\contexts\DetectionContext.tsx
echo // Add code here > frontend\src\hooks\useDetection.ts
echo // Add code here > frontend\src\hooks\useResults.ts
echo // Add code here > frontend\src\hooks\useUpload.ts
echo // Add code here > frontend\src\pages\AboutPage.tsx
echo // Add code here > frontend\src\pages\DetectionPage.tsx
echo // Add code here > frontend\src\pages\HomePage.tsx
echo // Add code here > frontend\src\pages\NotFoundPage.tsx
echo // Add code here > frontend\src\pages\ResultPage.tsx
echo // Add code here > frontend\src\services\api.ts
echo // Add code here > frontend\src\services\detection.ts
echo // Add code here > frontend\src\services\results.ts
echo // Add code here > frontend\src\styles\global.css
echo // Add code here > frontend\src\types\detection.ts
echo // Add code here > frontend\src\types\results.ts
echo // Add code here > frontend\src\App.tsx
echo // Add code here > frontend\src\index.tsx
echo // Add code here > frontend\package.json
echo // Add code here > frontend\tsconfig.json
echo // Add code here > frontend\tailwind.config.js
echo // Add code here > frontend\Dockerfile

REM Create script files
echo // Add code here > scripts\download_weights.py

REM Create root configuration files
echo // Add code here > docker-compose.yml
echo // Add code here > .env.example
echo // Add code here > README.md

echo Project structure created successfully!
echo Now you can copy the code for each file from our discussion.
cd ..

REM Create batch file to run the app
echo @echo off > run_app.bat
echo echo Starting DeepFake Detection Platform... >> run_app.bat
echo cd deepfake-detection >> run_app.bat
echo docker-compose up >> run_app.bat
echo. >> run_app.bat

echo All done! You can now:
echo 1. Copy code into each file
echo 2. Run run_app.bat to start the application