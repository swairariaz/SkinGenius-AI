🌿 SkinGenius AI - Your Personal Kitchen-Based Dermatologist 🤖

Welcome to SkinGenius AI, an innovative, AI-powered skincare assistant designed to provide personalized, kitchen-friendly remedies for skin concerns ranging from everyday issues like dryness to complex conditions like rosacea or hyperpigmentation. Say goodbye to expensive skincare products—SkinGenius leverages common household ingredients to craft safe, effective DIY solutions tailored to your unique skin needs. Whether you describe your concerns in text or (optionally) upload an image, SkinGenius delivers scientifically backed, user-friendly advice with a touch of care. Your skin is unique and beautiful, and we're here to help it shine! 💖

📹 Demo Video
Watch SkinGenius in action on YouTube to see how it transforms skincare with AI and kitchen remedies!

🎯 Project Overview

SkinGenius AI is a full-stack application that combines natural language processing (NLP), image analysis, and a robust backend to diagnose skin issues, recommend compatible ingredients, and generate DIY mask recipes. Built with scalability and user privacy in mind, it supports both text-based queries and optional image uploads (your choice, respecting your privacy). The project is containerized for easy deployment and tested via Swagger UI and Postman for reliability.





Core Mission: Empower users to address skin concerns using affordable, kitchen-based ingredients (e.g., honey, aloe vera, turmeric) instead of costly commercial products.



Scope: Covers normal skin issues (e.g., dryness, oiliness) to extreme conditions (e.g., Alopecia Areata, Atopic Dermatitis, Rosacea).



Privacy-First: Image uploads are optional, processed securely, and deleted after analysis to protect user data.



Workflow:





Listen: Understands user queries (text or image) and asks follow-ups for clarity.



Analyze: Diagnoses skin conditions using NLP and image classification.



Generate: Creates personalized DIY mask recipes with kitchen ingredients.



Adapt: Suggests ingredient substitutions if items are missing.



Guide: Provides safety tips, usage instructions, and FAQs (e.g., sunlight warnings).

🌟 Key Features





Text & Image Input 📝📸





Describe skin concerns (e.g., "I have acne on my cheeks") or upload an image for enhanced analysis.



Image uploads are optional, ensuring privacy; temporary files are deleted post-processing.



NLP-Driven Analysis 🧠





Detects skin conditions (e.g., acne, rosacea) and skin types (dry, oily, combination) using Hugging Face Transformers and regex-based symptom matching.



Incorporates emotional context (e.g., "I sense frustration") for empathetic, humanized responses.



Image Classification 🖼️





Analyzes skin images using a fine-tuned MobileNetV3 model trained on the HAM10000 dataset.



Maps medical labels (e.g., mel) to cosmetic concerns (e.g., acne, blackheads) with confidence scores.



Personalized Recipes 🍯





Generates DIY mask recipes tailored to skin type and condition, using kitchen ingredients (e.g., yogurt, turmeric).



Includes step-by-step instructions (Prepare, Apply, Remove) and safety warnings (e.g., "Patch test first!").



Ingredient Substitution 🔄





Suggests alternatives (e.g., sandalwood for turmeric) if ingredients are unavailable, using alternative_ingredients.csv.



Safety & Guidance ⚠️





Flags ingredient conflicts (e.g., Vitamin C + Niacinamide) and provides sunlight or pregnancy-related warnings.



Offers usage tips (e.g., application time, frequency) and FAQs for optimal results.



Session Persistence 💾





Maintains conversation history via JSON-based memory caching for seamless follow-up questions.



Scalable Deployment 🚀





Fully containerized with Docker and Docker Compose for easy setup and scalability.



Monitoring & Testing 📊





Tracks API metrics with Prometheus and model performance with MLflow.



Tested via Swagger UI (http://localhost:8000/docs) and Postman for robust endpoint validation.

🛠️ Tech Stack

SkinGenius is built with a modern, robust tech stack to ensure performance, scalability, and user-friendliness:

Data Layer 📂





ChromaDB: Vector database for ingredient embeddings (ingredients_embeddings.db).



JSON: Stores skin issues (skin_issues.json) and compatibility rules (compatibility_rules.json).



CSV: Manages ingredient data (ingredients.csv, alternative_ingredients.csv).



HAM10000 Dataset: Kaggle dataset for image classification, with metadata (HAM10000_metadata.csv) and images (HAM10000_images_part_1/2).

AI Core 🤖





Hugging Face Transformers:





distilbert-base-uncased-emotion: Emotion detection for empathetic responses.



all-MiniLM-L6-v2: Semantic search for conditions and ingredients.



T5 (fine-tuned): Generates DIY mask recipes.



PyTorch: Powers MobileNetV3 for image classification and model training.



Sentence Transformers: Generates embeddings for semantic ingredient retrieval.



NumPy: Handles similarity scoring for NLP.



Regex: Matches symptoms and skin types in user queries.



PIL: Processes images with pure-Python augmentations (no OpenCV dependency).

Backend 🛠️





FastAPI: High-performance REST API for analysis (/analyze) and follow-ups (/followup).



Uvicorn: Asynchronous server for FastAPI.



Pydantic: Validates request/response data with schemas.



Python-multipart: Handles image uploads.



Logging: Tracks errors and performance in skingenius.log.



MLflow: Logs model experiments and performance.



Prometheus: Monitors API metrics.

Frontend 🌐





Streamlit: User-friendly interface with dark theme, custom CSS, and image upload support.



Requests: Communicates with FastAPI endpoints.



UUID: Manages session tracking.



Markdown/Emojis: Enhances response formatting for a friendly UX.

Deployment 🚀





Docker: Containerizes the application for portability.



Docker Compose: Orchestrates FastAPI and Streamlit services.



bash: Runs services concurrently in Docker.

Other Tools 🧰





Pandas: Processes CSV data for embeddings and metadata.



tqdm: Visualizes training progress.



psutil: Monitors CPU and RAM usage.



Swagger UI/Postman: Tests API endpoints (http://localhost:8000/docs).

📂 Project Structure

📂 SkinGenius/
├── 📂 AI/
│   ├── core.py                    # Orchestrates NLP, retriever, and generator
│   ├── nlp_processor.py           # Detects conditions and emotions
│   ├── retriever.py               # Semantic ingredient search
│   ├── recipe_generator.py        # Generates DIY mask recipes
│   ├── skin_classifier.py         # Image-based condition detection
│   ├── fine_tuned.py              # Fine-tuned T5 model logic
│   ├── quick_tune.jsonl           # Fine-tuning data for T5
│   └── memory_cache.json          # Conversation history
├── 📂 data/
│   ├── skin_issues.json           # Skin condition database
│   ├── ingredients.csv            # Ingredient data
│   ├── ingredients_embeddings.db  # ChromaDB embeddings
│   ├── alternative_ingredients.csv # Substitution options
│   ├── compatibility_rules.json   # Ingredient conflict/synergy rules
│   └── 📂 skin_images/            # HAM10000 dataset
├── 📂 skincare_api/
│   ├── main.py                    # FastAPI endpoints
│   ├── schemas.py                 # Pydantic models
│   ├── monitoring.py              # Prometheus/MLflow tracking
│   └── mlflow_logs/               # Experiment logs
├── 📂 frontend/
│   ├── app.py                     # Streamlit UI
│   ├── feedback_system.py         # User feedback collection
│   └── assets/                    # CSS/images for UI
├── 📂 deployment/
│   ├── production.Dockerfile      # Production build
│   └── dev.Dockerfile            # Development build with hot-reload
├── 📜 create_embeddings.py        # Generates ChromaDB embeddings
├── 📜 run.py                      # Launches FastAPI (uvicorn)
├── 📜 test_all.py                 # Unit tests (pytest)
├── 📜 Dockerfile                  # Base Docker configuration
├── 📜 docker-compose.yml          # Orchestrates services
├── 📜 requirements.txt            # Python dependencies
└── 📜 README.md                   # Project documentation

🚀 Getting Started

Prerequisites





Python 3.10+: Ensure Python is installed.



Docker: Required for containerized deployment.



Optional GPU: Recommended for faster image classification (PyTorch).



Dependencies: Listed in requirements.txt.

Installation





Clone the Repository:

git clone https://github.com/your-username/SkinGenius.git
cd SkinGenius



Set Up Virtual Environment (optional for local development):

python -m venv skincare_ai
source skincare_ai/bin/activate  # Linux/Mac
skincare_ai\Scripts\activate    # Windows



Install Dependencies:

pip install -r requirements.txt



Prepare Data:





Place skin_issues.json, ingredients.csv, alternative_ingredients.csv, and compatibility_rules.json in the data/ directory.



Download the HAM10000 dataset from Kaggle and place images/metadata in data/skin_images/.



Generate Embeddings:

python create_embeddings.py

Running Locally





Start FastAPI Backend:

uvicorn skincare_api.main:app --host 0.0.0.0 --port 8000



Start Streamlit Frontend:

streamlit run frontend/app.py



Access the App:





Frontend: http://localhost:8501



API Docs (Swagger UI): http://localhost:8000/docs

Docker Deployment

Project includes full Docker configuration for seamless deployment.

docker compose up

Note: Container may require GPU acceleration for optimal performance (PyTorch-based image classification).

Testing





Unit Tests:

pytest test_all.py



API Testing:





Use Swagger UI at http://localhost:8000/docs or Postman to test /analyze and /followup endpoints.

🧪 Usage Examples





Text Query:





Input: "I have dry, flaky skin on my cheeks."



Output: Diagnosis (dry skin, possible eczema), recipe (e.g., honey + aloe vera mask), and safety tips (e.g., "Patch test first!").



Image Upload (optional):





Upload a skin image for analysis.



Output: Enhanced diagnosis (e.g., mel → acne), confidence score, and tailored recipe.



Follow-Up:





Input: "I don’t have turmeric."



Output: Suggests substitutes (e.g., sandalwood) and updates recipe.

📈 Monitoring & Feedback





Prometheus: Tracks API metrics (e.g., response time, error rates).



MLflow: Logs model performance and experiments.



User Feedback: Collected via feedback_system.py and stored in monitoring.py for continuous improvement.

🛡️ Safety & Privacy





Ingredient Safety: Avoids harmful combinations (e.g., Vitamin C + Niacinamide) using compatibility_rules.json.



Sunlight Warnings: Flags ingredients like turmeric that require evening use.



Pregnancy-Safe: Advises safe alternatives (e.g., oatmeal, honey) for sensitive users.



Privacy: Image uploads are optional, processed temporarily, and deleted after analysis.

🌍 Contributing

We welcome contributions to enhance SkinGenius! To contribute:





Fork the repository.



Create a feature branch (git checkout -b feature/your-feature).



Commit changes (git commit -m "Add your feature").



Push to the branch (git push origin feature/your-feature).



Open a pull request.

Please follow our Code of Conduct and ensure tests pass.

📜 License

This project is licensed under the MIT License. See LICENSE for details.

🙌 Acknowledgments





Hugging Face: For powerful NLP and transformer models.



Kaggle: For the HAM10000 dataset.



xAI: Inspiration for AI-driven innovation.



You: For exploring SkinGenius and embracing your skin’s unique journey! 💖



💚 Built with love for healthy, radiant skin using kitchen remedies! 💚

© 2025 SkinGenius by sAI | Powered by cutting-edge AI
