# 🌿 SkinGenius AI – Your Kitchen-Based Dermatologist 🤖

**SkinGenius AI** is an AI-powered skincare assistant that helps you address a wide range of skin concerns using affordable, kitchen-friendly ingredients. Whether you're dealing with simple dryness or complex conditions like Alopecia Areata, Atopic Dermatitis, or Rosacea, SkinGenius crafts personalized DIY remedies—no expensive products required. Simply describe your concern or upload an image (optional)—we respect your privacy—and receive scientifically backed, user-friendly solutions. Let your skin's unique beauty shine! 💖

## 📹 Demo Video

[![SkinGenius AI Demo](https://img.youtube.com/vi/KmuFZcWJJjY/maxresdefault.jpg)](https://www.youtube.com/watch?v=KmuFZcWJJjY&t=20s)

> Click the image above to watch the full demo on YouTube.

## 🎯 Project Overview

- **Mission:** Deliver personalized skincare solutions using common household ingredients (e.g., honey, aloe vera, turmeric) to address skin concerns affordably.
- **Scope:** Covers everything from normal skin issues (dryness, oiliness) to severe conditions (Alopecia Areata, Rosacea).
- **Privacy-First:** Optional image uploads are processed securely and deleted immediately.

### Workflow

1. **Listen:** Interpret text or image inputs; ask follow-ups for clarity.
2. **Analyze:** Diagnose conditions via NLP and image classification.
3. **Generate:** Produce tailored DIY mask recipes with kitchen ingredients.
4. **Adapt:** Offer ingredient substitutions if needed.
5. **Guide:** Provide safety tips, usage instructions, and FAQs (e.g., sunlight warnings).

## 🌟 Key Features

| Feature | Description |
|---------|-------------|
| **Text & Image Input** 📝📸 | Accepts text descriptions (e.g., "I have acne") or optional image uploads (deleted post-analysis) |
| **NLP-Driven Analysis** 🧠 | Detects skin types and conditions using Hugging Face Transformers and regex |
| **Emotion-Aware Responses** 😊 | Analyzes emotional context to provide empathetic, humanized advice |
| **Image Classification** 🖼️ | Uses MobileNetV3 and HAM10000 dataset to map medical labels (e.g., `mel`) to cosmetic concerns (e.g., acne) |
| **Personalized Recipes** 🍯 | Generates DIY mask recipes with step-by-step instructions and safety warnings |
| **Ingredient Substitution** 🔄 | Suggests alternatives (e.g., sandalwood for turmeric) via `alternative_ingredients.csv` |
| **Safety & Guidance** ⚠️ | Flags harmful combinations and provides pregnancy-safe and sunlight-safe advice |
| **Scalable Deployment** 🚀 | Containerized with Docker; monitored with Prometheus and MLflow; tested via Swagger UI/Postman |

## 🛠️ Tech Stack

| Component | Technologies |
|-----------|-------------|
| **Data Layer** 📂 | ChromaDB-powered vector database (`ingredients_embeddings.db` derived from `ingredients.csv`), JSON datasets (`skin_issues.json`, `compatibility_rules.json`), CSV (`alternative_ingredients.csv`), HAM10000 dermatology image dataset |
| **AI Core** 🤖 | Hugging Face Transformers, PyTorch (MobileNetV3), Sentence Transformers, NumPy, PIL, Regex |
| **Backend** 🛠️ | FastAPI, Uvicorn, Pydantic, Python-multipart, Logging, MLflow, Prometheus |
| **Frontend** 🌐 | Streamlit (dark theme, custom CSS), Markdown/Emojis |
| **Deployment** 🚀 | Docker, Docker Compose, Bash scripts |

## 📂 Project Structure

```
SkinGenius/
├── AI/                          # Core AI modules
│   ├── nlp_processor.py         # Text parsing and intent detection
│   ├── retriever.py             # Fetches relevant skincare data
│   ├── recipe_generator.py      # Generates personalized skincare recipes
│   ├── fine_tuned.py            # Fine-tuned model logic
│   ├── core.py                  # Main AI processing pipeline
│   ├── skin_classifier.py       # Image-based skin type & condition detection
│   ├── quick_tune.jsonl         # Model quick-tuning data
│   └── memory_cache.json        # Temporary memory cache
├── memory/                      # Persistent chat history (JSON files)
├── data/                        # Datasets and embeddings
│   ├── skin_issues.json         # Mapped skin conditions & solutions
│   ├── ingredients_embeddings.db # ChromaDB vector storage
│   ├── alternative_ingredients.csv # Ingredient substitution data
│   ├── compatibility_rules.json # Ingredient compatibility rules
│   └── skin_images/             # HAM10000 dermatology dataset
│       ├── HAM10000_images_part_1/
│       ├── HAM10000_images_part_2/
│       └── HAM10000_metadata.csv
├── skincare_api/                # Backend API service (FastAPI)
│   ├── main.py                  # API endpoints (/ask, /analyze-skin)
│   └── skingenius_logs/         # API logs and performance metrics
├── frontend/                    # Streamlit-based user interface
│   └── app.py                   # Main frontend app
├── create_embeddings.py         # Converts CSV ingredients into embeddings
├── run.py                       # Local app runner
├── test_all.py                  # Test suite
├── Dockerfile                   # Docker configuration
├── docker-compose.yml           # Multi-container orchestration
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- Optional: GPU for faster PyTorch image classification

### Installation

```bash
# Clone the repository
git clone https://github.com/swairariaz/SkinGenius-AI.git
cd SkinGenius

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Generate ingredient embeddings
python create_embeddings.py
```

### Run Locally

```bash
# Start backend API
uvicorn skincare_api.main:app --host 0.0.0.0 --port 8000

# Start frontend (in a new terminal)
streamlit run frontend/app.py
```

- Access the app at `http://localhost:8501`
- API documentation at `http://localhost:8000/docs`

### Docker Deployment

```bash
# Build and start all services
docker-compose up --build

# Run in detached mode
docker-compose up -d --build
```

### Testing

```bash
# Run unit tests
pytest

# Test API endpoints
# Use Swagger UI at http://localhost:8000/docs
# Or test with Postman/curl
```

## 🧪 API Endpoints

| Endpoint | Method | Description | Parameters |
|----------|--------|-------------|------------|
| `/ask` | POST | Get skincare advice from text input | `{"query": "I have dry skin"}` |
| `/analyze-skin` | POST | Analyze uploaded skin image | Form data with image file |
| `/docs` | GET | Interactive API documentation | - |
| `/health` | GET | Health check endpoint | - |

## 📊 Performance & Monitoring

- **Logging:** Comprehensive logging via Python's logging module
- **Monitoring:** Prometheus metrics for API performance
- **Model Tracking:** MLflow for experiment tracking
- **Testing:** Pytest for unit and integration tests

## 🔒 Privacy & Security

- **Image Privacy:** Uploaded images are processed in memory and immediately deleted
- **Data Protection:** No personal data is stored permanently
- **Secure Processing:** All AI processing happens locally within the container

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- HAM10000 dataset for dermatology images
- Hugging Face for transformer models
- ChromaDB for vector database capabilities
- The open-source community for invaluable tools and libraries

---

💚 *Crafted with passion for healthy, radiant skin using kitchen remedies* 💚

**© 2025 SkinGenius by sAI**
