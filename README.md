
  <h1>🌿 SkinGenius AI – Your Kitchen-Based Dermatologist 🤖</h1>
  <p><strong>SkinGenius AI</strong> is an AI-powered skincare assistant that helps you address a wide range of skin concerns using affordable, kitchen-friendly ingredients. Whether you’re dealing with simple dryness or complex conditions like Alopecia Areata, Atopic Dermatitis, or Rosacea, SkinGenius crafts personalized DIY remedies—no expensive products required. Simply describe your concern or upload an image (optional)—we respect your privacy—and receive scientifically backed, user-friendly solutions. Let your skin’s unique beauty shine! 💖</p>

  ## 📹 Demo Video

[![Watch the video](https://img.youtube.com/vi/KmuFZcWJJjY/0.jpg)](https://www.youtube.com/watch?v=KmuFZcWJJjY&t=20s)


  <h2 id="overview">🎯 Project Overview</h2>
  <ul>
    <li><strong>Mission:</strong> Deliver personalized skincare solutions using common household ingredients (e.g., honey, aloe vera, turmeric) to address skin concerns affordably.</li>
    <li><strong>Scope:</strong> Covers everything from normal skin issues (dryness, oiliness) to severe conditions (Alopecia Areata, Rosacea).</li>
    <li><strong>Privacy-First:</strong> Optional image uploads are processed securely and deleted immediately.</li>
    <li><strong>Workflow:</strong>
      <ol>
        <li><strong>Listen:</strong> Interpret text or image inputs; ask follow-ups for clarity.</li>
        <li><strong>Analyze:</strong> Diagnose conditions via NLP and image classification.</li>
        <li><strong>Generate:</strong> Produce tailored DIY mask recipes with kitchen ingredients.</li>
        <li><strong>Adapt:</strong> Offer ingredient substitutions if needed.</li>
        <li><strong>Guide:</strong> Provide safety tips, usage instructions, and FAQs (e.g., sunlight warnings).</li>
      </ol>
    </li>
  </ul>

  <h2 id="features">🌟 Key Features</h2>
  <table>
    <thead>
      <tr>
        <th>Feature</th><th>Description</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>Text &amp; Image Input</strong> 📝📸</td>
        <td>Accepts text descriptions (e.g., “I have acne”) or optional image uploads (deleted post-analysis).</td>
      </tr>
      <tr>
        <td><strong>NLP-Driven Analysis</strong> 🧠</td>
        <td>Detects skin types and conditions using Hugging Face Transformers and regex.</td>
      </tr>
      <tr>
        <td><strong>Emotion-Aware Responses</strong> 😊</td>
        <td>Analyzes emotional context to provide empathetic, humanized advice.</td>
      </tr>
      <tr>
        <td><strong>Image Classification</strong> 🖼️</td>
        <td>Uses MobileNetV3 and HAM10000 dataset to map medical labels (e.g., <code>mel</code>) to cosmetic concerns (e.g., acne).</td>
      </tr>
      <tr>
        <td><strong>Personalized Recipes</strong> 🍯</td>
        <td>Generates DIY mask recipes with step-by-step instructions and safety warnings.</td>
      </tr>
      <tr>
        <td><strong>Ingredient Substitution</strong> 🔄</td>
        <td>Suggests alternatives (e.g., sandalwood for turmeric) via <code>alternative_ingredients.csv</code>.</td>
      </tr>
      <tr>
        <td><strong>Safety &amp; Guidance</strong> ⚠️</td>
        <td>Flags harmful combinations and provides pregnancy-safe and sunlight-safe advice.</td>
      </tr>
      <tr>
        <td><strong>Scalable Deployment</strong> 🚀</td>
        <td>Containerized with Docker; monitored with Prometheus and MLflow; tested via Swagger UI/Postman.</td>
      </tr>
    </tbody>
  </table>

  <h2 id="tech-stack">🛠️ Tech Stack</h2>
  <table>
    <thead>
      <tr><th>Component</th><th>Technologies</th></tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>Data Layer</strong> 📂</td>
        <td>ChromaDB, JSON (<code>skin_issues.json</code>, <code>compatibility_rules.json</code>), CSV (<code>ingredients.csv</code>, <code>alternative_ingredients.csv</code>), HAM10000 Dataset</td>
      </tr>
      <tr>
        <td><strong>AI Core</strong> 🤖</td>
        <td>Hugging Face Transformers, PyTorch (MobileNetV3), Sentence Transformers, NumPy, PIL, Regex</td>
      </tr>
      <tr>
        <td><strong>Backend</strong> 🛠️</td>
        <td>FastAPI, Uvicorn, Pydantic, Python-multipart, Logging, MLflow, Prometheus</td>
      </tr>
      <tr>
        <td><strong>Frontend</strong> 🌐</td>
        <td>Streamlit (dark theme, custom CSS), Requests, Markdown/Emojis</td>
      </tr>
      <tr>
        <td><strong>Deployment</strong> 🚀</td>
        <td>Docker, Docker Compose, Bash scripts</td>
      </tr>
    </tbody>
  </table>

  <h2 id="project-structure">📂 Project Structure</h2>
  <ul>
    <li><code>AI/</code>
      <ul>
        <li><code>nlp_processor.py</code></li>
        <li><code>retriever.py</code></li>
        <li><code>recipe_generator.py</code></li>
        <li><code>fine_tuned.py</code></li>
        <li><code>core.py</code></li>
        <li><code>skin_classifier.py</code></li>
        <li><code>quick_tune.jsonl</code></li>
        <li><code>memory_cache.json</code></li>
      </ul>
    </li>
    <li><code>data/</code>
      <ul>
        <li><code>skin_issues.json</code></li>
        <li><code>ingredients.csv</code></li>
        <li><code>ingredients_embeddings.db</code></li>
        <li><code>alternative_ingredients.csv</code></li>
        <li><code>compatibility_rules.json</code></li>
        <li><code>skin_images/</code>
          <ul>
            <li><code>HAM10000_images_part_1/</code></li>
            <li><code>HAM10000_images_part_2/</code></li>
            <li><code>HAM10000_metadata.csv</code></li>
          </ul>
        </li>
      </ul>
    </li>
    <li><code>skincare_api/</code>
      <ul>
        <li><code>main.py</code> (Endpoints: <code>/ask</code>, <code>/analyze-skin</code>)</li>
        <li><code>schemas.py</code></li>
        <li><code>monitoring.py</code></li>
        <li><code>mlflow_logs/</code></li>
      </ul>
    </li>
    <li><code>frontend/</code>
      <ul>
        <li><code>app.py</code></li>
        <li><code>feedback_system.py</code></li>
        <li><code>assets/</code></li>
      </ul>
    </li>
    <li><code>deployment/</code>
      <ul>
        <li><code>production.Dockerfile</code></li>
        <li><code>dev.Dockerfile</code></li>
      </ul>
    </li>
    <li>Root files:
      <ul>
        <li><code>create_embeddings.py</code></li>
        <li><code>run.py</code></li>
        <li><code>test_all.py</code></li>
        <li><code>Dockerfile</code></li>
        <li><code>docker-compose.yml</code></li>
        <li><code>requirements.txt</code></li>
        <li><code>README.md</code> (this document)</li>
      </ul>
    </li>
  </ul>

  <h2 id="getting-started">🚀 Getting Started</h2>
  <h3>Prerequisites</h3>
  <ul>
    <li>Python 3.10+</li>
    <li>Docker & Docker Compose</li>
    <li>Optional GPU for PyTorch image classification</li>
  </ul>

  <h3>Installation</h3>
  <pre><code># Clone repository
git clone https://github.com/your-username/SkinGenius.git
cd SkinGenius

# (Optional) Virtual environment
python -m venv skincare_ai
source skincare_ai/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Prepare data
# Place JSON, CSV, and HAM10000 images in `data/` as described above.

# Generate embeddings
python create_embeddings.py
</code></pre>

  <h3>Running Locally</h3>
  <pre><code># Start backend
uvicorn skincare_api.main:app --host 0.0.0.0 --port 8000

# Start frontend
streamlit run frontend/app.py
</code></pre>
  <p>Access the app at <code>http://localhost:8501</code>, Swagger UI at <code>http://localhost:8000/docs</code>.</p>

  <h3>Docker Deployment</h3>
  <pre><code>docker-compose up
</code></pre>

  <h3>Testing</h3>
  <ul>
    <li>Unit Tests: <code>pytest test_all.py</code></li>
    <li>API Testing: Swagger UI / Postman</li>
  </ul>

  <h2 id="license">📜 License</h2>
  <p>Licensed under the MIT License. See <a href="LICENSE">LICENSE</a> for details.</p>

  <footer>
    <p>💚 <em>Crafted with passion for healthy, radiant skin using kitchen remedies!</em> 💚</p>
    <p>© 2025 SkinGenius by sAI</p>
  </footer>

</body>
</html>
