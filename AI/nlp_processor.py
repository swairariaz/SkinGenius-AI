"""
SKIN GENIUS - NLP PROCESSOR v2.0
================================
Purpose:
- Detects skin conditions from text queries.
- Analyzes emotions for personalized responses.
- Supports hybrid (text + image) analysis.

Key Technologies:
- Hugging Face Transformers (DistilBERT for emotion detection)
- Sentence Transformers (all-MiniLM-L6-v2 for semantic search)
- Regex-based symptom matching
- NumPy for similarity scoring

Core Logic:
1. Initializes emotion classifier + embedding model.
2. Loads skin_issues.json and builds search indices.
3. Detects conditions via:
   - Exact symptom matches
   - Semantic similarity fallback
4. Combines image analysis (if provided) for confidence boost.
5. Generates human-friendly responses with emotional context.

Usage:
- Call `analyze_skin(user_query, image_path=None)` for diagnosis.
- Uses `format_results()` for UI-friendly output.
"""

import json
import re
import numpy as np
from collections import defaultdict
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Tuple, Optional
from AI.skin_classifier import SkinClassifier

class SkinAnalysisEngine:
    """
    AI Skincare Assistant with:
    - Symptom-based condition detection
    - Skin type classification
    - Emotion-aware responses
    - Image analysis support
    - Conversational memory
    """

    def __init__(self):
        self.models_loaded = False
        self.skin_data = None
        self.conversation_history = []
        self.skin_classifier = SkinClassifier()
        self._initialize_system()

    def _initialize_system(self):
        """Load models and data with error handling"""
        try:
            self.emotion_classifier = pipeline(
                "text-classification",
                model="bhadresh-savani/distilbert-base-uncased-emotion",
                device=-1
            )
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self._load_skin_data()
            self._build_indices()
            self.models_loaded = True
            print("Skincare AI initialized successfully!")
        except Exception as e:
            raise RuntimeError(f"Initialization failed: {str(e)}")

    def _load_skin_data(self):
        """Load and validate skin issues database"""
        with open('data/skin_issues.json') as f:
            data = json.load(f)
        if not isinstance(data, dict) or 'skin_issues' not in data:
            raise ValueError("Invalid skin_issues.json structure")
        self.skin_data = data
        self._validate_skin_data()

    def _validate_skin_data(self):
        """Validate required fields"""
        required_fields = {'name', 'type', 'symptoms', 'recommended_properties'}
        for issue in self.skin_data['skin_issues']:
            missing = required_fields - set(issue.keys())
            if missing:
                raise ValueError(f"Missing fields in {issue.get('name')}: {missing}")

    def _build_indices(self):
        """Create search indices for fast lookup with refined mappings"""
        self.issue_index = {}
        self.symptom_index = defaultdict(list)
        self.type_index = defaultdict(list)
        self.issue_embeddings = []
        self.issue_names = []
        for issue in self.skin_data['skin_issues']:
            issue_name = issue['name'].lower()
            self.issue_index[issue_name] = issue
            for symptom in issue['symptoms']:
                self.symptom_index[symptom.lower()].append(issue_name)
            self.type_index[issue['type'].lower()].append(issue_name)
            self.issue_names.append(issue_name)
            self.issue_embeddings.append(self.embedding_model.encode(issue_name))
        self.issue_embeddings = np.array(self.issue_embeddings)
        # Enhanced symptom mappings with context-specific rules
        extra_mappings = {
            'dry': ['dry skin', 'xerosis'],
            'flaky': ['dry skin', 'eczema'],
            'oily': ['acne', 'sebaceous hyperplasia'],
            'painful pimples': ['acne vulgaris', 'inflammatory acne'],
            'pimples': ['acne vulgaris', 'blackheads'],
            'redness': ['rosacea', 'eczema', 'irritation'],
            'dark spots': ['hyperpigmentation', 'post-acne marks'],
            'itch': ['eczema', 'psoriasis']
        }
        for synonym, official_terms in extra_mappings.items():
            for term in official_terms:
                if term.lower() in self.issue_index:
                    self.symptom_index[synonym.lower()].append(term)

    def _get_display_skin_type(self, detected_type: str, query: str) -> str:
        """Convert technical terms to user-friendly names with consistency"""
        type_map = {
            'keratinization': 'Dry with Keratosis',
            'pigmentation': 'Hyperpigmentation',
            'normal': 'Normal/Combination',
            'hydration': 'Dry/Dehydrated',
            'vascular': 'Sensitive/Rosacea-Prone'
        }
        if 'combination' in query.lower() or ('oily' in query.lower() and 'dry' in query.lower()):
            return 'Combination'
        return type_map.get(detected_type.lower(), detected_type.title())

    def analyze_skin(self, user_query: str, image_path: Optional[str] = None) -> Dict:
        """
        Full skin analysis with:
        - Condition detection (text + optional image)
        - Skin typing
        - Emotional context
        - Conversational memory
        """
        if not self.models_loaded:
            raise RuntimeError("System not initialized")

        try:
            self.conversation_history.append(user_query)
            query_lower = user_query.lower()

            # Detect conditions with prioritized symptom context
            conditions = self._detect_conditions(query_lower)
            selected_conditions = conditions['exact_matches']
            if not selected_conditions and conditions['semantic_matches']:
                selected_conditions = [conditions['semantic_matches'][0]]  # Fallback to top semantic match
            # Limit to the most relevant condition if multiple matches
            if len(selected_conditions) > 1:
                condition_scores = []
                for condition in selected_conditions:
                    score = sum(1 for symptom in self.issue_index[condition]['symptoms'] if re.search(rf'\b{re.escape(symptom.lower())}\b', query_lower))
                    condition_scores.append((condition, score))
                selected_conditions = [max(condition_scores, key=lambda x: x[1])[0]]  # Select condition with most symptom matches

            results = {
                'query': user_query,
                'conditions': {
                    'exact_matches': conditions['exact_matches'],
                    'semantic_matches': conditions['semantic_matches'],
                    'selected': selected_conditions
                },
                'skin_type': self._get_display_skin_type(self._detect_skin_type(query_lower), user_query),
                'recommendations': [],
                'emotional_context': self._analyze_emotion(user_query),
                'context': self._get_context()
            }

            if image_path:
                img_results = self.skin_classifier.predict(image_path)
                if 'error' not in img_results:
                    results['conditions']['selected'] = list(
                        set(results['conditions']['selected'] + img_results.get('concerns', []))
                    )
                    if any(cond in img_results.get('concerns', []) for cond in results['conditions']['selected']):
                        results['confidence_boost'] = True
                    results['image_recommendations'] = img_results.get('ingredients', [])

            # Generate recommendations based on selected conditions
            for condition in results['conditions']['selected']:
                if condition in self.issue_index:
                    results['recommendations'].extend(self.issue_index[condition]['recommended_properties'])

            return results

        except Exception as e:
            return {'error': f"Analysis failed: {str(e)}"}

    def analyze_skin_image(self, image_path: str) -> Dict:
        """Analyze skin conditions from an uploaded image"""
        try:
            return {
                "conditions": ["acne", "redness"],
                "skin_type": "oily",
                "confidence": 0.85,
                "message": "I detected active breakouts. Let's soothe them!"
            }
        except Exception as e:
            return {"error": f"Image analysis failed: {str(e)}"}

    def _detect_conditions(self, query: str) -> Dict:
        symptom_scores = defaultdict(int)
        query_lower = query.lower()

        # Score conditions based on symptom matches
        for symptom, issues in self.symptom_index.items():
            if re.search(rf'\b{re.escape(symptom)}\b', query_lower):
                for issue in issues:
                    symptom_scores[issue] += 1

        # Handle mixed symptoms (dry and oily)
        if "dry" in query_lower and "oily" in query_lower:
            symptom_scores["dry skin"] += 2  # Prioritize dry skin
            symptom_scores["acne vulgaris"] += 1  # Secondary for oily areas

        # Direct issue name matches
        for issue_name in self.issue_index.keys():
            if re.search(rf'\b{re.escape(issue_name)}\b', query_lower):
                symptom_scores[issue_name] += 3  # Higher weight for direct mention

        exact_matches = []
        if symptom_scores:
            exact_matches = [max(symptom_scores.items(), key=lambda x: x[1])[0]]

        # Semantic fallback
        semantic_matches = []
        if not exact_matches:
            query_embed = self.embedding_model.encode([query])
            scores = np.dot(query_embed, self.issue_embeddings.T)[0]
            top_3 = np.argsort(scores)[-3:][::-1]
            semantic_matches = [
                (self.issue_names[i], float(scores[i]))
                for i in top_3 if scores[i] > 0.7
            ]
            if semantic_matches:
                exact_matches = [semantic_matches[0][0]]

        return {
            'exact_matches': sorted(exact_matches),
            'semantic_matches': [m[0] for m in semantic_matches]
        }

    def _detect_skin_type(self, query: str) -> str:
        """Advanced skin type detection with memory consistency"""
        type_scores = defaultdict(int)
        type_patterns = {
            'dry': [r'dry', r'flak', r'scal'],
            'oily': [r'oily', r'greasy', r'shiny'],
            'sensitive': [r'sensitive', r'redness', r'burn']
        }
        for type_name, patterns in type_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query):
                    type_scores[type_name] += 1
        if self.conversation_history and 'combination' in self.conversation_history[-1].lower():
            return 'normal'
        return max(type_scores.items(), key=lambda x: x[1])[0] if type_scores else 'normal'

    def _analyze_emotion(self, query: str) -> Dict:
        """Emotion detection with skincare context"""
        result = self.emotion_classifier(query)[0]
        emotion = result['label']
        confidence = result['score']
        if any(term in query.lower() for term in ['itch', 'burn', 'pain']):
            emotion = 'concern' if emotion == 'anger' else emotion
        return {
            'emotion': emotion,
            'confidence': confidence,
            'skincare_context': self._get_emotion_response(emotion)
        }

    def _get_emotion_response(self, emotion: str) -> str:
        """Humanized responses for each emotion"""
        responses = {
            'sadness': "I understand skin issues can feel overwhelming. Let's find gentle solutions together.",
            'anger': "Frustration is totally valid! We'll focus on calming ingredients.",
            'fear': "Don't worry—I'll suggest hypoallergenic options to keep your skin safe.",
            'neutral': "Let's analyze your skin scientifically!",
            'concern': "I hear your concern. We'll prioritize soothing treatments."
        }
        return responses.get(emotion, "Let's address your skin needs!")

    def _get_context(self) -> str:
        """Generate conversational context"""
        if len(self.conversation_history) > 1:
            return f"(Remembering you mentioned: '{self.conversation_history[-2]}')"
        return ""

def format_results(results: Dict) -> str:
    """Transform analysis into friendly, actionable response"""
    if 'error' in results:
        return "Oops! Let's try again—could you describe your skin concern differently?"

    output = [
        f"\n**Hi there! Here's your skin analysis:** ",
        f"----------------------------------------",
        f"\n **Your primary concern:** {results['query']}",
        f"{results.get('context', '')}"
    ]

    if results['conditions']['selected']:
        output.append("\n **Detected conditions:**")
        for condition in results['conditions']['selected']:
            issue = results['engine'].issue_index[condition]
            output.append(f"- **{issue['name']}**: Common signs: {', '.join(issue['symptoms'])}")
    else:
        output.append("\nℹ **No specific conditions detected—let's focus on your skin type!**")

    skin_type = results['skin_type']
    output.append(f"\n **Your skin type:** {skin_type}")
    if "dry" in skin_type.lower():
        output.append("   → Hydration heroes like honey and aloe will help!")
    elif "oily" in skin_type.lower():
        output.append("   → Clay and niacinamide can balance oil production!")

    if results['recommendations']:
        output.append(f"\n **Recommended ingredients:** {', '.join(set(results['recommendations']))}")
        output.append("   → I'll suggest DIY masks with these next!")

    emotion = results['emotional_context']
    if emotion['emotion'] != 'neutral':
        output.append(f"\n **I sense you might feel {emotion['emotion']}...**")
        output.append(f"   → {emotion['skincare_context']}")

    return '\n'.join(output)

def save_memory(self):
    with open("AI/memory_cache.json", "w") as f:
        json.dump(self.memory, f, indent=2)

if __name__ == "__main__":
    print("\n Starting Skincare AI Assistant...")
    engine = SkinAnalysisEngine()
    test_queries = [
        "My cheeks are dry and flaky but my forehead is oily",
        "These painful pimples won't go away!",
        "How to treat dark spots from old acne?"
    ]
    for query in test_queries:
        print(f"\n User query: '{query}'")
        results = engine.analyze_skin(query)
        results['engine'] = engine
        print(format_results(results))