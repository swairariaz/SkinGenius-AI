"""
SKIN GENIUS - CORE ORCHESTRATOR v2.2
====================================
Purpose:
- Coordinates NLP, Retriever, and Generator modules.
- Maps medical diagnoses to cosmetic recommendations.
- Handles fallback logic for robustness.

Key Technologies:
- PyTorch (for skin classifier)
- ChromaDB (for ingredient retrieval)
- JSON-based memory caching
- MLflow/Prometheus (monitoring)

Workflow:
1. Accepts user query + optional image.
2. NLP Processor → Detects conditions/skin type.
3. Retriever → Fetches compatible ingredients.
4. Generator → Creates DIY mask recipe.
5. Validates recipe steps + safety rules.

Critical Methods:
- `analyze_skin()`: Main analysis pipeline.
- `_map_to_cosmetic()`: Converts medical labels to skincare terms.
- Error handling: Fallback to default recipe if AI fails.
"""


import os
import torch
import json
import re
from typing import Dict, Optional, List
from PIL import Image
from torchvision import transforms
import datetime

# To Monitor Resources
import psutil
print(f"CPU: {psutil.cpu_percent()}% | RAM: {psutil.virtual_memory().percent}%")

# For Better Error Logging
import logging
logging.basicConfig(filename='skingenius.log', level=logging.INFO)

# Import all AI components
from AI.nlp_processor import SkinAnalysisEngine
from AI.retriever import IngredientRetriever
from AI.recipe_generator import RecipeGenerator
from AI.skin_classifier import SkinClassifier

class Config:
    """Centralized configuration"""
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MEDICAL_LABELS = {
        0: "akiec", 1: "bcc", 2: "bkl", 3: "df", 4: "mel", 5: "nv", 6: "vasc"
    }

class SkinGeniusCore:
    def __init__(self):
        """Initialize all AI subsystems"""
        try:
            self.classifier = SkinClassifier()
            self.nlp = SkinAnalysisEngine()
            self.retriever = IngredientRetriever()
            self.generator = RecipeGenerator()
            logging.info("Skincare AI initialized successfully!")
        except Exception as e:
            logging.error(f"Initialization failed: {str(e)}")
            raise

    def analyze_skin(self, user_query: str, image_path: Optional[str] = None) -> Dict:
        try:
            nlp_results = self.nlp.analyze_skin(user_query, image_path)
            logging.info(f"NLP results: {nlp_results}")

            if image_path:
                medical_result = self._analyze_medical_image(image_path)
                cosmetic_result = self._map_to_cosmetic(medical_result, nlp_results)
                nlp_results.update({
                    "medical_analysis": medical_result,
                    "cosmetic_analysis": cosmetic_result
                })
                image_conditions = [c for c in nlp_results["conditions"]["selected"] if
                                    c in nlp_results["conditions"]["exact_matches"]]
                nlp_results["conditions"]["selected"] = image_conditions or nlp_results["conditions"]["exact_matches"]

            conditions_tuple = tuple(nlp_results["conditions"]["selected"])
            ingredients = self.retriever.retrieve_ingredients(
                skin_type=nlp_results["skin_type"],
                conditions=conditions_tuple,
                properties=nlp_results.get("recommendations", ["soothing", "hydrating"])
            )
            logging.info(f"Retrieved ingredients: {[ing['name'] for ing in ingredients['recommended_ingredients']]}")

            skin_profile = {
                "type": nlp_results["skin_type"],
                "conditions": conditions_tuple[0] if conditions_tuple else "unknown",
                "emotional_state": nlp_results["emotional_context"]["emotion"]
            }
            image_data = {"type": nlp_results["skin_type"], "conditions": conditions_tuple} if image_path else None

            response = self.generator.generate_response(
                user_input=user_query,
                image_data=image_data,
                missing_ingredients=[],
                prev_skin_profile=skin_profile,
                recommended_ingredients=ingredients["recommended_ingredients"]
            )
            logging.info(f"Generator response: {response}")

            recipes = {
                "response": response["response"],
                "followup": response["followup"],
                "name": response["recipe"]["name"],
                "steps": response["recipe"]["steps"],
                "safety_warning": response["recipe"]["safety_warning"],
                "usage_time": response["recipe"]["usage_time"],
                "ingredients": response["recipe"]["ingredients"]
            }
            if nlp_results["emotional_context"]["emotion"] in ["anger", "fear"]:
                recipes["safety_override"] = "Ultra-gentle formula recommended"

            required_steps = ["Prepare:", "Apply:", "Remove:"]
            if not all(any(step.startswith(prefix) for step in recipes["steps"]) for prefix in required_steps):
                main_ingredient = recipes["ingredients"][0]
                other_ingredients = " and ".join(recipes["ingredients"][1:])
                recipes["steps"] = [
                    f"Prepare: Mix 1 tbsp {main_ingredient} with 1 tbsp {other_ingredients}",
                    f"Apply: Spread evenly for 10-15 mins",
                    f"Remove: Rinse with lukewarm water"
                ]
                recipes[
                    "response"] = f"{recipes['name']}: {'. '.join(recipes['steps'])}. {recipes['safety_warning']} Apply during {recipes['usage_time']}!"

            return {
                "diagnosis": nlp_results,
                "ingredients": ingredients,
                "recipes": recipes,
                "timestamp": datetime.datetime.now().isoformat()
            }

        except Exception as e:
            logging.error(f"Analysis failed: {str(e)}")
            return {
                "error": str(e),
                "fallback": "Please describe your skin concern or upload an image",
                "recipes": {
                    "response": "Default Mask: Mix aloe vera and yogurt. Apply for 10 minutes. Rinse with water. Patch test first! Apply during evening!",
                    "followup": "What’s next? Any missing ingredients or questions?",
                    "name": "Default Mask",
                    "steps": ["Prepare: Mix aloe vera and yogurt", "Apply: Spread for 10 minutes",
                              "Remove: Rinse with water"],
                    "safety_warning": "Patch test first!",
                    "usage_time": "evening",
                    "ingredients": ["aloe vera", "yogurt"]
                }
            }

    def _analyze_medical_image(self, image_path: str) -> Dict:
        """Analyze medical image using SkinClassifier"""
        try:
            result = self.classifier.predict(image_path)
            if 'error' in result:
                return {"condition": 0, "confidence": 0.0, "probabilities": [0.0] * 7}
            condition_map = {v: k for k, v in Config.MEDICAL_LABELS.items()}
            condition = condition_map.get(result.get('condition', 0), 'unknown')
            return {
                "condition": list(Config.MEDICAL_LABELS.keys()).index(condition) if condition != 'unknown' else 0,
                "confidence": result.get('confidence', 0.0),
                "probabilities": result.get('probabilities', [0.0] * 7)
            }
        except Exception as e:
            logging.error(f"Image analysis failed: {str(e)}")
            return {"condition": 0, "confidence": 0.0, "probabilities": [0.0] * 7}

    def _map_to_cosmetic(self, medical_result: Dict, nlp_results: Dict) -> Dict:
        condition_map = {
            0: ["redness", "dryness"],  # akiec
            1: ["pores", "dullness"],  # bcc
            2: ["dullness"],  # bkl
            3: ["dryness"],  # df
            4: ["acne", "blackheads"],  # mel
            5: ["pigmentation"],  # nv
            6: ["redness"]  # vasc
        }
        selected_conditions = nlp_results["conditions"]["selected"]
        primary_concern = "unknown"
        all_concerns = []
        if selected_conditions:
            primary_concern = selected_conditions[0].lower()
            if primary_concern in ["dark spots", "hyperpigmentation", "post-acne marks"]:
                primary_concern = "pigmentation"
            elif primary_concern in ["acne vulgaris", "blackheads", "inflammatory acne"]:
                primary_concern = "acne"
            elif primary_concern in ["rosacea", "irritation"]:
                primary_concern = "redness"
            elif primary_concern in ["dry skin", "xerosis", "eczema"]:
                primary_concern = "dryness"
            all_concerns = [primary_concern]
        else:
            primary_concern = condition_map.get(medical_result["condition"], ["pigmentation"])[0]
            all_concerns = condition_map.get(medical_result["condition"], ["pigmentation"])
        return {
            "primary_concern": primary_concern,
            "all_concerns": all_concerns
        }

if __name__ == "__main__":
    print("SKINGENIUS CORE ACTIVATED")
    try:
        core = SkinGeniusCore()
        result = core.analyze_skin(
            user_query="How to treat dark spots from old acne?",
            image_path="data/test_images/selfie.jpg"
        )
        print(json.dumps(result, indent=2))
    except Exception as e:
        logging.error(f"Main execution failed: {str(e)}")
        print(json.dumps({"error": str(e), "fallback": "Initialization or analysis failed"}, indent=2))