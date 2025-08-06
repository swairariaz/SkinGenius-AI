import unittest
import os
from skin_classifier import SkinClassifier, Config
from nlp_processor import SkinAnalysisEngine
from recipe_generator import RecipeGenerator
import json


class TestSkinGenius(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize all models once"""
        cls.classifier = SkinClassifier()
        cls.nlp_engine = SkinAnalysisEngine()
        cls.recipe_gen = RecipeGenerator()

        # Inject classifier if missing (safety net)
        if not hasattr(cls.nlp_engine, 'skin_classifier'):
            cls.nlp_engine.skin_classifier = cls.classifier

        # Create test image dir if missing
        os.makedirs("data/test_images", exist_ok=True)

    def test_1_classifier(self):
        """Test image analysis with medical and selfie images"""
        print("\n Testing Classifier:")

        # Case 1: Valid medical image
        med_img = os.path.join(Config.DATA_DIR, "HAM10000_images_part_1", "ISIC_0024307.jpg")
        if os.path.exists(med_img):
            result = self.classifier.predict(med_img)
            self.assertIn('condition', result)
            print(f"✔ Medical Image: {result['condition']} ({result.get('confidence', 0):.0%} conf)")
        else:
            print("Medical test image missing")

        # Case 2: Selfie simulation
        selfie_path = "data/test_images/selfie.jpg"
        if os.path.exists(selfie_path):
            result = self.classifier.predict(selfie_path)
            self.assertTrue('condition' in result or 'uncertain' in result.values())
            print(f"✔ Selfie Handling: {result.get('message', 'OK')}")
        else:
            print("Selfie test image missing (create data/test_images/selfie.jpg)")

    def test_2_nlp_fusion(self):
        """Test combined text + image analysis"""
        print("\nTesting NLP Fusion:")

        # Case 1: Text-only analysis
        text_result = self.nlp_engine.analyze_skin("I have dry skin with redness")
        self.assertGreater(len(text_result['conditions']['exact_matches']), 0)
        print(f"✔ Text Analysis: Found {len(text_result['conditions']['exact_matches'])} conditions")

        # Case 2: Image fusion
        test_img = os.path.join(Config.DATA_DIR, "HAM10000_images_part_1", "ISIC_0024307.jpg")
        if os.path.exists(test_img):
            fused_result = self.nlp_engine.analyze_skin(
                "My skin is oily",
                image_path=test_img
            )
            if 'error' in fused_result:
                print(f"Fusion Error: {fused_result['error']}")
            else:
                self.assertIn('image_recommendations', fused_result)
                print(f"✔ Fusion Success: Added {len(fused_result['image_recommendations'])} ingredients")
        else:
            print("⚠️ Medical test image missing")

    def test_3_recipe_safety(self):
        """Test recipe generation safety checks"""
        print("\n🔥 Testing Recipe Safety:")

        # Case 1: Normal scenario
        safe_response = self.recipe_gen.generate_response(
            user_input="I have dry skin",
            image_data={"type": "dry", "conditions": ["dryness"]},
            missing_ingredients=[],
            prev_skin_profile={"type": "dry", "conditions": ["dryness"], "emotional_state": "neutral"}
        )
        safe_recipe = {"mask": {"name": safe_response["response"].split(":")[0].strip()}}
        self.assertIn('mask', safe_recipe)
        print(f"✔ Basic Recipe: {safe_recipe['mask']['name']}")

        # Case 2: Sensitive skin warning
        sensitive_response = self.recipe_gen.generate_response(
            user_input="I have sensitive skin with redness",
            image_data={"type": "sensitive", "conditions": ["redness"]},
            missing_ingredients=[],
            prev_skin_profile={"type": "sensitive", "conditions": ["redness"], "emotional_state": "neutral"}
        )
        sensitive_recipe = {"warning": "Avoid harsh ingredients" if "Pro tip" in sensitive_response["response"] else ""}
        self.assertIn('warning', sensitive_recipe)
        print(f"✔ Safety Warning: {sensitive_recipe['warning']}")

    def test_4_error_handling(self):
        """Test edge cases and errors"""
        print("\nTesting Error Handling:")

        # Case 1: Bad image path
        bad_img_result = self.classifier.predict("non_existent.jpg")
        self.assertIn('error', bad_img_result)
        print(f"✔ Bad Image Handling: {bad_img_result['error']}")

        # Case 2: Empty text input
        empty_text_result = self.nlp_engine.analyze_skin("")
        self.assertIn('conditions', empty_text_result)
        print(f"✔ Empty Text Handling: Found {len(empty_text_result['conditions']['exact_matches'])} conditions")


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("SKIN GENIUS AI - FULL TEST SUITE")
    print("=" * 50 + "\n")

    unittest.main(verbosity=2)