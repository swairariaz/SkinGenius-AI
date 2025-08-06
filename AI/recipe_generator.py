"""
SKIN GENIUS - RECIPE GENERATOR v2.3
===================================
Purpose:
- Generates DIY mask recipes using fine-tuned T5.
- Handles ingredient substitutions.
- Provides safety guidelines.

Key Technologies:
- Hugging Face T5 (fine-tuned for skincare)
- ChromaDB (for alternative ingredients)
- Rule-based safety checks (sunlight/usage time)

Core Features:
1. Dynamic recipe generation:
   - Adapts to skin type (dry/oily/combo).
   - Follows template structure (Prepare/Apply/Remove).
2. Substitution system:
   - Uses CSV/common swaps for missing ingredients.
3. Safety layer:
   - Flags conflicts (e.g., "Avoid sunlight after turmeric").
   - Pregnancy/irritation warnings.

Memory:
- Caches last recipe for follow-up questions.
"""

from typing import List, Dict, Optional
from enum import Enum
from transformers import T5Tokenizer, T5ForConditionalGeneration
import json
import os
import re
import chromadb
from sentence_transformers import SentenceTransformer
import pandas as pd


class RecipeType(Enum):
    MASK = "mask"


class SafetyLevel(Enum):
    SAFE = 1
    CAUTION = 2
    ADVANCED = 3


class RecipeGenerator:
    def __init__(self):
        self.templates = {}
        self.safety_rules = {}
        self.common_swaps = {
            "turmeric": ["sandalwood", "multani mitti"],
            "honey": ["raw honey", "yogurt"],
            "thyme": ["neem oil", "aloe vera"],
            "tea tree oil": ["neem oil", "aloe vera"],
            "shea butter": ["coconut oil", "avocado"],
            "rose water": ["chamomile water", "aloe vera"],
            "papaya": ["banana", "yogurt"],
            "activated charcoal": ["multani mitti", "kaolin clay"],
            "kaolin clay": ["multani mitti", "oat flour"],
            "manuka honey": ["raw honey", "yogurt"],
            "oat milk": ["almond milk", "coconut milk"],
            "sanwa (little millet)": "arrowroot powder",
            "jojoba oil": "coconut oil",
            "tinda (apple gourd)": "zucchini",
            "aloe vera": ["cucumber", "cold milk"],
            "turmeric": ["sandalwood powder", "lemon juice"],
            "honey": ["neem powder", "turmeric"],
        }
        try:
            self.client = chromadb.PersistentClient(path="data/ingredients_embeddings.db")
            self.collection = self.client.get_collection(name="ingredients")
            with open("data/skin_issues.json", "r", encoding="utf-8") as f:
                skin_data = json.load(f)
                self.skin_issues = {}
                for i, issue in enumerate(skin_data.get("skin_issues", [])):
                    name = issue.get("name", f"issue_{i}")
                    if name not in self.skin_issues:
                        self.skin_issues[name] = {
                            "name": name,
                            "skin_type": self._map_type(issue.get("type", "unknown")),
                            "suggested_ingredients": issue.get("kitchen_hints", {}).get("ingredients", ["aloe vera"]),
                            "avoid_ingredients": issue.get("kitchen_hints", {}).get("avoid", []),
                            "symptoms": issue.get("symptoms", [])
                        }
            with open("data/compatibility_rules.json", "r", encoding="utf-8") as f:
                self.compatibility_rules = json.load(f)
        except FileNotFoundError as e:
            raise Exception(f"Data file not found: {e}")
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON in data files: {e}")
        except Exception as e:
            raise Exception(f"Failed to load data files: {e}")
        self._load_model()
        self.memory_cache_path = "data/memory_cache.json"
        self.memory = self._load_memory()
        self._ensure_memory_file()
        self._load_data()

    def _map_type(self, issue_type: str) -> str:
        if isinstance(issue_type, list):
            issue_type = issue_type[0]
        type_map = {
            "inflammatory": "oily",
            "hydration": "dry",
            "pigmentation": "combination",
            "sebaceous": "oily",
            "acute damage": "dry",
            "textural": "combination",
            "reactivity": "dry",
            "comedonal": "oily",
            "vascular": "combination",
            "irritation": "dry",
            "chronic inflammation": "dry",
            "dermatitis": "dry",
            "hyperpigmentation": "combination",
            "facial rash": "dry",
            "follicular": "oily",
            "pigmented": "combination",
            "urticaria": "dry",
            "viral": "dry",
            "fungal": "dry",
            "bacterial": "oily"
        }
        return type_map.get(issue_type.lower(), "unknown")

    def _load_data(self):
        self._load_recipe_templates()
        self._init_safety_guidelines()

    def _load_recipe_templates(self):
        self.templates = {
            RecipeType.MASK: {
                "dry": {"name": "Hydrating {base} Mask",
                        "steps": ["Mix ingredients", "Apply for 15 minutes", "Rinse with water"]},
                "oily": {"name": "Purifying {base} Mask",
                         "steps": ["Combine ingredients", "Apply for 10 minutes", "Rinse with cool water"]},
                "combination": {"name": "Balancing {base} Mask",
                                "steps": ["Blend ingredients", "Apply for 12 minutes", "Rinse with lukewarm water"]}
            }
        }

    def _init_safety_guidelines(self):
        self.safety_rules = {
            "dry": {"avoid": ["alcohol", "fragrance"], "sunlight_wait": 1},
            "oily": {"avoid": ["heavy oils"], "sunlight_wait": 2},
            "combination": {"avoid": ["strong acids"], "sunlight_wait": 1}
        }

    def _load_model(self):
        try:
            self.tokenizer = T5Tokenizer.from_pretrained("./models/fine_tuned_model")
            self.model = T5ForConditionalGeneration.from_pretrained("./models/fine_tuned_model")
        except FileNotFoundError as e:
            raise Exception(f"Model files not found in ./models/: {e}")
        except Exception as e:
            raise Exception(f"Failed to load fine-tuned model: {e}")

    def _load_memory(self):
        if os.path.exists(self.memory_cache_path):
            try:
                with open(self.memory_cache_path, "r", encoding="utf-8") as f:
                    memory = json.load(f)
                    return {
                        "user_preferences": memory.get("user_preferences", {}),
                        "last_recipe": memory.get("last_recipe", None),
                        "chat_history": memory.get("chat_history", [])
                    }
            except json.JSONDecodeError:
                return {"user_preferences": {}, "last_recipe": None, "chat_history": []}
        return {"user_preferences": {}, "last_recipe": None, "chat_history": []}

    def _ensure_memory_file(self):
        os.makedirs(os.path.dirname(self.memory_cache_path), exist_ok=True)
        if not os.path.exists(self.memory_cache_path):
            with open(self.memory_cache_path, "w", encoding="utf-8") as f:
                json.dump(self.memory, f)

    def _save_memory(self):
        with open(self.memory_cache_path, "w", encoding="utf-8") as f:
            json.dump(self.memory, f)

    def _analyze_skin(self, prompt: str, image_data: Optional[Dict] = None) -> Dict:
        if image_data:
            skin_type = self._map_type(image_data.get("type", "unknown"))
            conditions = image_data.get("conditions", ["unknown"])
        else:
            skin_type = "unknown"
            conditions = ["unknown"]
            prompt_lower = prompt.lower().strip()
            for issue_name in self.skin_issues:
                if issue_name.lower() in prompt_lower:
                    skin_type = self._map_type(self.skin_issues[issue_name]["skin_type"])
                    conditions = [issue_name]
                    break
            if skin_type == "unknown":
                for issue_name in self.skin_issues:
                    symptoms = self.skin_issues[issue_name].get("symptoms", [])
                    if any(re.search(r'\b' + re.escape(symptom.lower()) + r'\b', prompt_lower) for symptom in symptoms):
                        skin_type = self._map_type(self.skin_issues[issue_name]["skin_type"])
                        conditions = [issue_name]
                        break
        return {"type": skin_type, "conditions": conditions[0] if conditions else "unknown"}

    def _get_similar_ingredients(self, query: str, n_results: int = 3) -> List[Dict]:
        try:
            model = SentenceTransformer("all-MiniLM-L6-v2")
            query_embedding = model.encode([query]).tolist()
            results = self.collection.query(query_embeddings=query_embedding, n_results=n_results)
            return [
                {
                    "name": results["documents"][0][i],
                    "benefits": results["metadatas"][0][i].get("benefits", []),
                    "avoid_with": results["metadatas"][0][i].get("avoid_with", []),
                    "score": results["distances"][0][i]
                }
                for i in range(min(n_results, len(results["documents"][0])))
            ]
        except Exception as e:
            print(f"Warning: Failed to query ingredients: {e}")
            # Use condition-specific ingredients from skin_issues.json
            condition = query.lower()
            fallback_ingredients = self.skin_issues.get(condition, {}).get("suggested_ingredients",
                                                                           ["turmeric", "papaya", "rose water"])
            return [
                {"name": ing, "benefits": ["brightening" if "pigmentation" in condition else "soothing"],
                 "avoid_with": [], "score": 1.0}
                for ing in fallback_ingredients[:n_results]
            ]

    def _generate_ai_recipe(self, recipe_type: RecipeType, skin_profile: Dict,
                            recommended_ingredients: Optional[List[Dict]] = None) -> Dict:
        if not skin_profile["conditions"] or skin_profile["conditions"] == "unknown":
            return self._default_recipe(recipe_type)

        condition = skin_profile["conditions"]
        issue = self.skin_issues.get(condition, {})
        skin_type = skin_profile["type"]
        emotional_state = skin_profile.get("emotional_state", "neutral")

        selected_ingredients = []
        if recommended_ingredients:
            filtered_ingredients = []
            required_properties = issue.get("recommended_properties",
                                            ["brightening", "exfoliating"] if "pigmentation" in condition.lower() else [
                                                "soothing", "hydrating"])
            for ing in recommended_ingredients:
                avoid_conditions = ing.get("avoid_with", [])
                if not any(
                        condition.lower() in avoid.lower() or
                        skin_type.lower() in avoid.lower() or
                        (emotional_state == "fear" and "sensitive skin" in avoid.lower())
                        for avoid in avoid_conditions
                ):
                    matching_props = sum(1 for prop in ing.get("benefits", []) if prop in required_properties)
                    filtered_ingredients.append((ing, matching_props))
            filtered_ingredients.sort(key=lambda x: (x[1], x[0].get("score", 0)), reverse=True)
            selected_ingredients = [ing["name"] for ing, _ in filtered_ingredients[:3]]

            # Fallback to skin_issues.json if no suitable ingredients
            if not selected_ingredients:
                selected_ingredients = issue.get("suggested_ingredients", ["turmeric", "papaya"])[:3]
            while len(selected_ingredients) < 2:
                selected_ingredients.append(
                    issue.get("suggested_ingredients", ["rose water"])[len(selected_ingredients)])
        else:
            selected_ingredients = issue.get("suggested_ingredients", ["turmeric", "papaya"])[:3]

        selected_ingredients = list(dict.fromkeys(selected_ingredients))[:3]
        if len(selected_ingredients) < 2:
            selected_ingredients.append("rose water")

        main_ingredient = selected_ingredients[0]
        other_ingredients = " and ".join(selected_ingredients[1:])
        prompt = f"Create a {skin_type} skin {recipe_type.value} recipe for {condition} using {main_ingredient} and {other_ingredients}. Include 3 steps: Prepare, Apply, Remove. Safety tip: Patch test first."
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=128, truncation=True)
        outputs = self.model.generate(**inputs, max_length=128, num_return_sequences=1)
        recipe_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        steps = [s.strip() for s in recipe_text.split(". ") if s.strip().startswith(("Prepare:", "Apply:", "Remove:"))][
                :3]
        if len(steps) < 3:
            steps = [
                f"Prepare: Mix 1 tbsp {main_ingredient} with 1 tbsp {other_ingredients}",
                f"Apply: Spread evenly for 10-15 mins",
                f"Remove: Rinse with lukewarm water"
            ]

        return {
            "name": f"{condition.capitalize()} Relief Mask",
            "ingredients": selected_ingredients,
            "steps": steps,
            "safety_warning": "Patch test first!",
            "usage_time": "evening"
        }

    def _default_recipe(self, recipe_type: RecipeType) -> Dict:
        return {
            "name": "Default Relief Mask",
            "ingredients": ["honey", "yogurt"],
            "steps": ["Prepare: Mix 1 tbsp honey with 1 tbsp yogurt", "Apply: Spread for 10 mins",
                      "Remove: Rinse with cool water"],
            "safety_warning": "Patch test first!",
            "usage_time": "evening"
        }

    def _check_compatibility(self, ingredients: List[str]) -> List[str]:
        conflicts = []
        with open("data/compatibility_rules.json", "r") as f:
            rules = json.load(f)
        for i, ing_a in enumerate(ingredients):
            for ing_b in ingredients[i + 1:]:
                for conflict in rules["conflicts"]:
                    if {ing_a.lower(), ing_b.lower()} == {conflict["ingredient_A"].lower(),
                                                          conflict["ingredient_B"].lower()}:
                        conflicts.append(f"{ing_a} + {ing_b}: {conflict['reason']}")
        return conflicts

    def _get_alternative_ingredients(self, missing_ingredient: str, skin_condition: str) -> List[str]:
        """Fetch alternative ingredients based on skin condition and missing ingredient."""
        try:
            missing_ingredient = missing_ingredient.lower().strip()
            skin_condition = skin_condition.lower().strip()

            print(f"Looking for alternatives to '{missing_ingredient}' for condition '{skin_condition}'")

            if not os.path.exists("data/alternative_ingredients.csv"):
                print("CSV file not found, using common swaps")
                return self.common_swaps.get(missing_ingredient, [])

            df = pd.read_csv("data/alternative_ingredients.csv", encoding='utf-8')

            # Normalize all text data
            df = df.apply(lambda x: x.str.lower().str.strip() if x.dtype == "object" else x)

            # First try exact match for ingredient
            exact_match = df[df['name'] == missing_ingredient]

            # If no exact match, try partial match
            if exact_match.empty:
                exact_match = df[df['name'].str.contains(r'\b' + re.escape(missing_ingredient) + r'\b', regex=True)]

            if exact_match.empty:
                print(f"No match found for ingredient '{missing_ingredient}'")
                return self.common_swaps.get(missing_ingredient, [])

            # Prioritize rows that match both ingredient and condition
            condition_matches = exact_match[exact_match['skin_issue'] == skin_condition]

            if not condition_matches.empty:
                alternatives = condition_matches.iloc[0]['alternative_ingredients'].split(',')
                avoid_with = condition_matches.iloc[0]['avoid_with']
                print(f"Found condition-specific alternatives: {alternatives}")
            else:
                # Fall back to any alternative for this ingredient
                alternatives = exact_match.iloc[0]['alternative_ingredients'].split(',')
                avoid_with = exact_match.iloc[0]['avoid_with']
                print(f"Using general alternatives: {alternatives}")

            # Filter out alternatives that should be avoided with this condition
            filtered_alternatives = [
                alt.strip() for alt in alternatives
                if avoid_with != skin_condition and alt.strip()
            ]

            if not filtered_alternatives:
                print("All alternatives are avoided for this condition")
                return self.common_swaps.get(missing_ingredient, [])

            return filtered_alternatives

        except Exception as e:
            print(f"Error in alternative ingredient lookup: {str(e)}")
            return self.common_swaps.get(missing_ingredient, [])

    def _adapt_recipe(self, recipe: Dict, missing_ingredients: List[str]) -> Dict:
        """Adapts a recipe by replacing missing ingredients with suitable alternatives."""
        try:
            if not recipe or not isinstance(recipe, dict):
                raise ValueError("Invalid recipe format")

            if not missing_ingredients:
                return recipe

            # Create a copy to avoid modifying the original
            adapted_recipe = recipe.copy()
            condition = recipe["name"].replace(" Relief Mask", "").lower()
            issue = self.skin_issues.get(condition, {})
            adapted_ingredients = []
            replacement_log = []

            # Process each ingredient
            for ing in recipe["ingredients"]:
                if not any(m.lower() == ing.lower() for m in missing_ingredients):
                    adapted_ingredients.append(ing)
                else:
                    print(f"Replacing missing ingredient: {ing}")
                    alternatives = self._get_alternative_ingredients(ing, condition)

                    # Find first suitable alternative not already in recipe
                    swap = None
                    for alt in alternatives:
                        if alt.lower() not in [a.lower() for a in adapted_ingredients]:
                            swap = alt
                            break

                    # Fallback to common swaps if no alternative found
                    if not swap:
                        swap = self.common_swaps.get(ing.lower(), ing)
                        print(f"Using fallback swap: {swap}")

                    adapted_ingredients.append(swap)
                    replacement_log.append(f"{ing} → {swap}")

            # Ensure we have at least 2 ingredients
            if len(adapted_ingredients) < 2:
                fallbacks = issue.get("suggested_ingredients", ["rose water", "aloe vera"])
                adapted_ingredients.extend(fallbacks[:2 - len(adapted_ingredients)])

            # Generate new steps
            main_ingredient = adapted_ingredients[0] if adapted_ingredients else "yogurt"
            other_ingredients = " and ".join(adapted_ingredients[1:]) if len(adapted_ingredients) > 1 else "yogurt"

            try:
                prompt = (
                    f"Adapt this {condition} skin mask recipe using {main_ingredient} and {other_ingredients}. "
                    f"Provide 3 clear steps: Prepare, Apply, Remove. Keep it simple."
                )
                inputs = self.tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True)
                outputs = self.model.generate(
                    **inputs,
                    max_length=300,
                    num_beams=5,
                    early_stopping=True
                )
                recipe_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Robust step parsing
                steps = []
                step_keywords = ["prepare:", "apply:", "remove:"]
                for line in recipe_text.split(". "):
                    line = line.strip().lower()
                    for kw in step_keywords:
                        if line.startswith(kw) and len(steps) < 3:
                            steps.append(line.capitalize())
                            break

                # Fallback steps if generation failed
                if len(steps) < 3:
                    steps = [
                        f"Prepare: Mix 1 tbsp {main_ingredient} with 1 tbsp {other_ingredients}",
                        f"Apply: Spread evenly for 10-15 minutes",
                        f"Remove: Rinse with {'cool' if 'oily' in condition else 'lukewarm'} water"
                    ]

            except Exception as e:
                print(f"Step generation error: {e}")
                steps = [
                    f"Prepare: Mix 1 tbsp {main_ingredient} with 1 tbsp {other_ingredients}",
                    f"Apply: Leave on for 10-15 minutes",
                    f"Remove: Rinse gently with water"
                ]

            # Build the adapted recipe
            adapted_recipe.update({
                "ingredients": adapted_ingredients,
                "steps": steps,
                "adaptation_notes": replacement_log if replacement_log else None
            })

            return adapted_recipe

        except Exception as e:
            print(f"Critical error in recipe adaptation: {e}")
            # Return a safe default recipe if adaptation fails
            return {
                "name": "Adapted Relief Mask",
                "ingredients": ["yogurt", "honey"],
                "steps": [
                    "Prepare: Mix 1 tbsp yogurt with 1 tbsp honey",
                    "Apply: Spread evenly for 10 minutes",
                    "Remove: Rinse with lukewarm water"
                ],
                "safety_warning": "Patch test first!",
                "usage_time": "evening",
                "adaptation_error": str(e)
            }

    def _guide_response(self, recipe: Dict, question: str) -> str:
        if not recipe:
            return "Hey! Please generate a recipe first."
        if "sunlight" in question.lower():
            wait_time = self.safety_rules.get(recipe["name"].split()[0].lower(), {}).get("sunlight_wait", 1)
            return f"Hey! Avoid sunlight after this mask—it may irritate. Wait {wait_time} to 2 hours and use sunscreen."
        return f"Hey! Apply this mask during {recipe['usage_time']} for best results. Any other questions?"

    def generate_response(self, user_input: str, image_data: Optional[Dict] = None,
                          missing_ingredients: Optional[List[str]] = None,
                          prev_skin_profile: Optional[Dict] = None,
                          recommended_ingredients: Optional[List[Dict]] = None) -> Dict:
        skin_profile = self._analyze_skin(user_input, image_data) if not prev_skin_profile else prev_skin_profile
        recipe = self._generate_ai_recipe(RecipeType.MASK, skin_profile, recommended_ingredients)
        response_text = (
            f"✨ {recipe['name']} for your skin! 🌿\n"
            f"💖 **Ingredients**: {', '.join(recipe['ingredients'])}\n"
            f"🧴 **Steps**:\n"
            f"  1. {recipe['steps'][0]}\n"
            f"  2. {recipe['steps'][1]}\n"
            f"  3. {recipe['steps'][2]}\n"
            f"⚠️ **{recipe['safety_warning']}** Always test on a small patch first! 😊\n"
            f"🌙 **Best Time**: Apply during {recipe['usage_time']} for a glowing result! ✨"
        )
        self.memory["last_recipe"] = recipe
        self._save_memory()
        return {
            "response": response_text,
            "followup": "💕 What’s next? Missing any ingredients or have questions? 🌸",
            "recipe": recipe
        }

    def handle_followup(self, user_input: str) -> dict:
        if not self.memory.get("last_recipe"):
            return {"response": "Hey! Please generate a recipe first by specifying a skin issue. 😊"}

        last_recipe = self.memory["last_recipe"]
        response = ""
        recipe = None
        followup = "Anything else on your mind? I’m here to help! 🌿"

        # Standardize input: replace curly apostrophes and lowercase
        user_input = user_input.replace("’", "'").lower().strip()

        # Check for missing ingredients
        if any(term in user_input for term in ["don't have", "missing", "no"]):
            recipe_ingredients = [ing.lower() for ing in last_recipe["ingredients"]]
            missing_ingredients = []

            # Extract the ingredient the user mentions with improved matching
            match = re.search(r"(don't have|missing|no)\s+(.+?)(?:\s|$)", user_input)
            if match:
                user_mentioned = match.group(2).strip()
                # Find the closest matching ingredient
                for ing in last_recipe["ingredients"]:
                    if user_mentioned in ing.lower() or any(word in ing.lower() for word in user_mentioned.split()):
                        missing_ingredients.append(ing)
                        break
                if not missing_ingredients:
                    missing_ingredients = [ing for ing in last_recipe["ingredients"] if user_mentioned in ing.lower()]
                    if not missing_ingredients:
                        response = (
                            f"Hmm, I couldn’t identify '{user_mentioned}' as a missing ingredient. 🤔\n"
                            f"- **My Advice**: Please use an ingredient from the recipe (e.g., 'I don’t have {recipe_ingredients[0]}'). 🌿\n"
                            f"- **Next Step**: Let me know the exact ingredient! 💕"
                        )
                        return {"response": response, "followup": followup, "recipe": None}

            if missing_ingredients:
                print(f"Detected missing ingredients: {missing_ingredients}")
                condition = last_recipe["name"].replace(" Relief Mask", "").lower()
                recipe = self._adapt_recipe(last_recipe, missing_ingredients)

                if not recipe or recipe["ingredients"] == last_recipe["ingredients"]:
                    response = (
                        f"Hmm, couldn’t adapt the recipe for {', '.join(missing_ingredients)}. 🤔\n"
                        f"- **My Advice**: {missing_ingredients[0]} is key for your {last_recipe['name'].replace(' Relief Mask', '')} concern! 🌿\n"
                        f"- **Next Step**: Try getting it or let me know another missing item! 🛒💕"
                    )
                else:
                    steps = recipe.get("steps", ["Mix the new ingredients", "Apply for 10-15 mins", "Rinse with water"])[:3]
                    safety_warning = recipe.get("safety_warning", "Patch test first to avoid irritation.")
                    swapped_ing = next(ing for ing in recipe["ingredients"] if ing.lower() not in [m.lower() for m in last_recipe["ingredients"]])
                    response = (
                        f"Hey there! 😊 I see you don’t have {', '.join(missing_ingredients)}—no stress, we’ve got this!\n"
                        f"- **Good News**: I’ve swapped it with {swapped_ing} for your {recipe['name']}.\n"
                        f"- **How to Use It**:\n"
                        f"  1. {steps[0]}\n"
                        f"  2. {steps[1]}\n"
                        f"  3. {steps[2]}\n"
                        f"- **Safety Tip**: {safety_warning} ✅\n"
                        f"- **Best Time**: Apply during {recipe.get('usage_time', 'evening')} for the best glow. 🌙\n"
                        f"Let me know how it feels—your skin deserves the best! 💕"
                    )
                    self.memory["last_recipe"] = recipe
                    self._save_memory()
            else:
                response = (
                    f"Hmm, I couldn’t identify a missing ingredient from '{user_input}'. 🤔\n"
                    f"- **My Advice**: Please specify the ingredient you don’t have (e.g., 'I don’t have oatmeal'). 🌿\n"
                    f"- **Next Step**: Let me know, and I’ll find a substitute! 💕"
                )
        elif any(term in user_input for term in ["morning", "apply in the morning"]):
            response = (
                f"Great question! 😊 Here’s my take on using the {last_recipe['name']} in the morning:\n"
                f"- **My Advice**: Evening is best for repair, but morning is okay too!\n"
                f"- **Extra Step**: Use sunscreen to protect your skin. ☀️\n"
                f"Your skin will thank you—let me know if you need more tips! 💡"
            )
        elif any(term in user_input for term in ["sunlight", "sun", "hazardous", "dangerous", "risky"]):
            wait_time = self.safety_rules.get(last_recipe["name"].split()[0].lower(), {}).get("sunlight_wait", 1)
            response = (
                f"Thanks for asking—that’s smart! 😊 Here’s the deal with sunlight and the {last_recipe['name']}:\n"
                f"- **Caution**: Avoid sunlight to prevent irritation.\n"
                f"- **Wait Time**: Wait {wait_time} to 2 hours, then use sunscreen. 🕒☀️\n"
                f"Your skin’s safety is key—reach out if unsure! 🌿"
            )
        elif any(term in user_input for term in ["how often", "frequency", "how many times"]):
            response = (
                f"Love that you’re planning! 😊 Here’s how often for the {last_recipe['name']}:\n"
                f"- **Recommendation**: Once daily (evening) or every other day if sensitive.\n"
                f"- **Tip**: Patch test first! ✅\n"
                f"Got questions? I’m here! 🌸"
            )
        elif any(term in user_input for term in ["how long", "how much time", "duration"]):
            response = (
                f"Great question! 😊 Here’s how long for the {last_recipe['name']}:\n"
                f"- **Time**: 15-20 minutes for best results.\n"
                f"- **Adjustment**: Rinse earlier if dry. 💧\n"
                f"Let me know if you need tweaks! 🌿"
            )
        elif any(term in user_input for term in ["store", "storage", "keep it"]):
            response = (
                f"Good thinking! 😊 Here’s how to store the {last_recipe['name']}:\n"
                f"- **Where**: Airtight container in a cool, dry place.\n"
                f"- **How Long**: Use within 3-5 days. 🕒\n"
                f"- **Tip**: Refrigerate wet masks! ❄️\n"
                f"Let me know if you need more ideas! 🌸"
            )
        elif any(term in user_input for term in ["side effects", "problems", "reaction"]):
            response = (
                f"Smart to check! 😊 Here’s what to watch for with the {last_recipe['name']}:\n"
                f"- **Possible Issues**: Mild redness or itching—stop if it occurs.\n"
                f"- **Prevention**: Patch test first! ✅\n"
                f"- **Next Step**: Tell me if you notice anything odd. 🌿\n"
                f"I’ve got your back! 💕"
            )
        elif any(term in user_input for term in ["skin type", "suitable", "good for"]):
            skin_type = last_recipe.get("target_skin_type", "all types")
            response = (
                f"Love that you’re curious! 😊 Here’s about the {last_recipe['name']} and skin types:\n"
                f"- **Best For**: {skin_type}—great for you!\n"
                f"- **Note**: Tell me your skin type if different, and I’ll adjust. 🌿\n"
                f"- **Tip**: Patch test to confirm! ✅\n"
                f"Want more details? I’m here! 💡"
            )

        elif "evening" in user_input or "night" in user_input:
            response = (
                f"🌙 Evening Application Tips:\n"
                f"- Perfect time! Skin repairs itself at night\n"
                f"- Apply 1 hour before bed to avoid pillow transfer\n"
                f"- Follow with a light moisturizer after rinsing\n"
                f"- Best frequency: 3-4 times weekly for this mask"
            )

        elif "sun" in user_input or "light" in user_input:
            wait_time = self.safety_rules.get(last_recipe["name"].split()[0].lower(), {}).get("sunlight_wait", 1)
            response = (
                f"☀️ Sunlight After Application:\n"
                f"- Wait {wait_time} hour(s) minimum\n"
                f"- Redness risk increases with sun exposure\n"
                f"- Must use SPF 30+ sunscreen after waiting period\n"
                f"- Evening application avoids this concern completely"
            )

        elif "how many times" in user_input or "frequency" in user_input:
            response = (
                f"⏰ Usage Frequency Guide:\n"
                f"- Normal skin: 2-3 times weekly\n"
                f"- Sensitive skin: Once weekly with patch test\n"
                f"- Acute issues: Every other day for 2 weeks max\n"
                f"- Maintenance: Once weekly thereafter"
            )

        elif "how long" in user_input or "duration" in user_input:
            response = (
                f"⏳ Application Duration:\n"
                f"- Ideal time: 15-20 minutes\n"
                f"- Sensitive skin: Start with 5-10 minutes\n"
                f"- Clay masks: Until slightly damp (not cracking)\n"
                f"- Hydrating masks: Can leave longer (30 mins max)"
            )

        elif "tingling" in user_input or "burning" in user_input:
            response = (
                f"⚠️ Tingling/Burning Sensation:\n"
                f"- Mild tingling may be normal with active ingredients\n"
                f"- Burning means STOP immediately\n"
                f"- Rinse with cool water if uncomfortable\n"
                f"- Do a patch test next time before full application"
            )

        elif "store" in user_input or "storage" in user_input:
            response = (
                f"🧊 Storage Guidelines:\n"
                f"- Fresh ingredients: Refrigerate (3 days max)\n"
                f"- Dry powders: Airtight container (6 months)\n"
                f"- Oils: Cool dark place (1 year)\n"
                f"- Discard if color/smell changes"
            )

        elif "side effects" in user_input or "reaction" in user_input:
            response = (
                f"🚨 Possible Reactions:\n"
                f"- Redness: Normal if mild (15-30 mins)\n"
                f"- Itching/Burning: Rinse immediately\n"
                f"- Breakouts: May occur with purging ingredients\n"
                f"- Dryness: Follow with moisturizer\n"
                f"- Swelling: Seek medical attention"
            )

        elif "before or after" in user_input or "routine" in user_input:
            response = (
                f"🧴 Skincare Routine Order:\n"
                f"1. Cleanse → 2. Tone → 3. This Mask\n"
                f"4. Serums → 5. Moisturizer → 6. SPF (AM)\n"
                f"- Wait 15 mins after mask before next steps\n"
                f"- Avoid strong actives (retinol) on same day"
            )

        elif "pregnant" in user_input or "breastfeeding" in user_input:
            response = (
                f"🤰 Pregnancy/Breastfeeding Safety:\n"
                f"- Avoid masks with: Retinol, salicylic acid\n"
                f"- Safe ingredients: Oatmeal, honey, yogurt\n"
                f"- Always consult your doctor first\n"
                f"- Do patch test with hormonal changes"
            )

        elif "age" in user_input or "old" in user_input:
            response = (
                f"👵 Age-Specific Advice:\n"
                f"- Teens: Focus on oil control\n"
                f"- 20s-30s: Prevention + hydration\n"
                f"- 40s+: More hydration, gentle exfoliation\n"
                f"- This mask is best for: {last_recipe['name'].split()[0]} skin"
            )

        elif "breakouts" in user_input or "purging" in user_input:
            response = (
                f"🤔 Breakouts vs Purging:\n"
                f"- Purging: Small bumps in problem areas (2-4 weeks)\n"
                f"- Breakouts: New inflamed pimples\n"
                f"- Stop if breakouts worsen after 3 uses\n"
                f"- Try honey instead if sensitive"
            )

        elif "sensitive" in user_input or "irritated" in user_input:
            response = (
                f"🌸 Sensitive Skin Tips:\n"
                f"- Reduce application time by half\n"
                f"- Mix with aloe vera to dilute\n"
                f"- Avoid physical scrubs\n"
                f"- Always patch test first!\n"
                f"- Try once weekly only"
            )

        elif "results" in user_input or "work" in user_input:
            response = (
                f"📈 Expected Results Timeline:\n"
                f"- Immediate: Soothing, hydration\n"
                f"- 1 week: Brighter complexion\n"
                f"- 2-4 weeks: Visible improvement\n"
                f"- Best results with consistent use\n"
                f"- Take before/after photos!"
            )

        elif "shower" in user_input or "bath" in user_input:
            response = (
                f"🚿 Shower Application Tips:\n"
                f"- Apply after cleansing in steamy shower\n"
                f"- Avoid direct water spray\n"
                f"- Rinse before exiting shower\n"
                f"- Follow with cool water splash\n"
                f"- Pat dry gently"
            )

        # Existing fallback
        else:
            response = (
                f"💡 About your {last_recipe['name']}:\n"
                f"- Best used in the {last_recipe.get('usage_time', 'evening')}\n"
                f"- Key benefits: {self._get_benefits(last_recipe['ingredients'])}\n\n"
                f"Ask me about:\n"
                f"- Application frequency\n"
                f"- Sun safety\n"
                f"- Sensitive skin adjustments\n"
                f"- Expected results timeline\n"
                f"- Storage instructions"
            )

        return {
            "response": response,
            "followup": followup,
            "recipe": recipe
        }




if __name__ == "__main__":
    try:
        generator = RecipeGenerator()
        print("Generator initialized successfully.")

        with open("data/skin_issues.json", "r", encoding="utf-8") as f:
            skin_data = json.load(f)
            test_conditions = [issue["name"] for issue in skin_data.get("skin_issues", [])]
        for condition in test_conditions[:2]:  # Limit to 2 for quick testing
            result = generator.generate_response(f"I have {condition}")
            print(f"Test - {condition} Recipe:")
            print(json.dumps(result, indent=2))

        result = generator.generate_response("I have rashes")
        followup1 = generator.handle_followup("Can I go in sunlight after this?")
        print("Test - Sunlight Follow-up for rashes:")
        print(json.dumps(followup1, indent=2))
        followup2 = generator.handle_followup("I don’t have oatmeal")
        print("Test - Missing Ingredient Follow-up for rashes:")
        print(json.dumps(followup2, indent=2))

    except Exception as e:
        print(f"Error during testing: {e}")