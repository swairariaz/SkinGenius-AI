"""
SKIN GENIUS - INGREDIENT RETRIEVER v1.1
=======================================
Purpose:
- Semantic search for skincare ingredients.
- Filters by skin type/condition compatibility.
- Optimizes for synergies + avoids conflicts.

Key Technologies:
- ChromaDB (vector database)
- Sentence Transformers (query embeddings)
- Conflict resolution rules (JSON-based)

Logic Flow:
1. Converts user query → ChromaDB search.
2. Applies filters:
   - Skin-type adjustments (e.g., avoid alcohol for dry skin)
   - Ingredient conflicts (e.g., Vitamin C + Niacinamide)
3. Boosts scores for synergistic pairs (e.g., honey + aloe).
4. Returns top 15 safe ingredients.

Data Sources:
- ingredients.csv → Embedded in ChromaDB.
- compatibility_rules.json → Avoids harmful combos.
"""

import chromadb
from typing import List, Dict, Optional
import json
from collections import defaultdict
import os


class IngredientRetriever:
    def __init__(self):
        """
        Initialize the ingredient retriever with ChromaDB connection and rules.
        Handles all ingredient search and compatibility logic.
        """
        try:
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(path="data/ingredients_embeddings.db")
            self.collection = self.client.get_collection("ingredients")

            # Load compatibility rules
            self.conflicts = []
            self.synergies = []
            self.skin_type_rules = []
            self._load_rules()

        except Exception as e:
            raise RuntimeError(f"Failed to initialize IngredientRetriever: {str(e)}")

    def _load_rules(self) -> None:
        """Load and validate compatibility rules from JSON file"""
        try:
            rules_path = os.path.join('data', 'compatibility_rules.json')
            if not os.path.exists(rules_path):
                raise FileNotFoundError(f"Rules file not found at {rules_path}")

            with open(rules_path, 'r', encoding='utf-8') as f:
                rules = json.load(f)
                self.conflicts = rules.get("conflicts", [])
                self.synergies = rules.get("synergies", [])
                self.skin_type_rules = rules.get("skin_type_adjustments", [])

        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format in compatibility rules")
        except Exception as e:
            raise RuntimeError(f"Error loading rules: {str(e)}")

    def _create_query(self, skin_type: str, conditions: List[str], properties: List[str]) -> str:
        """
        Create optimized search query from parameters

        Args:
            skin_type: User's skin type (dry, oily, etc.)
            conditions: List of skin conditions
            properties: List of desired ingredient properties

        Returns:
            Formatted query string for semantic search
        """
        conditions_str = " and ".join(conditions) if conditions else "skin concerns"
        properties_str = ", ".join(properties) if properties else "skin improvement"
        return f"{skin_type} skin with {conditions_str} needing {properties_str}"

    def retrieve_ingredients(self,
                             skin_type: str,
                             conditions: List[str],
                             properties: List[str]) -> Dict:
        """
        Main retrieval function with full pipeline:
        1. Semantic search
        2. Skin-type filtering
        3. Conflict resolution
        4. Synergy optimization

        Returns:
            {
                "recommended_ingredients": List[Dict],
                "avoid_combinations": List[str],
                "skin_type": str,
                "conditions": List[str]
            }
        """
        try:
            # 1. Semantic search
            query_text = self._create_query(skin_type, conditions, properties)
            raw_results = self.collection.query(
                query_texts=[query_text],
                n_results=25  # Get extra for filtering
            )

            # 2. Format results
            ingredients = self._format_results(raw_results)
            if not ingredients:
                raise ValueError("No ingredients found matching query")

            # 3. Apply skin type rules
            filtered = self._apply_skin_type_rules(ingredients, skin_type)

            # 4. Remove conflicts
            compatible = self._filter_conflicts(filtered)

            # 5. Boost synergies
            optimized = self._boost_synergies(compatible)

            return {
                "skin_type": skin_type,
                "conditions": conditions,
                "recommended_ingredients": optimized[:15],  # Return top 15
                "avoid_combinations": self._get_conflicts_to_avoid(optimized),
                "search_query": query_text
            }

        except Exception as e:
            raise RuntimeError(f"Ingredient retrieval failed: {str(e)}")

    def _format_results(self, chroma_results: Dict) -> List[Dict]:
        """Convert ChromaDB API results to clean ingredient dictionaries"""
        formatted = []
        for name, meta, score in zip(
                chroma_results["documents"][0],
                chroma_results["metadatas"][0],
                chroma_results["distances"][0]
        ):
            try:
                formatted.append({
                    "name": name,
                    "benefits": meta["benefits"],
                    "properties": [p.strip() for p in meta["properties"].split(",")],
                    "avoid_with": [
                        a.strip()
                        for a in meta["avoid_with"].split(",")
                        if meta["avoid_with"]
                    ],
                    "score": float(score),
                    "raw_score": float(score)  # Keep original before boosts
                })
            except (KeyError, AttributeError) as e:
                continue  # Skip malformed entries

        return formatted

    def _apply_skin_type_rules(self, ingredients: List[Dict], skin_type: str) -> List[Dict]:
        """Apply skin-type specific filters and boosts"""
        rules = next(
            (r for r in self.skin_type_rules
             if r["skin_type"].lower() == skin_type.lower()),
            None
        )

        if not rules:
            return ingredients

        # Remove ingredients to avoid
        filtered = [
            ing for ing in ingredients
            if not any(
                avoid.lower() in ing["name"].lower()
                for avoid in rules.get("avoid_combinations", [])
            )
        ]

        # Boost recommended ingredients
        for ing in filtered:
            if any(
                    rec.lower() in ing["name"].lower()
                    for rec in rules.get("recommended_pairs", [])
            ):
                ing["score"] *= 1.15  # 15% boost

        return filtered

    def _filter_conflicts(self, ingredients: List[Dict]) -> List[Dict]:
        """Remove ingredients that conflict with others in the list"""
        conflict_map = defaultdict(list)
        for rule in self.conflicts:
            conflict_map[rule["ingredient_A"].lower()].append(rule["ingredient_B"].lower())
            conflict_map[rule["ingredient_B"].lower()].append(rule["ingredient_A"].lower())

        filtered = []
        added_names = set()

        # Sort by score descending
        for ing in sorted(ingredients, key=lambda x: -x["score"]):
            name_lower = ing["name"].lower()

            # Check against already added ingredients
            if not any(
                    conflict in added_names
                    for conflict in conflict_map.get(name_lower, [])
            ):
                filtered.append(ing)
                added_names.add(name_lower)

        return filtered

    def _boost_synergies(self, ingredients: List[Dict]) -> List[Dict]:
        """Boost scores for synergistic ingredient pairs"""
        synergy_map = defaultdict(list)
        for rule in self.synergies:
            synergy_map[rule["ingredient_A"].lower()].append(rule["ingredient_B"].lower())
            synergy_map[rule["ingredient_B"].lower()].append(rule["ingredient_A"].lower())

        # Apply boosts
        for ing in ingredients:
            synergies = [
                s for s in synergy_map.get(ing["name"].lower(), [])
                if s in [i["name"].lower() for i in ingredients]
            ]
            if synergies:
                ing["synergies"] = synergies
                ing["score"] *= 1.25  # 25% boost for synergies

        return sorted(ingredients, key=lambda x: -x["score"])

    def _get_conflicts_to_avoid(self, ingredients: List[Dict]) -> List[str]:
        """Generate warnings for incompatible ingredient pairs"""
        conflicts = set()
        ingredient_names = {ing["name"].lower() for ing in ingredients}

        for ing in ingredients:
            for to_avoid in ing["avoid_with"]:
                if to_avoid.lower() in ingredient_names:
                    conflicts.add(f"{ing['name']} + {to_avoid}")

        return sorted(conflicts)