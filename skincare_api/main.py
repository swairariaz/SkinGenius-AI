"""
SKIN GENIUS - FASTAPI BACKEND v1.0
==================================
Purpose:
- REST API for AI analysis + follow-ups.
- Manages session memory.

Endpoints:
1. POST /analyze:
   - Input: Text + optional image.
   - Output: Diagnosis + recipe.
2. POST /followup:
   - Input: Question + session_id.
   - Output: Adapted recipe/advice.

Key Features:
- Session persistence (JSON files).
- Temporary image cleanup.
- Error fallbacks (default recipe).
- CORS-enabled for frontend.

Tech Stack:
- FastAPI + Uvicorn
- Python logging (skingenius.log)
"""

from fastapi import FastAPI, Form, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from AI.core import SkinGeniusCore
import os
import uuid
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    filename='skingenius.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = FastAPI(title="SkinCare AI API")

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AI Core
try:
    core = SkinGeniusCore()
    logging.info("Skincare AI initialized successfully!")
except Exception as e:
    logging.error(f"Failed to initialize SkinGeniusCore: {str(e)}")
    raise RuntimeError(f"Initialization failed: {str(e)}")

# Memory System
MEMORY_DIR = os.path.join(os.path.dirname(__file__), "..", "memory")
os.makedirs(MEMORY_DIR, exist_ok=True)

def get_memory_path(session_id: str):
    """Generate path for session memory file."""
    return os.path.join(MEMORY_DIR, f"{session_id}.json")

# Custom dependency to handle image parameter flexibly
async def process_image(request: Request) -> Optional[str]:
    form = await request.form()
    image = form.get("image")
    image_path = None
    if image and hasattr(image, "filename") and image.filename:
        image_path = f"temp_{uuid.uuid4()}.jpg"
        try:
            with open(image_path, "wb") as buffer:
                buffer.write(await image.read())
            return image_path
        except Exception as e:
            logging.error(f"Failed to process image: {str(e)}")
            return None
    logging.warning(f"Invalid or no image provided: {image}")
    return None

@app.post("/analyze")
async def analyze_skin(
    text: str = Form(...),
    session_id: str = Form(None),
    image_path: Optional[str] = Depends(process_image)  # Use custom image processing
):
    """
    Analyze skin condition based on text and optional image.
    - text: User description of skin concern.
    - session_id: Unique session identifier (generated if None).
    - image: Optional uploaded image for medical analysis.
    """
    try:
        # Generate session ID if not provided
        session_id = session_id or str(uuid.uuid4())
        memory_path = get_memory_path(session_id)
        logging.info(f"Processing request for session_id: {session_id}")

        # Call AI Core
        analysis = core.analyze_skin(
            user_query=text,
            image_path=image_path
        )
        logging.info(f"Analysis completed for query: {text}")

        # Ensure recipes are included (fallback if missing)
        if 'recipes' not in analysis or not analysis['recipes'].get('response'):
            analysis['recipes'] = {
                "response": f"Default Mask: Mix basic ingredients. Apply for 10 minutes. Rinse with water. Pro tip: Avoid if skin is irritated. Apply during Evening!",
                "followup": "What’s next? Any missing ingredients or questions?",
                "name": "Default Mask",
                "steps": ["Mix basic ingredients", "Apply for 10 minutes", "Rinse with water"],
                "safety_warning": "Avoid if skin is irritated.",
                "usage_time": "Evening",
                "ingredients": ["basic ingredients"],
                "safety_override": "Ultra-gentle formula recommended"
            }
            logging.warning(f"No recipe generated for query '{text}'. Using default recipe.")
        # Update memory
        memory = {
            "session_id": session_id,
            "conversation": [{
                "user_input": text,
                "system_response": analysis,
                "timestamp": datetime.now().isoformat()
            }],
            "current_state": analysis
        }
        with open(memory_path, "w", encoding="utf-8") as f:
            json.dump(memory, f, ensure_ascii=False, indent=2)
        logging.info(f"Memory updated at {memory_path}")

        return {
            "status": "success",
            "session_id": session_id,
            "analysis": analysis
        }

    except Exception as e:
        logging.error(f"Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

    finally:
        # Clean up temporary image file
        if image_path and os.path.exists(image_path):
            try:
                os.remove(image_path)
                logging.info(f"Cleaned up temporary image: {image_path}")
            except Exception as e:
                logging.error(f"Failed to clean up image {image_path}: {str(e)}")

@app.post("/followup")
async def handle_followup(
    question: str = Form(...),
    session_id: str = Form(...)
):
    """
    Handle follow-up questions based on session context.
    - question: User’s follow-up query.
    - session_id: Session identifier from previous analysis.
    """
    try:
        memory_path = get_memory_path(session_id)
        if not os.path.exists(memory_path):
            logging.error(f"Session not found for session_id: {session_id}")
            raise HTTPException(status_code=404, detail="Session not found")

        with open(memory_path, "r", encoding="utf-8") as f:
            memory = json.load(f)

        # Use RecipeGenerator's handle_followup
        generator_response = core.generator.handle_followup(question)
        response = generator_response.get("response", "No response generated.")
        recipe = generator_response.get("recipe", None)
        followup = generator_response.get("followup", "Anything else you need?")

        # Additional intent detection
        answer = response
        alternatives = []
        if "how long" in question.lower():
            if recipe and recipe["steps"]:
                apply_step = next((s for s in recipe["steps"] if s.startswith("Apply:")), "Apply: Spread for 10-15 mins")
                answer = f"Apply the mask for {apply_step.split('for')[-1].strip()}. Always patch test first!"
                alternatives = ["Follow the recipe steps", "Check with a dermatologist for extended use"]
        elif "substitute" in question.lower() or "replace" in question.lower():
            if recipe:
                answer = f"You can substitute ingredients in {recipe['name']} with similar ones like {', '.join(core.generator.common_swaps.get(recipe['ingredients'][0], ['rose water']))}."
                alternatives = core.generator.common_swaps.get(recipe["ingredients"][0], ["rose water"])

        # Update memory
        memory["conversation"].append({
            "user_input": question,
            "system_response": {
                "answer": answer,
                "recipe": recipe,
                "followup": followup,
                "alternatives": alternatives
            },
            "timestamp": datetime.now().isoformat()
        })
        memory["current_state"] = {"recipes": recipe} if recipe else memory["current_state"]
        with open(memory_path, "w", encoding="utf-8") as f:
            json.dump(memory, f, ensure_ascii=False, indent=2)
        logging.info(f"Follow-up memory updated at {memory_path}")

        return {
            "status": "success",
            "response": {
                "answer": answer,
                "recipe": recipe,
                "followup": followup,
                "alternatives": alternatives
            }
        }

    except json.JSONDecodeError as e:
        logging.error(f"Invalid memory file for session_id {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Invalid session data")
    except Exception as e:
        logging.error(f"Follow-up failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("skincare_api.main:app", host="0.0.0.0", port=8000, reload=True)