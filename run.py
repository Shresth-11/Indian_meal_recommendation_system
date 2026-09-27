import json
import io
import os
import random

from flask import Flask, render_template, request, jsonify, Response
from typing import TypedDict, Literal

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from pypdf import PdfReader
from dotenv import load_dotenv

# ── Load environment variables from .env ─────────────────────────────────────
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Please set it in your .env file or cloud environment/secrets.")

# ── Flask app — template_folder points to project root where chat.html lives ─
app = Flask(__name__, template_folder=".")


# ─────────────────────────────────────────────────────────────────────────────
# 1. SHARED STATE
#    All agents read from and write to this single state dictionary.
# ─────────────────────────────────────────────────────────────────────────────
class MealPlanState(TypedDict):
    # User inputs (set once at start)
    age: str
    gender: str
    weight: str
    height: str
    health_condition: str
    allergies: str
    food_preference: str            # "vegetarian" | "vegan" | "Non-Vegetarian"
    medical_report_text: str        # raw text extracted from PDF, or fallback string

    # Agent outputs (populated sequentially)
    medical_analysis: str           # Agent 1 output
    nutrition_profile: str          # Agent 2 output
    meal_plan: str                  # Agent 3 output (may be revised)
    validation_result: str          # Agent 4 output: "APPROVED" or feedback
    revision_count: int             # tracks revision loops (max 2)


# ─────────────────────────────────────────────────────────────────────────────
# 2. LLM FACTORY
#    A robust wrapper class that handles random key rotation, rate-limiting (429),
#    and retry backoff automatically.
# ─────────────────────────────────────────────────────────────────────────────
# ── Default to high-speed openai/gpt-oss-20b (1,000 tok/sec) ─────────────────
# Can be overridden via GROQ_MODEL environment variable in Render
MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")


def extract_content(response) -> str:
    """Extract clean string content from LLM response safely, handling reasoning models and multi-part content."""
    if not response:
        return ""

    content = getattr(response, "content", "")
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(part.get("text", ""))
            else:
                parts.append(str(part))
        content = "".join(parts)
    elif not isinstance(content, str):
        content = str(content)

    content = content.strip()

    # Fallback to reasoning_content if content is empty (Groq reasoning models)
    if not content and hasattr(response, "additional_kwargs"):
        ak = response.additional_kwargs
        reasoning = ak.get("reasoning_content") or ak.get("reasoning") or ak.get("thought")
        if reasoning and isinstance(reasoning, str):
            content = reasoning.strip()

    return content


class RobustChatGroq:
    def __init__(self, temperature: float = 0.6, max_tokens: int = 4096):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model = MODEL_NAME

    def invoke(self, messages, **kwargs):
        import time
        import re

        attempts = 0
        max_attempts = 10
        last_exception = None

        while attempts < max_attempts:
            try:
                # Use reasoning_effort=low to prevent long thinking loops
                llm = ChatGroq(
                    api_key=GROQ_API_KEY,
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    model_kwargs={"reasoning_effort": "low"},
                )
                res = llm.invoke(messages, **kwargs)
                extracted = extract_content(res)
                if hasattr(res, "content") and not res.content and extracted:
                    res.content = extracted
                return res
            except Exception as e:
                last_exception = e
                error_str = str(e).lower()

                # If rate-limited (429), respect Groq's wait window
                if "rate" in error_str or "429" in error_str or "limit" in error_str:
                    attempts += 1
                    match = re.search(r"try again in ([\d\.]+)s", error_str)
                    if not match:
                        match = re.search(r"try again in ([\d\.]+) second", error_str)

                    if match:
                        wait_time = float(match.group(1)) + 1.5
                    else:
                        wait_time = 4.0

                    wait_time = min(wait_time, 15.0)
                    print(f"Rate limit on {self.model}. Waiting {wait_time:.1f}s before retry ({attempts}/{max_attempts})...")
                    time.sleep(wait_time)
                    continue

                # If model_kwargs was rejected, retry without it
                if "reasoning_effort" in error_str:
                    try:
                        llm_fallback = ChatGroq(
                            api_key=GROQ_API_KEY,
                            model=self.model,
                            temperature=self.temperature,
                            max_tokens=self.max_tokens,
                        )
                        res = llm_fallback.invoke(messages, **kwargs)
                        extracted = extract_content(res)
                        if hasattr(res, "content") and not res.content and extracted:
                            res.content = extracted
                        return res
                    except Exception as fallback_err:
                        last_exception = fallback_err

                attempts += 1
                time.sleep(2)

        raise last_exception

def get_llm() -> RobustChatGroq:
    return RobustChatGroq(temperature=0.6, max_tokens=4096)



# ─────────────────────────────────────────────────────────────────────────────
# 3. AGENT 1 — Medical Report Analyzer
#    Reads extracted PDF text + self-reported condition.
#    Outputs: dietary constraints mapped to Indian food context.
# ─────────────────────────────────────────────────────────────────────────────
def analyze_medical_report(state: MealPlanState) -> dict:
    llm = get_llm()

    system_prompt = """You are a medical dietitian AI specializing in Indian dietary patterns.
Analyze the patient's health information and any provided medical report text.
Extract key health markers and translate them into specific dietary constraints
using Indian food context (roti, dal, rice, sabzi, ghee, pickle, etc.).

Always output in EXACTLY this format:

MEDICAL ANALYSIS:
- [key health observations or conditions noted]

DIETARY CONSTRAINTS (Indian context):
- [specific food restrictions using Indian food names]
- [specific food recommendations using Indian food names]

SPECIAL NOTES:
- [any critical alerts or warnings for the meal planner]"""

    user_content = f"""Patient Profile:
- Health Condition: {state['health_condition']}
- Allergies: {state['allergies']}
- Diet Preference: {state['food_preference']}

Medical Report / Lab Results:
{state['medical_report_text']}

Analyze the above and provide precise dietary constraints using Indian food context."""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ])

    return {"medical_analysis": response.content}


# ─────────────────────────────────────────────────────────────────────────────
# 4. AGENT 2 — Nutritionist
#    Calculates BMR/TDEE and adjusts macros for health conditions.
#    Frames targets using Indian food groups.
# ─────────────────────────────────────────────────────────────────────────────
def calculate_nutrition(state: MealPlanState) -> dict:
    llm = get_llm()

    system_prompt = """You are an expert sports nutritionist and dietitian specializing in Indian dietary patterns.
Calculate precise daily nutritional requirements using the Mifflin-St Jeor BMR equation:
  - Men:   BMR = 10 × weight(kg) + 6.25 × height(cm) − 5 × age + 5
  - Women: BMR = 10 × weight(kg) + 6.25 × height(cm) − 5 × age − 161
  - TDEE  = BMR × 1.55 (moderate activity)

Adjust calorie and macro targets based on the medical analysis provided.
Frame ALL macronutrient sources using Indian food groups.

Always output in EXACTLY this format:

NUTRITION PROFILE:
- Daily Calories: [X] kcal  ([reason: deficit/surplus/maintenance])
- Protein: [X]g  (dal, paneer, curd, sprouts, chicken, fish, soy chunks)
- Carbohydrates: [X]g  (whole wheat roti, brown rice, jowar, bajra)
- Fats: [X]g  (ghee in moderation, mustard oil, coconut oil)
- Fiber: [X]g+  (seasonal vegetables, whole pulses, salads)
- Sodium: [limit if applicable]

MEAL DISTRIBUTION:
- Breakfast: [X] kcal
- Lunch: [X] kcal
- Snack: [X] kcal
- Dinner: [X] kcal
- Chai/Buttermilk buffer: [X] kcal

HEALTH-BASED ADJUSTMENTS:
- [specific macro/food adjustments based on medical analysis]"""

    user_content = f"""Patient Metrics:
- Age: {state['age']} years
- Gender: {state['gender']}
- Weight: {state['weight']} kg
- Height: {state['height']} cm
- Health Condition: {state['health_condition']}
- Diet Preference: {state['food_preference']}

Medical Analysis from Agent 1:
{state['medical_analysis']}

Calculate personalized daily nutrition targets with Indian food group context."""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ])

    return {"nutrition_profile": response.content}


# ─────────────────────────────────────────────────────────────────────────────
# 5. AGENT 3 — Meal Plan Generator
#    Creates a 7-day Indian-only meal plan.
#    On revision: reads validation_result and fixes specific issues.
# ─────────────────────────────────────────────────────────────────────────────
def generate_meal_plan_node(state: MealPlanState) -> dict:
    llm = get_llm()

    # Build revision context if this is a retry
    revision_context = ""
    if state["revision_count"] > 0 and state["validation_result"]:
        revision_context = f"""
⚠️ REVISION REQUIRED — Fix ALL the following issues from the previous plan:
{state['validation_result']}

Every issue listed above MUST be corrected. Do not repeat the same mistakes.
"""

    diet = state["food_preference"]
    allergies = state["allergies"]

    system_prompt = f"""You are an expert Indian cuisine chef and certified nutritionist.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRICT RULES — NO EXCEPTIONS WHATSOEVER:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. ONLY authentic, traditional Indian dishes. Zero exceptions.
2. BANNED — Western: pasta, burger, sandwich, pizza, cereal, oats with berries,
   smoothie bowl, avocado toast, scrambled eggs on toast, granola, salad with quinoa.
3. BANNED — Chinese: noodles, fried rice, spring rolls, stir-fry, momos (unless Indian style).
4. BANNED — Fusion dishes that mix Western/Chinese with Indian.
5. Diet preference is STRICTLY "{diet}":
   - vegetarian → no meat, no eggs, no fish
   - vegan → no meat, no eggs, no fish, no dairy (no paneer, no curd, no ghee — use coconut oil)
   - Non-Vegetarian → can include chicken, mutton, fish, eggs in Indian preparations
6. ALLERGIES → "{allergies}" — NEVER include these ingredients in ANY meal.
7. Include calorie count and macro breakdown (protein/carbs/fats) for EACH meal.
8. Vary regional cuisines across 7 days: North Indian, South Indian, Bengali, Gujarati, Rajasthani.

APPROVED BREAKFAST IDEAS (pick varied ones):
Poha, Upma, Idli-Sambhar, Medu Vada, Plain Paratha / Aloo Paratha / Methi Paratha,
Dosa with Chutney, Uttapam, Besan Chilla, Moong Dal Chilla, Sabudana Khichdi,
Daliya (broken wheat porridge), Thepla with Curd, Vermicelli Upma (Semiya Upma),
Pesarattu, Aloo Puri, Rava Idli, Pongal.

APPROVED LUNCH IDEAS:
Dal-Chawal, Rajma-Chawal, Chole-Roti, Kadhi-Chawal, Biryani (veg or non-veg),
Khichdi, Fish Curry with Rice, Chicken Curry with Roti, Mutton Rogan Josh with Roti,
Sambar-Rice, Rasam-Rice, Bisi Bele Bath, Dal Baati Churma, Pav Bhaji (occasionally).

APPROVED SNACK IDEAS:
Roasted Makhana, Chana Chaat, Sprout Chaat, Dhokla, Kachori, Samosa,
Roasted Chana, Fruit Chaat (Indian fruits), Chaas (Buttermilk), Lassi,
Murmura Chaat, Masala Peanuts (if no peanut allergy), Poha Chivda.

APPROVED DINNER IDEAS:
Dal-Roti (any dal variety), Palak Paneer with Roti, Aloo Gobi with Roti,
Baingan Bharta with Roti, Mixed Vegetable Curry with Rice, Rajma Masala,
Egg Curry with Roti (non-veg), Fish Curry with Rice (non-veg),
Chicken Tikka Masala with Roti (non-veg), Paneer Bhurji with Roti,
Matar Paneer with Roti, Moong Dal Tadka with Rice, Khichdi with Ghee and Pickle.

{revision_context}"""

    user_content = f"""Create a complete 7-day personalized Indian meal plan for:
- Age: {state['age']} years | Gender: {state['gender']}
- Weight: {state['weight']} kg | Height: {state['height']} cm
- Health Condition: {state['health_condition']}
- Allergies: {state['allergies']}
- Diet Preference: {state['food_preference']}

Nutrition Targets:
{state['nutrition_profile']}

Medical Dietary Constraints:
{state['medical_analysis']}

Output EXACTLY in this pipe-delimited format for every day. No extra text, no deviations:

### Day 1 | Monday
BREAKFAST | [Dish Name] (~[X] kcal) — Protein:[X]g, Carbs:[X]g, Fats:[X]g. [One-sentence description or tip.]
LUNCH | [Dish Name] (~[X] kcal) — Protein:[X]g, Carbs:[X]g, Fats:[X]g. [One-sentence description or tip.]
SNACKS | [Dish Name] (~[X] kcal) — Protein:[X]g, Carbs:[X]g, Fats:[X]g. [One-sentence description or tip.]
DINNER | [Dish Name] (~[X] kcal) — Protein:[X]g, Carbs:[X]g, Fats:[X]g. [One-sentence description or tip.]

### Day 2 | Tuesday
BREAKFAST | ...
LUNCH | ...
SNACKS | ...
DINNER | ...

Follow this exact pattern for Day 1 (Monday) through Day 7 (Sunday). Use pipe | as separator. No bullet points, no extra markdown."""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ])

    meal_plan_text = extract_content(response)
    return {"meal_plan": meal_plan_text}


# ─────────────────────────────────────────────────────────────────────────────
# 6. AGENT 4 — Validator
#    Checks: (1) allergy violations, (2) medical violations, (3) Indian-only.
#    Routes back to Generator if issues found (max 2 retries).
# ─────────────────────────────────────────────────────────────────────────────
def validate_meal_plan(state: MealPlanState) -> dict:
    llm = get_llm()

    system_prompt = """You are a strict meal plan auditor for an Indian nutrition system.
You perform THREE checks on every meal plan submitted to you.

CHECK 1 — ALLERGY COMPLIANCE:
  Scan every ingredient in every meal against the patient's allergy list.
  Flag any match with the exact day, meal, and ingredient.

CHECK 2 — MEDICAL COMPLIANCE:
  Cross-reference meals against the medical dietary constraints.
  Flag meals that clearly violate the stated restrictions
  (e.g., white rice biryani for a diabetic patient, deep-fried foods for heart disease).

CHECK 3 — INDIAN-ONLY ENFORCEMENT:
  Verify that every single dish is a recognizable, traditional Indian dish.
  Flag any Western/Chinese/fusion items (e.g., pasta, oats smoothie bowl,
  avocado toast, stir-fry noodles, granola, Caesar salad, pancakes).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESPONSE FORMAT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
If ALL three checks pass with no issues → respond with exactly one word:
APPROVED

If ANY issue is found → respond with:
REVISION NEEDED:
- [Day X, Meal: specific violation and how to fix it]
- [Day X, Meal: specific violation and how to fix it]
(list every issue clearly so the chef can fix them)

Do NOT approve a plan with even a single non-Indian item or allergy violation."""

    user_content = f"""AUDIT THIS MEAL PLAN:

Patient Allergies: {state['allergies']}
Health Condition: {state['health_condition']}
Diet Preference: {state['food_preference']}

Medical Constraints:
{state['medical_analysis']}

━━━━━━━━━━━━━━━━━━━━
MEAL PLAN:
━━━━━━━━━━━━━━━━━━━━
{state['meal_plan']}

Run all three checks and respond with APPROVED or REVISION NEEDED."""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ])

    validation_text = response.content.strip()
    new_revision_count = state["revision_count"]

    # Increment revision count only when sending back for a revision
    if not validation_text.startswith("APPROVED"):
        new_revision_count += 1

    return {
        "validation_result": validation_text,
        "revision_count": new_revision_count,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. CONDITIONAL EDGE — after Validator
#    Route back to Generator if revision needed and retries remain.
#    Route to END if approved or max retries (2) hit.
# ─────────────────────────────────────────────────────────────────────────────
def should_revise(state: MealPlanState) -> Literal["generate_meal_plan", "end"]:
    validation = state.get("validation_result", "")
    revision_count = state.get("revision_count", 0)

    if validation.startswith("APPROVED") or revision_count >= 2:
        return "end"
    return "generate_meal_plan"


# ─────────────────────────────────────────────────────────────────────────────
# 8. BUILD & COMPILE LANGGRAPH
# ─────────────────────────────────────────────────────────────────────────────
def build_graph() -> StateGraph:
    graph = StateGraph(MealPlanState)

    # Register nodes
    graph.add_node("analyze_medical_report", analyze_medical_report)
    graph.add_node("calculate_nutrition", calculate_nutrition)
    graph.add_node("generate_meal_plan", generate_meal_plan_node)
    graph.add_node("validate_meal_plan", validate_meal_plan)

    # Entry point
    graph.set_entry_point("analyze_medical_report")

    # Linear edges (sequential pipeline)
    graph.add_edge("analyze_medical_report", "calculate_nutrition")
    graph.add_edge("calculate_nutrition", "generate_meal_plan")
    graph.add_edge("generate_meal_plan", "validate_meal_plan")

    # Conditional edge: Validator → Generator (revision) or END (approved)
    graph.add_conditional_edges(
        "validate_meal_plan",
        should_revise,
        {
            "generate_meal_plan": "generate_meal_plan",
            "end": END,
        },
    )

    return graph.compile()


# Compile once at startup
meal_plan_graph = build_graph()
print("LangGraph compiled successfully. All 4 agents ready.")


# ─────────────────────────────────────────────────────────────────────────────
# 9. FLASK ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    """Serve the main UI."""
    return render_template("chat.html")


@app.route("/generate-meal-plan", methods=["POST"])
def generate_meal_plan():
    """
    Accepts multipart/form-data with user profile fields + optional PDF file.
    Streams SSE events as the LangGraph pipeline executes.

    SSE event types:
      { type: "progress",  step: "analyzing_medical" | "calculating_nutrition" |
                                 "generating_plan" | "validating_plan" }
      { type: "revision",  attempt: 1 | 2 }
      { type: "meal_plan", content: "<markdown>", has_warnings: bool }
      { type: "error",     message: "<error text>" }
      { type: "done" }
    """
    try:
        # ── Parse form fields ──────────────────────────────────────────────
        age             = request.form.get("age", "")
        gender          = request.form.get("gender", "")
        weight          = request.form.get("weight", "")
        height          = request.form.get("height", "")
        health_condition = request.form.get("health_condition", "None")
        allergies       = request.form.get("allergies", "None")
        food_preference = request.form.get("food_preference", "vegetarian")

        # ── Extract PDF text (optional) ───────────────────────────────────
        medical_report_text = (
            "No medical report uploaded. "
            "Base all recommendations on the self-reported health condition only."
        )

        if "medical_report" in request.files:
            pdf_file = request.files["medical_report"]
            if pdf_file and pdf_file.filename.lower().endswith(".pdf"):
                try:
                    pdf_bytes = pdf_file.read()
                    reader = PdfReader(io.BytesIO(pdf_bytes))
                    extracted = "\n".join(
                        page.extract_text() or "" for page in reader.pages
                    ).strip()

                    if extracted:
                        # Limit text to first 10,000 characters to prevent request size errors
                        medical_report_text = extracted[:10000]
                    else:
                        medical_report_text = (
                            "PDF uploaded but no extractable text found "
                            "(may be a scanned image). Using self-reported condition only."
                        )
                except Exception as pdf_err:
                    medical_report_text = (
                        f"Could not parse PDF ({pdf_err}). "
                        "Using self-reported condition only."
                    )

        # ── Build initial graph state ──────────────────────────────────────
        initial_state: MealPlanState = {
            "age": age,
            "gender": gender,
            "weight": weight,
            "height": height,
            "health_condition": health_condition,
            "allergies": allergies,
            "food_preference": food_preference,
            "medical_report_text": medical_report_text,
            "medical_analysis": "",
            "nutrition_profile": "",
            "meal_plan": "",
            "validation_result": "",
            "revision_count": 0,
        }

        # Map LangGraph node names → frontend step keys
        NODE_TO_STEP = {
            "analyze_medical_report": "analyzing_medical",
            "calculate_nutrition":    "calculating_nutrition",
            "generate_meal_plan":     "generating_plan",
            "validate_meal_plan":     "validating_plan",
        }

        # ── SSE generator ──────────────────────────────────────────────────
        def stream_events():
            latest_meal_plan   = ""
            latest_validation  = ""

            try:
                for chunk in meal_plan_graph.stream(
                    initial_state, stream_mode="updates"
                ):
                    for node_name, updates in chunk.items():

                        # 1. Emit progress tick for each node that completes
                        step_key = NODE_TO_STEP.get(node_name, node_name)
                        yield f"data: {json.dumps({'type': 'progress', 'step': step_key})}\n\n"

                        # 2. Track latest meal plan output
                        if "meal_plan" in updates:
                            plan_candidate = extract_content(updates["meal_plan"]) if not isinstance(updates["meal_plan"], str) else updates["meal_plan"].strip()
                            if plan_candidate:
                                latest_meal_plan = plan_candidate

                        # 3. Track validation and detect revision loops
                        if "validation_result" in updates:
                            latest_validation  = updates["validation_result"]
                            revision_count_now = updates.get("revision_count", 0)

                            if (
                                not latest_validation.startswith("APPROVED")
                                and revision_count_now < 2
                            ):
                                yield f"data: {json.dumps({'type': 'revision', 'attempt': revision_count_now})}\n\n"

                # ── Pipeline complete — send final meal plan ────────────────
                has_warnings = (
                    "APPROVED WITH WARNINGS" in latest_validation
                    or (
                        not latest_validation.startswith("APPROVED")
                        and bool(latest_meal_plan)
                    )
                )

                yield f"data: {json.dumps({'type': 'meal_plan', 'content': latest_meal_plan, 'has_warnings': has_warnings})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"

            except Exception as exc:
                yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

        return Response(
            stream_events(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",   # disable nginx buffering
                "Connection": "keep-alive",
            },
        )

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)