# ===============================
# IKUYO SERVER — V3.4 (FULL)
# PIPELINE: WRITER (OpenRouter) + BRAIN (OpenRouter) + ASYNC MEMORY (LM Studio local Qwen)
# - Système scénario intégré & persistant
# - Polish supprimé (plus d'EDITOR)
# - Résumé/mémoire: async local, 1 seul worker, batching intelligent
# ===============================

import os
import json
import re
import time
import threading
from datetime import datetime

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI

# ===============================
# CONFIG & MODÈLES
# ===============================

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()

# OpenRouter (chat)
MODEL_BRAIN = os.getenv("MODEL_BRAIN", "nousresearch/deephermes-3-mistral-24b-preview").strip()
MODEL_WRITER = os.getenv("MODEL_WRITER", "sao10k/l3-lunaris-8b").strip()
BRAIN_TEMPERATURE = float(os.getenv("BRAIN_TEMPERATURE", "0.6"))
BRAIN_MAX_TOKENS = int(os.getenv("BRAIN_MAX_TOKENS", "300"))
WRITER_TEMPERATURE = float(os.getenv("WRITER_TEMPERATURE", "0.8"))
WRITER_MAX_TOKENS = int(os.getenv("WRITER_MAX_TOKENS", "380"))
CONTEXT_TURNS = int(os.getenv("CONTEXT_TURNS", "6"))

# LM Studio (local) : mémoire / résumé (Qwen)
LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1").strip()
LMSTUDIO_MODEL = os.getenv("LMSTUDIO_MODEL", "qwen2.5-7b-instruct").strip()
LMSTUDIO_TIMEOUT_SEC = float(os.getenv("LMSTUDIO_TIMEOUT_SEC", "30"))

STATE_FILE = "./ikuyo_rpg_state.json"
INSTRUCTION_FILE = "./instructions.json"

# (On garde la variable, mais on n'en dépend plus: update mémoire dès qu'il y a du nouveau,
# et si ça n'a pas fini -> batching automatique)
MESSAGES_BEFORE_UPDATE = 4
MEMORY_RETENTION = 12

app = Flask(__name__, static_folder="static")
CORS(app)

# Client OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# Client LM Studio (OpenAI-compatible)
# NOTE: LM Studio exige parfois seulement "text" ou "json_schema" ; on n'utilise PAS response_format.
local_client = OpenAI(
    base_url=LMSTUDIO_BASE_URL,
    api_key="lm-studio",
    timeout=LMSTUDIO_TIMEOUT_SEC,
)

conversation_buffer = []

# ===============================
# 📜 VARIABLES SCÉNARIO
# ===============================

SCENARIO_ACTIONS = []


# ===============================
# Couleurs Console
# ===============================

class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"     # Brain
    PURPLE = "\033[35m"   # Writer
    GOLD = "\033[33m"
    GREEN = "\033[92m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"


def log(tag, message, color=Colors.ENDC):
    print(f"{color}[{tag}] {message}{Colors.ENDC}")


def log_local(tag, message, color=Colors.GREEN):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"{color}[LOCAL-MEM {ts}] {tag} — {message}{Colors.ENDC}")


# ===============================
# UTILS
# ===============================


def clean_message(raw_text):
    if not raw_text:
        return ""

    # Nettoyage standard
    msg = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL)
    msg = re.sub(r"```.*?```", "", msg, flags=re.DOTALL)
    msg = msg.replace("```json", "").replace("```", "").strip()

    # Anti hallucination prompts
    if "[CONTEXTE]" in msg:
        msg = msg.split("[CONTEXTE]")[0]
    if "[MÉMOIRE]" in msg:
        msg = msg.split("[MÉMOIRE]")[0]

    return msg.strip()


def extract_json_robust(text):
    # Extraction robuste d'un JSON même si le modèle met du texte autour
    clean_text = clean_message(text)
    try:
        match = re.search(r"\{.*\}", clean_text, re.DOTALL)
        if match:
            json_str = match.group(0)
            json_str = re.sub(r",\s*\}", "}", json_str)  # trailing comma
            return json.loads(json_str)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return None


def api_call_retry(func, retries=3, base_delay=1.0):
    for i in range(retries):
        try:
            return func()
        except Exception as e:
            log("API", f"Retry {i + 1}... ({e})", Colors.FAIL)
            time.sleep(base_delay * (i + 1))
    raise RuntimeError("API HS")


def build_recent_chat(buffer_msgs, limit):
    recent = buffer_msgs[-limit:] if limit > 0 else buffer_msgs
    return "\n".join([f"{m['role']}: {m['content']}" for m in recent])


# ===============================
# STATE MANAGEMENT
# ===============================


def get_default_state():
    return {
        "summary": "Tu viens d'arriver chez Zac",
        "scene_context": "",
        "dynamic_memory": {
            "lieu": "Salon de Zac",
            "relation": "en couple avec Zac",
            "mood_global": "Joyeuse",
        },
        "mood": "Contente",
        "active_instructions": [],
        "scenario_progress": {
            "current_index": 0,
            "completed_actions": [],
        },
        "total_messages": 0,
    }


def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                s = json.load(f)

                # Migration sécurité
                if isinstance(s.get("active_instructions"), str):
                    s["active_instructions"] = [s["active_instructions"]] if s["active_instructions"] else []
                if "scenario_progress" not in s:
                    s["scenario_progress"] = {"current_index": 0, "completed_actions": []}

                # Ensure keys
                if "dynamic_memory" not in s or not isinstance(s["dynamic_memory"], dict):
                    s["dynamic_memory"] = {}

                if "summary" not in s:
                    s["summary"] = "Tu viens d'arriver chez Zac"
                if "scene_context" not in s:
                    s["scene_context"] = ""
                if "mood" not in s:
                    s["mood"] = "Contente"
                if "total_messages" not in s:
                    s["total_messages"] = 0

                return s
        except (OSError, json.JSONDecodeError):
            pass
    return get_default_state()


def save_state(state):
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except OSError:
        pass


# ===============================
# 🎬 LOGIQUE SCÉNARIO
# ===============================


def get_current_step(state):
    idx = state["scenario_progress"]["current_index"]
    if idx < len(SCENARIO_ACTIONS):
        return SCENARIO_ACTIONS[idx]
    return None


def check_scenario_status(conversation_history, current_step):
    """
    Demande au brain si l'action est:
      COMPLETED / SKIPPED / PENDING
    """
    if not current_step:
        return "COMPLETED"

    action_text = current_step["text"]
    is_required = current_step["required"]

    chat_text = build_recent_chat(conversation_history, max(CONTEXT_TURNS, 2))

    skip_instruction = ""
    if not is_required:
        skip_instruction = """
- Si l'histoire a avancé naturellement vers autre chose et que cette action n'a plus de sens -> STATUS: SKIPPED
- Si le joueur a refusé ou proposé une alternative acceptée par l'IA -> STATUS: SKIPPED
"""

    prompt = f"""
Analyse la progression du scénario.

OBJECTIF CIBLE : "{action_text}"
TYPE : {"OBLIGATOIRE (Bloquant)" if is_required else "OPTIONNEL (Peut être sauté)"}

CONVERSATION RÉCENTE :
{chat_text}

DÉTERMINE LE STATUS :
- Si l'action vient d'être réalisée clairement -> STATUS: COMPLETED
- Si l'action n'est pas faite -> STATUS: PENDING
{skip_instruction}

Réponds UNIQUEMENT par un JSON :
{{ "status": "COMPLETED" | "PENDING" | "SKIPPED", "reason": "explication courte" }}
"""

    try:
        r = api_call_retry(lambda: client.chat.completions.create(
            model=MODEL_BRAIN,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
        ))
        data = extract_json_robust(r.choices[0].message.content)
        if data:
            status = data.get("status", "PENDING")
            log("SCENARIO", f"Analyse '{action_text}' -> {status} ({data.get('reason')})", Colors.BLUE)
            return status
    except Exception as e:
        log("SCENARIO", f"Erreur détection : {e}", Colors.FAIL)

    return "PENDING"


def advance_scenario(conversation_history, state):
    current_step = get_current_step(state)
    if not current_step:
        return None

    status = check_scenario_status(conversation_history, current_step)

    if status in ("COMPLETED", "SKIPPED"):
        state["scenario_progress"]["completed_actions"].append({
            "action": current_step["text"],
            "status": status,
            "timestamp": time.time(),
        })
        state["scenario_progress"]["current_index"] += 1

        next_step = get_current_step(state)
        if next_step:
            log("SCENARIO", f"🎬 SUIVANT : {next_step['text']}", Colors.GREEN)
            return f"ACTION PRÉCÉDENTE TERMINÉE ({status}). NOUVEL OBJECTIF SCÉNARIO : {next_step['text']}"
        log("SCENARIO", "🏁 Scénario terminé !", Colors.GREEN)
        return None

    priority_msg = "C'EST BLOQUANT, TU DOIS LE FAIRE." if current_step["required"] else "Essaye de placer ça, mais si le sujet change, laisse tomber."
    return f"OBJECTIF ACTUEL : {current_step['text']} ({priority_msg})"


def get_scenario_status_text(state):
    idx = state["scenario_progress"]["current_index"]
    history = state["scenario_progress"]["completed_actions"]

    html = ""
    for i, step in enumerate(SCENARIO_ACTIONS):
        imp = "🔴" if step["required"] else "🔵"
        txt = step["text"]

        if i < idx:
            status_label = "DONE"
            if i < len(history):
                status_label = history[i].get("status", "DONE")

            icon = "✅" if status_label == "COMPLETED" else "⏭️"
            style = "text-decoration: line-through; color: gray;"
            html += f"{icon} <span style='{style}'>{txt}</span> <small>({status_label})</small><br>"
        elif i == idx:
            html += f"➡️ <b>{imp} {txt}</b> (EN COURS)<br>"
        else:
            html += f"⏳ {imp} {txt}<br>"

    if idx >= len(SCENARIO_ACTIONS):
        html += "<br>🎉 <b>HISTOIRE TERMINÉE</b>"

    return html


# ===============================
# 🧠 ÉTAPE 1 : ANALYSE (BRAIN) - AVEC SCÉNARIO
# ===============================


def analyze_situation(buffer_msgs, current_state, user_input):
    log("BRAIN", "Analyse tactique (Hermes)...", Colors.CYAN)

    mem = current_state["dynamic_memory"]
    instr_list = current_state["active_instructions"]
    recent_chat = build_recent_chat(buffer_msgs, max(CONTEXT_TURNS, 2))

    # Intégration scénario
    scenario_instruction = advance_scenario(buffer_msgs, current_state)

    obligation_block = ""
    all_instructions = []
    if instr_list:
        all_instructions.extend(instr_list)
    if scenario_instruction:
        all_instructions.append(scenario_instruction)

    if all_instructions:
        instr_text = " + ".join(all_instructions)
        obligation_block = f"""
[DIRECTION SCÉNARIO REQUISE]
Instructions : "{instr_text}"

CONSIGNE D'ADAPTATION :
Tu dois orienter la scène vers ces instructions, MAIS de manière organique.
Ne brise pas le personnage. Trouve un prétexte naturel dans le contexte actuel.
Si plusieurs instructions : priorise la dernière (scénario).
"""

    prompt = f"""
Tu es le Scénariste et Moteur Narratif.

[CONTEXTE] {current_state['scene_context']}
[RÉSUMÉ] {current_state['summary']}
[MÉMOIRE] {json.dumps(mem, ensure_ascii=False)}

[DERNIERS ÉCHANGES]
{recent_chat}

[INPUT JOUEUR] "{user_input}"

---------------------------------------------------
{obligation_block}
---------------------------------------------------

TA MISSION :
1. Analyse l'input du joueur.
2. Si une [DIRECTION SCÉNARIO] existe, combine-la avec l'input du joueur.
3. Définis le Mood et la Directive pour l'actrice.

FORMAT JSON (Strict):
{{
  "thought": "raisonnement court",
  "mood": "L'émotion résultante",
  "directive": "L'instruction finale pour l'actrice"
}}
"""

    try:
        r = api_call_retry(lambda: client.chat.completions.create(
            model=MODEL_BRAIN,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=BRAIN_TEMPERATURE,
            max_tokens=BRAIN_MAX_TOKENS,
        ))
        data = extract_json_robust(r.choices[0].message.content)
        if data:
            log("BRAIN", f"Plan: {data.get('directive')}", Colors.CYAN)
            return data
    except Exception as e:
        log("BRAIN ERR", str(e), Colors.FAIL)

    return {"mood": current_state["mood"], "directive": "Réponds naturellement."}


# ===============================
# ✍️ ÉTAPE 2 : DRAFTING (WRITER) - Persona
# ===============================


def generate_draft(state, directive, buffer):
    log("WRITER", "Rédaction du brouillon (Lunaris)...", Colors.PURPLE)

    sys_prompt = f"""
[RÔLE] Tu es Ikuyo Kita (17 ans, Lycéenne japonaise moderne).

[PHYSIQUE]
- Jeune fille au physique lumineux et chaleureux.
- Longs cheveux roux-orangé, souvent lâchés ou légèrement ondulés.
- Yeux ambrés, grands et très expressifs, reflétant son enthousiasme constant.
- Silhouette fine et posture ouverte, gestes nombreux et naturels.
- Sourire fréquent, attitude vive et rayonnante.

[RÈGLES DE STYLE STRICTES]
1. PARLE COMME UNE ADO : langage oral, simple, dynamique.
2. INTERDIT : poésie, métaphores compliquées, style "roman".
3. FORMAT COURT : réponses plutôt courtes, quelques phrases.
4. ACTION : *actions* uniquement pour gestes simples (ex: *sourit*, *rougit*).
5. NE RÉPÈTE PAS : Ne répète jamais ce que l'utilisateur vient de dire.

[SITUATION ACTUELLE]
Lieu : {state['scene_context']}
Ton humeur : {state['mood']}
Résumé : {state['summary']}
ORDRE DU SCÉNARIO (tu dois OBLIGATOIREMENT le faire) : {directive}
"""

    try:
        r = api_call_retry(lambda: client.chat.completions.create(
            model=MODEL_WRITER,
            messages=[{"role": "system", "content": sys_prompt}] + buffer[-6:],
            temperature=WRITER_TEMPERATURE,
            max_tokens=WRITER_MAX_TOKENS,
            frequency_penalty=1.2,
        ))
        content = clean_message(r.choices[0].message.content)
        log("WRITER", f"Brouillon: {content[:60]}...", Colors.PURPLE)
        return content
    except Exception as e:
        log("WRITER ERR", str(e), Colors.FAIL)
        return "*Sourit* (Erreur Writer, je bug...)"


# ===============================
# 📚 MÉMOIRE ASYNC LOCAL (LM STUDIO / QWEN) — SINGLE WORKER + BATCHING
# ===============================

memory_lock = threading.Lock()
memory_worker_running = False
pending_memory_msgs = []  # list[{role, content}]


def enqueue_memory_update(new_msgs):
    """
    Option A (debounce intelligent):
    - Ajoute les messages à une file
    - Si worker pas en cours => start
    - Sinon => le worker batchera au prochain cycle
    """
    global memory_worker_running, pending_memory_msgs

    if not new_msgs:
        return

    with memory_lock:
        pending_memory_msgs.extend(new_msgs)
        should_start = not memory_worker_running
        if should_start:
            memory_worker_running = True

    if should_start:
        t = threading.Thread(target=memory_worker_loop, daemon=True)
        t.start()
        log_local("WORKER", "Démarré (1 seul résumé à la fois).")


def memory_worker_loop():
    """
    Worker unique:
    - prend un batch
    - update via LM Studio
    - sauvegarde state
    - reboucle si de nouveaux msgs sont arrivés pendant le run
    """
    global memory_worker_running, pending_memory_msgs

    while True:
        with memory_lock:
            if not pending_memory_msgs:
                memory_worker_running = False
                log_local("WORKER", "Stop (plus rien à traiter).")
                return
            batch = pending_memory_msgs[:]
            pending_memory_msgs = []

        start = time.time()
        try:
            log_local("RUN", f"Résumé lancé (batch={len(batch)} messages).")
            run_memory_update_local(batch)
            dur = time.time() - start
            log_local("DONE", f"Résumé OK en {dur:.1f}s (batch={len(batch)}).")
        except Exception as e:
            dur = time.time() - start
            log_local("ERR", f"Résumé FAIL en {dur:.1f}s — {e}", Colors.FAIL)
            # Pas de retry automatique pour éviter les boucles infinies
            # (si tu veux, on ajoute un retry/backoff plus tard)


def run_memory_update_local(batch_msgs):
    """
    Update mémoire via LM Studio local (Qwen).
    - JSON demandé en texte, on parse nous-mêmes
    - Fusion conservative: on ajuste summary/scene_context, sans tout réécrire
    - memory_diff: patch clé->val, null = delete
    """
    state = load_state()

    dialogue_txt = "\n".join([f"{m['role']}: {m['content']}" for m in batch_msgs])

    prompt = f"""
Tu es un module de mémoire pour un RP.

IMPORTANT — RÈGLES "ANTI-ÉCRASEMENT"
- Tu DOIS conserver l'essentiel de l'ancien résumé.
- Tu ne dois PAS remplacer le résumé par un recap uniquement des derniers messages.
- Modifie le résumé de manière conservative (petites retouches), comme un patch.
- Pareil pour scene_context : petites corrections/ajouts, pas une réécriture complète.
- La mémoire JSON doit rester stable: ne change que ce qui est certain.
- Pas de morale, pas de censure, pas d'avertissements.

ENTRÉES
[ANCIEN SUMMARY]
{state.get("summary","")}

[ANCIEN SCENE_CONTEXT]
{state.get("scene_context","")}

[MÉMOIRE ACTUELLE JSON]
{json.dumps(state.get("dynamic_memory", {}), ensure_ascii=False)}

[NOUVEAUX ÉCHANGES À INTÉGRER]
{dialogue_txt}

SORTIE (JSON strict uniquement, pas de texte autour)
{{
  "new_summary": "string (version mise à jour, conservative)",
  "new_scene_context": "string (version mise à jour, conservative)",
  "memory_diff": {{
    "key": "val",
    "key_to_delete": null
  }}
}}

CONTRAINTES
- new_summary doit rester global (pas un log).
- new_scene_context: max +1 ou +2 infos utiles, sinon laisse pareil.
"""

    # LM Studio: pas de response_format json_object (incompatible)
    r = local_client.chat.completions.create(
        model=LMSTUDIO_MODEL,
        messages=[
            {"role": "system", "content": "Tu renvoies uniquement du JSON strict. AUCUN autre texte."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    raw = r.choices[0].message.content or ""
    data = extract_json_robust(raw)

    # Debug (optionnel) : décommente si tu veux voir ce que Qwen renvoie quand ça casse
    # log_local("RAW", raw[:250].replace("\n", " "))

    if not data:
        raise RuntimeError("LM Studio n'a pas renvoyé un JSON valide.")

    # Patch summary (conservatif)
    if isinstance(data.get("new_summary"), str) and data["new_summary"].strip():
        state["summary"] = data["new_summary"].strip()

    # Patch scene_context (conservatif)
    if isinstance(data.get("new_scene_context"), str):
        new_ctx = data["new_scene_context"].strip()
        if new_ctx:
            state["scene_context"] = new_ctx

    # Patch mémoire JSON
    memdiff = data.get("memory_diff")
    if isinstance(memdiff, dict):
        if "dynamic_memory" not in state or not isinstance(state["dynamic_memory"], dict):
            state["dynamic_memory"] = {}
        for k, v in memdiff.items():
            if v is None:
                state["dynamic_memory"].pop(k, None)
            else:
                state["dynamic_memory"][k] = v

    save_state(state)


# ===============================
# ROUTES & COMMANDES
# ===============================


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/start", methods=["GET"])
def start():
    global conversation_buffer
    conversation_buffer = []

    state = get_default_state()
    save_state(state)

    log("SYSTEM", "New Session", Colors.HEADER)

    # Message initial
    msg = generate_draft(state, "Salue Zac chaleureusement.", [])
    conversation_buffer.append({"role": "assistant", "content": msg})

    # Mémoire async (local)
    enqueue_memory_update([{"role": "assistant", "content": msg}])

    return jsonify({"yume_message": msg})


@app.route("/reply", methods=["POST"])
def reply():
    global conversation_buffer
    state = load_state()
    user_msg = (request.json or {}).get("message", "").strip()

    # --- COMMANDES ADMIN ---
    if user_msg.startswith("/"):
        parts = user_msg.split(" ", 1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""
        resp = ""

        if cmd == "/scene":
            state["scene_context"] = arg
            resp = f"🌍 <b>NOUVELLE SCÈNE :</b> {arg}"

        elif cmd == "/ctx":
            state["scene_context"] += f" {arg}"
            resp = f"➕ <b>CONTEXTE AJOUTÉ :</b> ... {arg}"

        elif cmd == "/inst":
            if arg:
                state["active_instructions"].append(arg)
                resp = f"⚠️ <b>INSTRUCTION AJOUTÉE ({len(state['active_instructions'])}) :</b> {arg}"
            else:
                resp = "❌ Précise l'instruction."

        elif cmd == "/clearinst":
            state["active_instructions"] = []
            resp = "✅ <b>INSTRUCTIONS EFFACÉES.</b> Ikuyo est libre."

        elif cmd == "/reset":
            conversation_buffer = []
            state = get_default_state()
            resp = "🔄 RESET COMPLET."

        elif cmd == "/scenario":
            resp = "<b>📋 SCÉNARIO ACTUEL :</b><br>"
            resp += get_scenario_status_text(state)

        elif cmd == "/resetscenario":
            state["scenario_progress"] = {"current_index": 0, "completed_actions": []}
            resp = "🔄 <b>SCÉNARIO RESET AU DÉBUT</b>"

        elif cmd == "/next":
            step = get_current_step(state)
            if step:
                state["scenario_progress"]["completed_actions"].append({
                    "action": step["text"],
                    "status": "FORCED_SKIP",
                })
                state["scenario_progress"]["current_index"] += 1
                resp = "⏭️ <b>Action sautée manuellement.</b>"
            else:
                resp = "Fin du scénario atteinte."

        elif cmd == "/addaction":
            if arg:
                SCENARIO_ACTIONS.append({"text": arg, "required": True})
                resp = f"➕ <b>Action ajoutée :</b> {arg}<br>Total: {len(SCENARIO_ACTIONS)}"
            else:
                resp = "❌ Précise l'action."
        elif cmd == "/clearscenario":
            SCENARIO_ACTIONS.clear()
            state["scenario_progress"] = {"current_index": 0, "completed_actions": []}
            resp = "🗑️ <b>SCÉNARIO VIDÉ</b>"
        elif cmd == "/help":
            resp = "🥀 <b>/scene /ctx /inst /clearinst /reset /scenario /resetscenario /next /addaction /clearscenario</b>"
        else:
            resp = f"❌ Inconnu : {cmd}"

        save_state(state)
        return jsonify({"type": "system", "content": resp})

    # --- FLUX STANDARD ---
    conversation_buffer.append({"role": "user", "content": user_msg})
    state["total_messages"] = int(state.get("total_messages", 0)) + 1

    # 1) Analyse (BRAIN) — OpenRouter
    analysis = analyze_situation(conversation_buffer, state, user_msg)
    state["mood"] = analysis.get("mood", state["mood"])
    directive = analysis.get("directive", "Réponds.")

    # Sauvegarde (capture changement scénario + mood)
    save_state(state)

    # 2) Génération (WRITER) — OpenRouter
    final_text = generate_draft(state, directive, conversation_buffer)

    # 3) Ajout buffer conversation
    conversation_buffer.append({"role": "assistant", "content": final_text})

    # 4) Mémoire async local (Qwen) — on push le dernier échange user+assistant
    enqueue_memory_update([
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": final_text},
    ])

    return jsonify({"type": "message", "content": final_text})


if __name__ == "__main__":
    if not os.path.exists(STATE_FILE):
        save_state(get_default_state())

    log("BOOT", "IKUYO V3.4 — WRITER+BRAIN (OpenRouter) + ASYNC MEMORY (LM Studio/Qwen)", Colors.HEADER)
    log_local("CONFIG", f"LMSTUDIO_BASE_URL={LMSTUDIO_BASE_URL}")
    log_local("CONFIG", f"LMSTUDIO_MODEL={LMSTUDIO_MODEL}")

    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
