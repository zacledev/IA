# IA

Serveur Flask pour le chat RP d'Ikuyo Kita avec pipeline OpenRouter + mémoire LM Studio.

## Démarrage rapide

1. Copiez l'exemple d'environnement :

```bash
cp .env.example .env
```

2. Renseignez `OPENROUTER_API_KEY` dans `.env`.
   - Optionnel : ajustez `MODEL_BRAIN`, `MODEL_WRITER`, `*_TEMPERATURE`, `*_MAX_TOKENS` et `CONTEXT_TURNS` pour équilibrer qualité et vitesse.
3. Installez les dépendances (ex. `flask`, `flask-cors`, `python-dotenv`, `openai`).
4. Lancez le serveur :

```bash
python server.py
```

Le front est servi depuis `static/index.html`.
