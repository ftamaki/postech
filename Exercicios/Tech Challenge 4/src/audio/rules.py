# -*- coding: utf-8 -*-

# Regras baseline (rápidas e explicáveis) para sinais de risco em contexto de saúde.
# Você pode calibrar depois, mas isso já cumpre o requisito de "detecção de anomalias/sinais".

RISK_RULES_PTBR = [
    {
        "id": "anxiety_signal",
        "label": "Sinal de ansiedade",
        "severity": "medium",
        "keywords_any": [
            "ansioso", "ansiosa", "ansiedade", "nervoso", "nervosa", "apreensiva", "apreensivo",
            "preocupada", "preocupado", "pânico", "crise", "taquicardia", "falta de ar", "não consigo respirar",
            "medo", "muito medo", "assustada", "assustado",
        ],
    },
    {
        "id": "depression_signal",
        "label": "Sinal de depressão / humor deprimido",
        "severity": "high",
        "keywords_any": [
            "triste", "muito triste", "sem esperança", "desanimada", "desanimado", "não tenho energia",
            "não vejo sentido", "não aguento", "vontade de sumir", "choro", "chorando", "culpa", "culpada",
            "não consigo", "exausta", "cansada demais",
        ],
    },
    {
        "id": "postpartum_signal",
        "label": "Sinal de risco pós-parto (triagem textual)",
        "severity": "high",
        "keywords_any": [
            "pós-parto", "depois que o bebê nasceu", "depois do parto", "não consigo cuidar", "não consigo amamentar",
            "não consigo dormir", "insônia", "sem dormir", "me sinto incapaz", "não tenho vínculo",
        ],
    },
    {
        "id": "domestic_violence_signal",
        "label": "Sinal de possível violência doméstica",
        "severity": "high",
        "keywords_any": [
            "me bateu", "me agrediu", "apanhei", "agressão", "me ameaçou", "ameaça", "tenho medo dele",
            "violência", "ele me controla", "não deixa", "ele gritou", "ele me xingou",
        ],
    },
]

# Palavras que elevam severidade quando aparecem perto de sinais (heurística simples)
BOOST_TERMS = [
    "muito", "demais", "intenso", "insuportável", "todos os dias", "sempre", "nunca"
]
