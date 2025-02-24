# Liste de mots toxiques par catégorie
TOXIC_WORDS = {
    # Insultes et mots haineux
    'idiot', 'stupid', 'dumb', 'moron', 'retard', 'fool', 
    'loser', 'pathetic', 'worthless', 'useless',
    
    # Mots discriminatoires
    'nazi', 'racist', 'bigot', 'fascist', 'supremacist',
    
    # Violence
    'kill', 'die', 'murder', 'hate', 'destroy', 'attack',
    
    # Grossièretés
    'fuck', 'shit', 'damn', 'bitch', 'ass', 'crap',
    
    # Harcèlement
    'troll', 'spam', 'harass', 'stalker',
    
    # Toxicité politique
    'libtard', 'trumptard', 'snowflake', 'cuck',
    
    # Menaces
    'threat', 'revenge', 'hunt', 'track',
}

# Dictionnaire des catégories pour une utilisation plus fine si nécessaire
TOXIC_CATEGORIES = {
    'insults': {
        'idiot', 'stupid', 'dumb', 'moron', 'retard', 'fool',
        'loser', 'pathetic', 'worthless', 'useless'
    },
    'discrimination': {
        'nazi', 'racist', 'bigot', 'fascist', 'supremacist'
    },
    'violence': {
        'kill', 'die', 'murder', 'hate', 'destroy', 'attack'
    },
    'profanity': {
        'fuck', 'shit', 'damn', 'bitch', 'ass', 'crap'
    },
    'harassment': {
        'troll', 'spam', 'harass', 'stalker'
    },
    'political': {
        'libtard', 'trumptard', 'snowflake', 'cuck'
    },
    'threats': {
        'threat', 'revenge', 'hunt', 'track'
    }
} 