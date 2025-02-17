"""Generate test data for pipeline development"""
import pandas as pd
import random
import numpy as np

# Template data for generating realistic entries
COURSE_TEMPLATES = {
    'digital': {
        'titles': [
            "Digitale Medien im Unterricht",
            "Online-Lernplattformen effektiv nutzen",
            "Digitale Werkzeuge für den Unterricht",
            "Medienkompetenz fördern",
            "Digital gestützter Unterricht"
        ],
        'descriptions': [
            "In diesem Kurs lernen Sie den Einsatz digitaler Medien im Unterricht. Schwerpunkte: Präsentationstools, digitale Tafeln, Lern-Apps.",
            "Einführung in die Nutzung von Lernplattformen. Themen: Moodle, MS Teams, digitale Zusammenarbeit.",
            "Praktischer Einsatz digitaler Werkzeuge im Unterricht. Fokus auf: Tablets, Smartphones, interaktive Übungen.",
            "Workshop zur Förderung der Medienkompetenz. Inhalte: Internetrecherche, Datenschutz, soziale Medien.",
            "Grundlagen des digital gestützten Unterrichts. Module: Blended Learning, Flipped Classroom, Online-Assessment."
        ]
    },
    'traditional': {
        'titles': [
            "Klassenführung und Management",
            "Effektive Unterrichtsgestaltung",
            "Bewertung und Notengebung",
            "Elterngespräche führen",
            "Methodentraining für Lehrkräfte"
        ],
        'descriptions': [
            "Dieser Kurs vermittelt Grundlagen der Klassenführung. Themen: Regeln aufstellen, Störungen vermeiden, positives Lernklima.",
            "Methoden für einen strukturierten Unterricht. Schwerpunkte: Zeitmanagement, Unterrichtsplanung, Methodenvielfalt.",
            "Seminar zur objektiven Leistungsbewertung. Inhalte: Bewertungskriterien, Notenschlüssel, Feedback.",
            "Training für erfolgreiche Elterngespräche. Module: Gesprächsführung, Konfliktmanagement, Dokumentation.",
            "Methodische Grundlagen des Unterrichtens. Fokus: Gruppenarbeit, Einzelarbeit, Präsentationstechniken."
        ]
    }
}

def generate_test_entries(n_entries=10):
    """Generate sample training entries with realistic content"""
    entries = []
    
    for i in range(n_entries):
        # Randomly choose category and template
        category = random.choice(['digital', 'traditional'])
        title = random.choice(COURSE_TEMPLATES[category]['titles'])
        description = random.choice(COURSE_TEMPLATES[category]['descriptions'])
        
        # Add some variation
        if random.random() > 0.5:
            title += f" ({2024 - random.randint(0,2)})"
        
        entries.append({
            'title': title,
            'description': description
        })
    
    df = pd.DataFrame(entries)
    df.to_csv('teacher_training_data.csv', index=False)
    print(f"Generated {n_entries} test entries")
    print("\nSample entries:")
    print(df.head(3).to_string())
    
def generate_human_codes(n_entries=10):
    """Generate simulated human codes with bias towards digital content"""
    df = pd.read_csv('teacher_training_data.csv')
    
    # Simulate human coding based on content
    human_codes = []
    for _, row in df.iterrows():
        # More likely to code as 1 if contains digital keywords
        digital_keywords = ['digital', 'online', 'medien', 'plattform']
        is_digital = any(keyword in row['title'].lower() or 
                        keyword in row['description'].lower() 
                        for keyword in digital_keywords)
        
        # 80% chance of 1 if digital, 20% chance if not
        probability = 0.8 if is_digital else 0.2
        code = '1' if random.random() < probability else '0'
        human_codes.append(code)
    
    codes_df = pd.DataFrame({
        'title': df['title'],
        'human_code': human_codes
    })
    
    codes_df.to_csv('human_codes.csv', index=False)
    print(f"\nGenerated {len(codes_df)} human codes")
    print("\nSample codes:")
    print(codes_df.head(3).to_string())

if __name__ == "__main__":
    print("Generating test data...")
    generate_test_entries(20)  # Generate 20 entries
    generate_human_codes() 