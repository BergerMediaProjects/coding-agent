"""
Test data generator for the AI pipeline
Generates example course descriptions and their corresponding codes
"""
import pandas as pd
import yaml
import os
import random

def load_coding_scheme(file_path='coding_scheme.yml'):
    """Load the coding scheme from YAML file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            scheme = yaml.safe_load(file)
            print(f"\nLoaded {len(scheme)} categories from coding scheme")
            return scheme
    except Exception as e:
        print(f"Error loading coding scheme: {e}")
        return {}

def generate_test_entries():
    """Generate example course entries"""
    entries = [
        {
            'title': "Lerninhalte nutzerfreundlich aufbereiten - Kursgestaltung mit Moodle",
            'description': """04.05.2021
14:00 - 16:00
Diese Fortbildung wird online stattfinden.

Zielgruppe
Die Fortbildung richtet sich an Lehrende, Mitarbeitende, TutorInnen und die Lehre unterstützende Studierende der Hochschule München sowie anderer bayerischer Hochschulen.

Voraussetzungen
"Erste Schritte in Moodle" oder vergleichbare Erfahrungen mit Moodle

Lernziele
Inhalte
In dieser Fortbildung setzen Sie sich mit den aktuell in Moodle verfügbaren Kursformaten auseinander. Ein besonderer Fokus liegt hierbei auf den vielfältigen Konfigurationsmöglichkeiten des Grid Formats."""
        },
        {
            'title': "Lehrvideos leicht gemacht - Umsetzung und Einsatz bei Dokumentation, Reflexion und Feedback",
            'description': """02.12.2021
09:30 - 12:30
In dieser Fortbildung lernen Sie die praktische Umsetzung didaktischer Einsatzszenarien mittels Lehrvideos rund um die Themen Dokumentation, Reflexion und Feedback in der Lehre kennen.

Raum T0.012
T-Bau (Dachauer Straße 100a, EG links)

Zielgruppe
Die Fortbildung richtet sich an Lehrende der Hochschule München sowie anderer bayerischer Hochschulen."""
        }
    ]
    
    # Save entries to CSV
    df = pd.DataFrame(entries)
    df.to_csv('training_data.csv', index=False)
    
    print("\nGenerated test entries:")
    for i, entry in enumerate(entries, 1):
        print(f"\nEntry {i}:")
        print(f"Title: {entry['title']}")
        print(f"Description excerpt: {entry['description'][:100]}...")
    
    return df

def generate_codes(entries_df, coding_scheme):
    """Generate codes for each entry according to the coding scheme"""
    # Start with titles
    codes_df = pd.DataFrame({'title': entries_df['title']})
    
    # Generate codes for each category
    for category, details in coding_scheme.items():
        codes = []
        for _, entry in entries_df.iterrows():
            # Simple rule: 1 if digital-related, 0 otherwise
            is_digital = any(word in entry['title'].lower() 
                           for word in ['digital', 'online', 'moodle', 'video'])
            code = "1" if is_digital else "0"
            codes.append(code)
        
        codes_df[category] = codes
    
    # Save to Excel
    codes_df.to_excel('human_codes.xlsx', index=False)
    print(f"\nGenerated codes for {len(coding_scheme)} categories")
    return codes_df

def main():
    print("Generating test dataset...")
    
    # Load coding scheme
    scheme = load_coding_scheme()
    if not scheme:
        return
        
    # Generate test entries
    entries_df = generate_test_entries()
    
    # Generate codes
    codes_df = generate_codes(entries_df, scheme)

if __name__ == "__main__":
    main() 