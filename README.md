# DLBDSEDA02_D
Dieses Repository dient als Codeabgabe für den Kurs DLBDSEDA02_D

## Wie dieses Repo zu nutzen ist: 
Die geforderte Datenanalyse ist in 3 Hauptphasen geteilt. 
Die Skripte bauen auf den jeweiligen Ergebnissdateien der Vorgängerphase auf.
Benötigte Abhängigkeiten sind in requirements.txt gelistet.

## Phase 1: 
**Skripte:** 
- Mastodon_Post_Scraper.py

### Mastodon_Post_Scraper.py
Das Skript läd, solange es läuft, in einem 30 Minuten Intervall aktuelle Mastodon Posts für die Hashtags „köln“ und „koeln“ herunter und nutzt dabei die Post-ID zur identifikation des letzten gescrapten Inhalts.
Die ID wird auch genutzt um Posts welche über beide Hashtags geladen wurden zu deduplizieren. 
Die Ergebnisse werden in die Datei „scraped_content.json“ abgelegt.
Das Skript kann mehrfach gestopt und gestartet werden, da es zu Beginn immer wieder die schon vorhandenen Posts einließt.

## Phase 2: 
**Skripte:**
- Data_Preprocessing.py
- Entity_Analysis_Accounts_Hashes.py

### Data_Preprocessing.py
Dieses Skript führt eine Vorverarbeitung der gescrapten Mastodon Posts durch. Als Input nimmt es die Datei „scraped_content.json“.
Neben HTML-Artefaktentfernung, Tokenisierung und Stopwortentfernung wurden zusätzlich Limitationen für eine minimale Anzahl vorhandener Tokens im Post, Posts welche nur aus Zahlenwerten bestehen sowie auf ausschließlich deutschsprachige Posts implementiert, da die Ergebnisse hier zunächst unzufriedenstellend waren. 
Die Ergebnisse liegen unter „preprocessed_posts.json“.
Erzeugt außerdem eine statistische Datei zur manuellen Überprüfung der Tokenqualität nach Preprocessing ("word_frequencies.json").

### Entity_Analysis_Accounts_Hashes.py
Nimmt als Input „scraped_content.json“. 
Führt eine statistische Auswertung der häufigsten Nutzeraccounts durch und zählt die Häufigkeit aller Hashtags ("Köln" und "Koeln" sind hier naturgemäß unter den Spitzenwerten).
Für die Zählung der Nutzeraccounts wird der eindeutige Identifier "acct" genutzt.
Schreibt die Ergebnisse in "accounts_hashtags_results.json". 
Plotted die Ergebnisse außerdem visuell in den Ordner "plots_accounts_hashtags".

## Phase 3: 
**Skripte:**
- Topic_Analyzer_LSA_TfIdf.py
- Topic_Analyzer_LDA_BoW.py

### Topic_Analyzer_LSA_TfIdf.py
Wertet die 5 häufigsten Themen über alle Posts in "preprocessed_posts.json" mittels TfIdf und LSA als Methodik aus.

Voreingestellte Parameter: 
- Gesuchte Themen: 5
- Begriffe pro Thema: 12
- minimale Begriffhäufigkeit in Korpus total: 3
- maximal Begriffhäufigkeit in Korpus in Prozent: 0.5

Schreibt die Ergebnisse in "lsa_results.json".
Plottet die Ergebnisse außerdem visuell in den Ordner ""plots_lsa".

### Topic_Analyzer_LDA_BoW.py
Wertet die 5 häufigsten Themen über alle Posts in "preprocessed_posts.json" mittels BoW und LDA als Methodik aus.
Variable "POOL_BY_DAY" kann gesetzt werden um Beiträge tageweise zusammenzufassen und längere Texte für bessere Themenerkennung zu generieren. Implementiert nach dem Vorbild von: https://doi.org/10.1145/2484028.2484166 .

Voreingestellte Parameter: 
- Gesuchte Themen: 5
- Begriffe pro Thema: 12
- minimale Begriffhäufigkeit in Korpus total: 3
- maximal Begriffhäufigkeit in Korpus in Prozent: 0.5

Schreibt die Ergebnisse in "lda_results.json".
Plottet die Ergebnisse außerdem visuell in den Ordner ""plots_lda".
