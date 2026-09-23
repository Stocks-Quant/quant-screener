# Options-Screener

Drei Agenten, eine Shortlist. Der Screener sammelt täglich Kurse, Optionsvolumen und Open Interest für den S&P 100, sucht Stellen, an denen Optionsaktivität und Kursbewegung unterschiedliche Geschichten erzählen, und gibt höchstens fünf Ticker zur Recherche aus.

Er ist ein Screener, kein Signal. Ungewöhnliche Aktivität ist ein Grund, sich ein Unternehmen anzusehen. Sie ist kein Grund, eine Position einzugehen.

| Agent | Umsetzung | Aufgabe |
|---|---|---|
| 1. Daten | `screener/fetch_data.py`, GitHub Actions | sammelt, datiert, belegt. Fehlt etwas, steht dort `UNAVAILABLE` |
| 2. Analyse | `screener/analyze.py`, GitHub Actions | beschreibt Aktivität relativ zur eigenen Historie, nie bullisch oder bärisch |
| 3. Flagging | Claude, geplanter Task | sucht die langweilige Erklärung, wählt höchstens 5, schreibt ins Dashboard |
| Auswertung | `screener/evaluate_flags.py` | misst jede Flag nach 10 Handelstagen gegen die Basisrate |

## Tagesablauf

| Zeit (UTC) | Berlin Sommerzeit | Lauf | Was passiert |
|---|---|---|---|
| 21:37 Mo bis Fr | 23:37 | Abend | Tagesbalken, Aktienvolumen, Optionsvolumen des Handelstags, Termine |
| 11:41 Mo bis Fr | 13:41 | Morgen | Open Interest nach dem Handelstag, Analyse, Auswertung |
| 13:07 und 15:33 | 15:07, 17:33 | Wiederholung | nur falls die Quelle das Open Interest noch nicht aktualisiert hatte |
| ca. 16:15 | 18:15 | Claude | Flagging-Agent liest die Analyse und füllt das Dashboard |

Warum zwei Läufe: Optionsvolumen steht nach Handelsschluss fest, das Open Interest erst am nächsten Morgen. Nur die Veränderung des Open Interest zeigt, ob Positionen eröffnet oder geschlossen wurden. Ohne sie ist eine Flag kein Beleg für irgendetwas, deshalb gibt es ohne sie keine Kandidaten.

## Einrichtung

Das Repo ist öffentlich: Actions-Minuten sind dann kostenlos, und der tägliche Claude-Task liest die Daten ohne Zugangsdaten. Im Repo liegen nur Code und öffentliche Marktdaten.

* Actions, Options-Screener, Run workflow, Modus `smoke`: führt die Tests aus und fragt drei Ticker live ab. Grün heißt, die Datenquelle funktioniert von GitHub aus.
* Die Zeitpläne starten von selbst. Die ersten Kandidaten erscheinen, sobald 15 Handelstage Baseline gesammelt sind, also nach gut drei Wochen. Vorher zeigt der Bericht Werte, aber keine Flags.
* Der Flagging-Agent ist ein geplanter Claude-Task mit dem Prompt aus `prompts/flagging_agent.md`.

## Was die Zahlen bedeuten

* **Optionsvolumen** zählt nur Kontrakte, deren letzter Handel am Handelstag war. Yahoo zeigt bei nicht gehandelten Kontrakten weiter das alte Volumen an; das wird herausgefiltert. Erfasst werden Verfallstermine bis 120 Tage.
* **z** ist der Abstand des heutigen Volumens (logarithmiert) vom Median der eigenen letzten 30 Tage, gemessen in robuster Streuung. **Faktor** ist Volumen geteilt durch Median. Unusual heißt z mindestens 2 und Faktor mindestens 2. Gerankt wird nach z, nicht nach absoluter Größe, denn eine große Zahl bei einer großen Aktie ist normal.
* **OI-Veränderung Top-Kontrakte**: Open Interest nach dem Handelstag minus vorher, summiert über die zehn aktivsten Kontrakte, geteilt durch deren Volumen. Ab plus 30 % gilt "überwiegend neu eröffnet", ab minus 30 % "überwiegend geschlossen".
* Kontrakte, die am Handelstag selbst verfallen, zählen zum Volumen, aber nicht zu den Top-Kontrakten: bei ihnen gibt es danach kein Open Interest mehr. Ist für weniger als die Hälfte des Top-Volumens eine Veränderung messbar, bleibt die Einordnung UNAVAILABLE.
* **OI-Basisprüfung** vergleicht das Open Interest vom Abend mit dem Stand vom Morgen davor. Weicht es ab, ist die Datenbasis unsicher.
* **Kurs-z**: Tagesrendite geteilt durch die eigene 20-Tage-Volatilität. Kandidat nur, wenn der Betrag unter 1 liegt.
* **Hinweise** sind langweilige Erklärungen, die der Code selbst erkennt: Quartalszahlen, Ex-Dividende (mit Muster für Dividendenarbitrage), Verfallstag und Quad Witching, branchenweit oder marktweit erhöhtes Volumen, Kurzläufer.

Nicht bestimmbar aus diesen Daten, bei jeder Flag: ob gekauft oder verkauft wurde, ob die Gegenseite eröffnet oder geschlossen hat, ob es eine Wette oder eine Absicherung ist.

## Trefferquote

`data/evaluation/summary.md` misst jede Flag 10 Handelstage später. Treffer heißt: Überrendite gegenüber dem Sektor-ETF von mindestens 2 Sigma, egal in welche Richtung, weil der Screener keine Richtung behauptet. Die Quote der Kandidaten steht neben der Basisrate aller Tickertage. Nur wenn sie klar darüber liegt, sagt der Screener mehr als der Zufall. Nach etwa drei Monaten lässt sich das zum ersten Mal ehrlich beurteilen.

## Dateien

```
config/universe.csv          S&P 100 mit Sektor und Sektor-ETF (Stand 21.09.2026)
config/settings.json         alle Schwellen
data/prices/<TICKER>.csv     ein Jahr Tageskurse, bei jedem Abendlauf erneuert
data/sessions/<Datum>/       pro Handelstag: Abend- und Morgendaten, analysis.csv, candidates.json, report.md
data/flags_log.csv           jede Code-Flag
data/evaluation/             outcomes.csv, summary.json, summary.md
data/latest.json             Zeiger auf den letzten analysierten Handelstag
prompts/flagging_agent.md    Prompt des Claude-Tasks
```

## Grenzen

* Die Daten kommen über yfinance von Yahoo Finance: kostenlos, inoffiziell, verzögert. Die Schnittstelle kann sich ändern oder drosseln. Vor jeder Entscheidung die Zahl an der Quelle prüfen.
* Wann Yahoo das Open Interest aktualisiert, ist nicht dokumentiert. Deshalb gibt es drei Morgenversuche; jeder weitere holt nur Ticker neu, die noch alt oder nicht abrufbar waren. Bleiben einzelne Ticker alt, steht im Bericht `TEILWEISE` und deren Veränderung ist UNAVAILABLE. Bleibt fast alles alt, steht dort `STALE` und es gibt an dem Tag keine Kandidaten.
* Feiertage verschieben den Verfallstag manchmal auf Donnerstag. Das wird nicht korrigiert.
* Optionen sind gehebelt und verfallen. Richtig in der Richtung und falsch im Timing verliert trotzdem alles.

## Plan B, falls Yahoo GitHub blockt

Wenn der Smoke-Test oder die täglichen Läufe mit Rate-Limit-Fehlern scheitern, blockt Yahoo die Rechenzentrums-Adressen von GitHub. Dann den Workflow auf einem eigenen Rechner laufen lassen (zum Beispiel der DGX Spark): Settings, Actions, Runners, New self-hosted runner, die angezeigten Befehle dort ausführen und in `.github/workflows/screener.yml` die Zeile `runs-on: ubuntu-latest` durch `runs-on: self-hosted` ersetzen. Alles andere bleibt gleich.

## Lokal

```
pip install -r requirements.txt
python -m pytest -q                     # Tests mit simuliertem Markt, ohne Netz
python -m screener.run smoke            # Live-Test mit drei Tickern
python -m screener.run evening --allow-past --symbols AAPL,MSFT
python -m screener.run morning
```
