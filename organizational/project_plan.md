[Aktuell nur OptWare Notizen: Noch nicht im Optimierungsstream besprochen!]

# Ap3 Problemklasse Optimierung
Entwicklung der Benchmarking-Prozedur für Problemklassen der Optimierung.

## AP3.1 Metriken (07.2023 – 06.2024)
### Ziel
Mittels geeigneter Metriken sollen Mappings, En- bzw. Decodings und Kompilierung (AP3.3) sowie hybride Lösungsverfahren (AP3.4) gesamtheitlich und jeweils für sich vergleichbar sein.
Arten von Metriken:
  - Komplexitätsmaße, um die Laufzeit zu benchmarken
  - Lösungsgütemaße, um die Qualität der Lösung und die Wahrscheinlichkeit dieser zu vergleichen.
### Mindestanforderung
Die folgenden Metriken sind geprüft und spezifiziert
- [ ] Rechenaufwand (konventionelle und QC-Rechenzeit)
- [ ] Weitere Berechnungsaufwände (Design, Embedding und Latency), die durch den Einsatz des Quantencomputers entstehen
- [ ] Bewertung der Lösung (Validität und Qualität der Lösung)
- [ ] Technische QC-Metriken (Fidelity und Lösungsenergie)
- [ ] Bewertung der Lösungsstrategie (Konvergenz)
- [ ] Abschätzung der manuellen Aufwände bei der Anwendung (siehe auch Potenzialanalyse)

### Dokumentation/Bericht/Publikation
D1 Katalog von Metriken mit Einschätzung ihrer Anwendbarkeit

### Aktueller Stand
- Im [LaTeX PDF-Dokument](..%2Fdoc%2Fapplication_benchqc%2Fapplication_benchqc.pdf) finden sich eine Übersicht der Metriken
- Am 17.07.2024 vereinbart, es wäre sinnvoll sich den Q-Score noch anzuschauen (was genau ist das?) 



## AP3.2 Sequenzen (07.2023 - 06.2024)
### Ziel
Konkreten Anwendungen des AP7 wurden strukturiert, ausgewählt und auf wenige Basisprobleme mit geeigneter Komplexität reduziert. Der Fokus wurde auf die Notwendigkeit des Benchmarkings gelegt.

### Dokumentation/Bericht/Publikation
D2: Katalog von Problemsequenzen mit steigender Komplexität

### Aktueller Stand
- Fließbandabstimmung / Assembly Line Balancing -> 20er Instanzen sollen auf QC Hardware gerechnet werden können
- Reihenfolgeplanung -> Testdatengenerator ist vorhanden, kleinere Instanzen können erzeugt werden
- Testfahrzeugkonfiguration -> Kleine Instanzen fehlen noch
- Am 17.07.2024 vereinbart, evtl. Sensorpositioning Sequenzen hinzufügen, sofern der UseCase weiter verfolgt werden soll

## AP3.3 Mapping und Kompilierung (07.2023 – 06.2025)
### Ziel
Geeignete Strategien wie Optimierungsprobleme auf Quantencomputer abgebildet, übertragen und ausgewertet werden können, wurden entworfen. Beispielsweise als Hamiltonian aus einer QUBO-Formulierung.
### Dokumentation/Bericht/Publikation
D3: Implementierung ausgewählter Probleminstanzen auf QC-HW, abhängig von der Größe und den Eigenschaften der zur Verfügung stehenden HW
### Aktueller Stand
- Fließbandabstimmung / Assembly Line Balancing
  - “Klassisch” mit binärer Codierung
  - Eigene Penaltyterme
  - Experimente mit reduzierter Qubit Anzahl

## AP3.4 Lösungsverfahren (01.2024 - 12.2024)
### Ziel
Bei der Erstellung der Lösungsverfahren wurden stets aktuelle Forschungsergebnisse im Bereich hybrider Algorithmen zur kombinatorischen Optimierung im Blick behalten und auf Passfähigkeit analysiert und ggf. einbezogen. Insbesondere sollen hybride Ansätze untersucht worden sein.
### Dokumentation/Bericht/Publikation
D4: Katalog an Lösungsverfahren für kombinatorische Optimierungsprobleme
### Aktueller Stand
- Fließbandabstimmung / Assembly Line Balancing
  - Lösung kleinerer Instanzen mit bis zu 7 Tasks ist möglich
  - Presolving Verfahren um Instanzen zu verkleinern (ist es dann schon hybrid?)
  - Viele Ideen aus AP3.3 basieren auf aktuellen Papern
  - Weitere Ideen: Postprocessing (steepest descent)

## AP3.5 Potenzialanalyse (01.2025 – 12.2025)
### Ziel
Evaluation der gesamten Benchmarking-Prozedur in Zusammenarbeit aller beteiligten Partner wurde durchgeführt. Die Ergebnisse wurden in einer taskspezifischen Performancemessung für die Sequenzen komplexer werdender Probleminstanzen in einem technischen Bericht zusammengefasst. Eine Potenzialanalyse der Quantenansätze inklusive der spezifischen Wahl von Mapping/En-Deconding/Kompilierung (AP3.3) und Lösungsverfahren (AP3.4) wurde durchgeführt.
### Dokumentation/Bericht/Publikation
D5: technischer Bericht/Paper und fundierte ökonomische Abschätzung zur Potenzialanalyse basierend auf den 1) Metriken, 2) Probleminstanzen, 3) Mapping und 4) Lösungswegen.
### Aktueller Stand
- Am 17.07.2024 wurde vorgeschlagen, Paper zu diesem Thema zu fokussieren (Möglich wären zwei unterschiedliche Ansätze: Technischer Bericht und fundierte ökonomische Abschätzung)
- Fließbandabstimmung / Assembly Line Balancing
  - unterschiedlichen Ansätze aus AP3.3 ermöglichen eine Abschätzung für die benötigten Codierungs-Qubits
  - weitere Studien zu Qubit Zahl nach Embedding werden benötigt
  - beginn Experimente auf unterschiedlicher Hardware 

# AP4 Entwicklung Gesamtframework
Entwicklung der übergreifenden Architektur des Frameworks, in denen die Bausteine der anderen Arbeitspakete AP1-3 eingebettet werden. Ziel des Arbeitspakets ist weiterhin die konkrete Implementierung eines lauffähigen Codes der Benchmarking Suite sowie das Aufsetzen der notwendigen Entwicklungsumgebung wie z.B. Cloud Servern, die zur koordinierten Datenverarbeitung benötigt werden.

## AP4.1 Anforderungsanalyse (01.2023-12.2024)
### Ziel
Es wurde herausgearbeitet, welche Infrastruktur für das Gesamtframework verwendet werden soll.
### Anforderung
Beachtet wurden dabei die folgenden Punkte:
- [ ] Stand der Technik und Best Practices in der Benchmarking Suite
- [ ] insbesondere passend für die bereits untersuchte Fallanwendungen
- [ ] als Ausgangspunkt kann das Vorgehen in Quarks berücksichtigt werden
- [ ] auch lange bestehenden und bewährten Benchmarking Suites bei Problemen des klassischen maschinellen Lernens können als Beispiel dienen
- [ ] mögliche Clouddienste sollten mit allen Projektpartnern abgesprochen werden um möglicherweise sensible Daten der Projektpartner zu schützen
### Dokumentation/Bericht/Publikation
Internes Lastenheft, welches die konkret benötigte Infrastruktur zur Zusammenarbeit definiert.
### Aktueller Stand
- Am 17.07.2024 besprochen: BMW überprüft die Notwendigkeit der Erstellung eines Lastenheftes

## AP4.2 Architektur (07.2024-12.2024)
### Ziel
Die Anforderungen aus dem in 4.1 erstellten Lastenheft wurden in eine Softwarearchitektur übertragen.
### Anforderung
- [ ] höchstmögliche Modularität und Flexibilität gewahrt
- [ ] eine Erweiterung auf künftige Fallanwendungen und HW-Entwicklungen im Nachgang des Projekts soll möglich sein
### Dokumentation/Bericht/Publikation
Dokumentierte und mit den Projektpartnern abgestimmte Architektur als Klassendiagramm z.B. in UML. Darstellung sollte auch die gewünschten Anforderungen an Attribute und Methoden jeder Klasse enthalten.
### Aktueller Stand
- Wahrscheinlich soll Quark verwendet werden - die Gesamtarchitektur ist daher schon vorhanden 
- Es wird dauerhaft um Feedback zu Quark gebeten, um die Architektur zu verbessern


## AP4.3 Implementierung (01.2025-12.2025)
### Ziel
In AP4.2 entworfene Architektur soll als installierbares Pythonpaket implementiert, getestet und dokumentiert worden sein.
### Anforderung
- [ ] im Lastenheft niedergelegten Anforderungen werden nachvollziehbar implementiert und dokumentiert
- [ ] Cloud-Ressourcen wurden erzeugt, Zugänge angelegt und die Projektpartner in die Nutzung der Infrastruktur eingeführt
### Dokumentation/Bericht/Publikation
Abnahme der Infrastruktur durch die Projektpartner und ein Repository mit lauffähigem Code.
### Aktueller Stand
- Umsetzung eines allgemeinen Bin Packings in Quark (internes GitLab)
- Umsetzung Assembly Line Balancing in Quark (internes GitLab)


# Ap7 Use Cases Optimierung
Ziel ist die Vorbereitung der Optimierungs Use Cases auf die Lösung mithilfe von Quantenalgorithmen in AP3. Dazu wird der Kontext definiert, das Problem mathematisch modelliert und mithilfe von klassischen und heuristischen Ansätzen gelöst. Die Lösung kann dazu verwendet werden die QC-Algorithmen zu benchmarken.

## AP7.1 Kontext (01.2023-06.2023)
### Ziel
Technische Kennzahlen und die typische Größe industrierelevanter Probleme wurde ermittelt, um daraus ableiten zu können, wie viele Qubits zur Lösung des Problems benötigt wird.
Schritte
- [ ] definiere Relevanz des Problems für die Produktpalette
- [ ] konkretisiere Einordnung der wirtschaftlichen Verwertung
- [ ] sammle technische Kennzahlen (welche konkret?).
- [ ] erarbeite typische Größe industrierelevanten Probleme
- [ ] schlussfolgere Anzahl an benötigter Qubits zur Lösung des Problems
### Aktueller Stand
- Beschreibung der Use Cases im [LaTeX PDF-Dokument](..%2Fdoc%2Fapplication_benchqc%2Fapplication_benchqc.pdf) 
    - [x] Fließbandabstimmung / Assembly Line Balancing
    - [x] Reihenfolgeplanung 
    - [x] Testfahrzeugkonfiguration
    - [ ] ...

## AP7.2 Math. Modellierung (04.2023 – 06.2024)
### Ziel
Mathematische Modellierung der Use-Cases und iterative Anpassung an den Quantenalgorithmus in Abstimmung mit AP3.
### Schritte
- [ ] untersuche, inwieweit die einzelnen Use-Cases auf Grundproblemklassen zurückgeführt werden können.
- [ ] mehrere Anwendungen dürfen mit ähnlichen Verfahren mit Benchmarks untersucht werden
### Aktueller Stand
- Erste klassische Algorithmen formuliert
  - Fließbandabstimmung / Assembly Line Balancing
    - MIP-Binpacking, CP und heuristische Ansätze
  - Testfahrzeugkonfiguration
    - MIP-Binpacking (theoretisch)

## AP7.3 Klassische Lösung (09.2023-12.2025)
### Ziel
Klassische Lösungen für die Probleminstanzen wurden gefunden, so dass diese für das Benchmark mit dem Quantencomputer herangezogen werden können, um einen möglichen Quantenvorteil zu evaluieren.
### Anforderung
- [ ] Löse Optimierungsprobleme bis zu einer gewissen Problemgröße klassisch
- [ ] bei größeren Instanzen sollen z.B. heuristische Lösungsansätze eingesetzt werden
### Aktueller Stand
- Fließbandabstimmung / Assembly Line Balancing
  - MIP-Binpacking, CP und heuristische Ansätze sind umgesetzt
