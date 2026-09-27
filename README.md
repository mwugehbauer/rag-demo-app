# RAG Lab – Chunking, Embeddings & Retrieval

Eine kleine Streamlit-Lehrdemo für:

**PDF → Parsing → Chunking → Embeddings → ChromaDB → Top-k Retrieval → Augmented Prompt**

Die beiden bereitgestellten Dokumente sind bereits im Ordner `documents/` enthalten:

- `PO_2026_MA_BM.pdf`
- `Modulhandbuch_Master_PO26__Stand_16.09.26.pdf`

## Empfohlene Umgebung

- Python **3.11** (am unkompliziertesten für Chroma / Sentence Transformers)
- VS Code
- Internetzugang beim **ersten** Start, damit das Embedding-Modell heruntergeladen werden kann

## 1. Projekt in VS Code öffnen

Entpacke den Projektordner und öffne `rag_demo_app` in VS Code.

## 2. Virtuelle Umgebung anlegen

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Falls PowerShell die Aktivierung blockiert, kannst du alternativ in der Windows-Eingabeaufforderung verwenden:

```cmd
.venv\Scripts\activate.bat
```

## 3. Pakete installieren

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Hinweis: `sentence-transformers` installiert auch PyTorch. Das kann beim ersten Mal einige Minuten dauern.

## 4. App starten

```powershell
streamlit run app.py
```

oder unter Windows per Doppelklick auf:

`run_windows.bat`

Der Browser öffnet normalerweise automatisch `http://localhost:8501`.

## 5. In der App

1. Beide Dokumente auswählen.
2. `Chunk Size = 250`, `Overlap = 50`, `Top-k = 5` belassen.
3. **Index erstellen / neu aufbauen** anklicken.
4. Im Tab **Chunks verstehen** einzelne Chunks und den Overlap ansehen.
5. Im Tab **Retrieval testen** eine Frage stellen.
6. Im Tab **RAG-Prompt ansehen** prüfen, welche Chunks tatsächlich in den LLM-Kontext gelangen würden.

## Gute Live-Experimente für die Vorlesung

### Experiment A – Chunk Size

- 100 Wörter
- 250 Wörter
- 500 Wörter

Frage: *Welche Voraussetzungen gelten für die Masterarbeit?*

Beobachte, welche Chunks gefunden werden und wie viel Kontext sie enthalten.

### Experiment B – Overlap

- 0 Wörter
- 50 Wörter
- 150 Wörter

Beobachte im Tab **Chunks verstehen**, welche Textteile zwischen zwei Chunks wiederholt werden.

### Experiment C – Top-k

- 2
- 5
- 10

Top-k verändert **nicht den Index**. Es verändert nur, wie viele Retrieval-Ergebnisse in den späteren LLM-Kontext gelangen.

## Wichtige didaktische Vereinfachungen

- Chunking ist **wortbasiert** und bleibt innerhalb einer PDF-Seite. Das macht Quelle und Seite leicht nachvollziehbar.
- Die App nutzt ein mehrsprachiges Embedding-Modell:
  `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- ChromaDB verwendet **Cosine Distance**.
- Die angezeigte Similarity wird für die Demo näherungsweise als `1 - distance` berechnet.
- Es wird absichtlich noch **kein LLM** aufgerufen. So bleibt sichtbar, was Retrieval und Augmentation leisten, bevor Generation hinzukommt.

## Projektstruktur

```text
rag_demo_app/
├── app.py
├── requirements.txt
├── README.md
├── run_windows.bat
├── .gitignore
└── documents/
    ├── PO_2026_MA_BM.pdf
    └── Modulhandbuch_Master_PO26__Stand_16.09.26.pdf
```

## Später mögliche Erweiterungen

- Claude API als finalen **Generate**-Schritt ergänzen
- semantisches Chunking statt fixer Wortfenster
- Hybrid Retrieval (BM25 + Vector Search)
- Reranking
- Similarity Threshold
- Evaluation mit einem festen Testset
- Quellenangaben direkt in der finalen Antwort
