# Multimedia Retrieval (HS26) — University of Basel

The **Multimedia Retrieval** course at the University of Basel explores information retrieval systems spanning text, images, audio, and video content.

## Repository Structure

```
mmir-unibasel-hs26/
├── publish/          ← Book + Quiz App (hosted via GitHub Pages)
│   ├── index.html    ← Jupyter Book output (built from mmir-unibasel-content/book)
│   ├── quiz/         ← Quiz application
│   └── ...
├── demos/            ← Interactive demo notebooks (Jupyter)
├── exercises/        ← Theoretical and practical exercises
├── learning/         ← Additional learning materials (book-related)
└── README.md
```

## Online Resources

- **Book & Quiz**: https://roger-weber.github.io/mmir-unibasel-hs26/
- **Quiz App**: https://roger-weber.github.io/mmir-unibasel-hs26/quiz/
- [ADAM University Basel](https://adam.unibas.ch/) (for enrolled students)
- [Public Web Site](https://dmi.unibas.ch/de/studium/computer-science-informatik/lehrangebot-hs26/lecture-multimedia-retrieval/)

## Getting Started

### Clone this repository

```bash
git clone https://github.com/roger-weber/mmir-unibasel-hs26.git
cd mmir-unibasel-hs26
```

### Run demos locally

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows

pip install -r demos/requirements.txt
jupyter notebook demos/
```

### Exercises

See the [exercises/](./exercises/) folder for theoretical and practical exercises assigned throughout the course.

## Helpful Software

- **Python**: [Download](https://www.python.org/downloads/) (3.11+)
- **Jupyter**: `pip install notebook`
- **IDE**: [VSCode](https://code.visualstudio.com/) with Python + Jupyter extensions
