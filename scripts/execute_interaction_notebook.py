"""Execute the interaction notebook in a fresh kernel and save failures honestly."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
path = root/'notebooks/interaction_linear_regression.ipynb'
output = root/'outputs/interaction/executed_notebook.ipynb'
output.parent.mkdir(parents=True, exist_ok=True)
nb = nbformat.read(path, as_version=4)
nbformat.validate(nb)
try:
    NotebookClient(nb, timeout=1200, kernel_name='python3', resources={'metadata': {'path': str(root)}}).execute()
except Exception:
    nbformat.write(nb, output)
    raise
nbformat.write(nb, output)
print('Saved', output)
