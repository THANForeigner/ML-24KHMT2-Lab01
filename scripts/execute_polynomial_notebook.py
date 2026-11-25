"""Execute the Polynomial Linear Regression notebook in a fresh clean kernel and save outputs."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
path = root / 'notebooks/sources/polynomial_linear_regression/polynomial_linear_regression.ipynb'
output = root / 'outputs/polynomial/executed_notebook.ipynb'
output.parent.mkdir(parents=True, exist_ok=True)

nb = nbformat.read(path, as_version=4)
nbformat.validate(nb)

print(f"[INFO] Bat dau thuc thi {path} trong clean kernel...")
try:
    client = NotebookClient(
        nb,
        timeout=1800,
        kernel_name='python3',
        resources={'metadata': {'path': str(root)}}
    )
    client.execute()
except Exception as exc:
    nbformat.write(nb, output)
    print(f"[ERROR] Thuc thi that bai: {exc}. Da luu partial notebook tai {output}")
    raise

nbformat.write(nb, output)
print(f"[SUCCESS] Hoan tat thuc thi notebook. Da luu ket qua tai {output}")
