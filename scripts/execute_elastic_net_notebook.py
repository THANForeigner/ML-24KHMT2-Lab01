"""Execute the Elastic Net notebook in a fresh clean kernel and save outputs."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
path = root / 'notebooks/elastic_net_regression.ipynb'
output = root / 'outputs/elastic_net/executed_notebook.ipynb'
output.parent.mkdir(parents=True, exist_ok=True)

nb = nbformat.read(path, as_version=4)
nbformat.validate(nb)

print(f'Executing {path} in clean kernel...')
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
    print(f'Execution halted with error: {exc}. Saved partial notebook to {output}')
    raise

nbformat.write(nb, output)
print(f'Notebook execution completed successfully. Saved to {output}')
