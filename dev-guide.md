### Jupyter Notebook set kernel from the project Virtual Environment (UV):

> To set a JupyterLab notebook kernel using uv, register the virtual environment as a kernel. Below commands include to create the environment, to add the kernel support, and to make it available in Jupyter:

**Steps to Set Up UV Kernel in JupyterLab**

1. **Initialize Project (Optional):** Create a new project directory and initialize it with uv. This is an optional step you may skip it if you already have own separate directory for your project:

```
uv init my-project
cd my-project
```

2. **Create Virtual Environment:** Create a `.venv` directory:

```
uv venv
```

3. **Install `ipykernel`:** Install inside the virtual environment:

```
uv pip install ipykernel
```

4. **Register Kernel:** Register this environment as a kernel in Jupyter:

```
uv run python -m ipykernel install --user --name=uv-kernel --display-name "Python (uv)"
```

5. Run JupyterLab: Launch JupyterLab (optionally using `uv` ):

```
uv run jupyter lab
```

6. Select Kernel: In the JupyterLab notebook, click the kernel name in the top right corner and select "Python (uv)".
