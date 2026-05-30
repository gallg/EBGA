# Installation Guide

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## Installation

### Recommended: Virtual Environment

```bash
# Create a virtual environment
python -m venv .venv

# Activate it
# On Linux/Mac:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate

# Install dependencies
pip install numpy scikit-learn scipy
```

### Direct Installation

If you don't want to use a virtual environment:

```bash
pip install numpy scikit-learn scipy
```

### Development Installation

For developing the EBGA framework:

```bash
# Clone the repository
git clone <repository-url>
cd EBGA

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install in development mode
pip install -e .

# Install development dependencies
pip install pytest sphinx
```

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | >=1.20 | Numerical operations |
| scikit-learn | >=0.24 | Data preprocessing, metrics, sklearn compatibility |
| scipy | >=1.7 | Scientific computing (optional, for some utils) |

## Verify Installation

```bash
# Run the main benchmark
python main.py

# You should see output like:
# EBGA Neural Network Framework - Benchmark Tests
# ============================================================
# Framework features:
#   ✓ Completely gradient-free (no backpropagation)
#   ...
```

If you see errors, please check:
1. Python version: `python --version`
2. Package versions: `pip list`
3. All dependencies are installed

## Troubleshooting

### ImportError: No module named 'EBGA'

**Solution:** Add the EBGA directory to your Python path:

```bash
# Option 1: Install in development mode
pip install -e .

# Option 2: Add to PYTHONPATH
export PYTHONPATH=${PYTHONPATH}:/path/to/EBGA

# Option 3: Run from the EBGA directory
cd /path/to/EBGA
python main.py
```

### ModuleNotFoundError: No module named 'numpy'

**Solution:** Install numpy:

```bash
pip install numpy
```

### Version Conflicts

**Solution:** Ensure all packages are up to date:

```bash
pip install --upgrade numpy scikit-learn scipy
```

### Permission Errors

**Solution:** Use a virtual environment or use `--user` flag:

```bash
pip install --user numpy scikit-learn scipy
```

## Platform-Specific Notes

### Linux

- All standard installation methods work
- Virtual environments recommended

### macOS

- All standard installation methods work
- Use `source .venv/bin/activate` to activate virtual environment

### Windows

- Use `.venv\Scripts\activate` to activate virtual environment
- May need to run commands as Administrator for system-wide installs
- Consider using WSL (Windows Subsystem for Linux) for better compatibility

## Docker (Optional)

For containerized installation:

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN pip install numpy scikit-learn scipy

CMD ["python", "main.py"]
```

Then build and run:

```bash
docker build -t ebga .
docker run -it ebga
```

## Testing

After installation, run the tests:

```bash
# Run main benchmarks
python main.py

# Run IXI dataset tests (if data is available)
python test_IXI.py
```

## Next Steps

Once installed, check out:
- [Tutorials](tutorials.md) - Step-by-step guides
- [API Reference](api.md) - Complete documentation
- [Framework Overview](overview.md) - Conceptual introduction
