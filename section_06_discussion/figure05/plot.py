"""Current Figure 5 entry point; previous renderer is retained in historical/."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).with_name("plot_current.py")),run_name="__main__")
