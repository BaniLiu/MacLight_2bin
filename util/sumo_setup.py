import os
import site
import sys
from pathlib import Path


def configure_sumo():
    if "SUMO_HOME" not in os.environ:
        for site_dir in site.getsitepackages():
            sumo_dir = Path(site_dir) / "sumo"
            if (sumo_dir / "tools").exists() and (sumo_dir / "bin").exists():
                os.environ["SUMO_HOME"] = str(sumo_dir)
                break

    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home:
        sumo_bin = str(Path(sumo_home) / "bin")
        path_parts = os.environ.get("PATH", "").split(os.pathsep)
        if sumo_bin not in path_parts:
            os.environ["PATH"] = os.pathsep.join([sumo_bin, *path_parts])

    if sys.platform.startswith("win"):
        os.environ.pop("LIBSUMO_AS_TRACI", None)
    else:
        os.environ.setdefault("LIBSUMO_AS_TRACI", "1")
