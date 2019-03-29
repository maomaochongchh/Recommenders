import subprocess
subprocess.run(["pytest","tests/unit","-m","not notebooks and not spark and not gpu"]