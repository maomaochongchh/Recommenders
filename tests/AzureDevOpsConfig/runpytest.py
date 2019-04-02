import subprocess
subprocess.run(["pytest","tests/unit/test_python_utils.py","-m","not notebooks and not spark and not gpu"])