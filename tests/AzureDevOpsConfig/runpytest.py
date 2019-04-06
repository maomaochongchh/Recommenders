import subprocess
#subprocess.run(["python", "-m", "pytest","tests/unit","-v" "-m","not notebooks and not spark and not gpu"])
subprocess.run(["pytest","tests/unit","-v" "-m","not notebooks and not spark and gpu", "--junitxml=reports/test-unit.xml"])
