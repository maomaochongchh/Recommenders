import subprocess
#subprocess.run(["python", "-m", "pytest","tests/unit","-m","not notebooks and not spark and not gpu"])
subprocess.run(["pytest","tests/unit","-m","not notebooks and not spark and not gpu", "--junitxml=reports/test-unit.xml"])
