import subprocess
from azureml.core import Run

subprocess.run(["pytest","tests/unit","-m","not notebooks and not spark and not gpu", "--junitxml=reports/test-unit.xml"])
#subprocess.run(["pytest","tests/unit/test_python_utils.py","-m","not notebooks and not spark and gpu", "--junitxml=reports/test-unit.xml"])


#run = Run.get_context()
#name_of_upload = "reports"
#path_on_disk = "reports"
#run.upload_folder(name_of_upload, path_on_disk)