import os

project_structure = {
    "app": ["main.py", "config.py"],
    "app/api": [],
    "app/services": [],
    "app/models": [],
    "airflow_dags": ["training_pipeline_dag.py"],
    "docker": ["Dockerfile", "docker-compose.yml"],
    "ml": ["config.py"],
    "ml/data": [],
    "ml/pipeline": [],
    "ml/model": [],
    "ml/prediction": [],
    "ml/utils": ["logger.py"],
    "models": [],
    "notebooks": [],
    "scripts": ["train_model.py", "predict_model.py", "register_model.py"],
}

root_files = [".env", ".gitignore", "README.md", "requirements.txt"]

logger_template = '''import logging
import os

def get_logger(name: str, log_file: str = "logs/general.log"):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger
'''

def create_project():
    for folder, files in project_structure.items():
        os.makedirs(folder, exist_ok=True)
        for file in files:
            path = os.path.join(folder, file)

            if folder == "ml/utils" and file == "logger.py":
                with open(path, "w") as f:
                    f.write(logger_template)
            else:
                with open(path, "w") as f:
                    f.write(f"# {file}\n")

    for file in root_files:
        with open(file, "w") as f:
            f.write(f"# {file}\n")

    print("✅ ML project structure created!")

if __name__ == "__main__":
    create_project()