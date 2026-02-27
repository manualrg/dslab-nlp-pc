import os
import pickle
import shutil

def create_or_clean_folder(path: str):
    if not os.path.exists(path):
        print(f"Creating the folder: {path}")
        os.mkdir(path)
    else:
        print("Experiment folder already exists.")
        exp_files = os.listdir(path)
        if len(exp_files) == 0:
            print(f"No files to clean, the folder is empty")
        for file in exp_files:
            os.remove(
                os.path.join(path, file)
            )
            print(f"Removed: {file}")


def register_model(
        model,
        metadata,
        model_version_id,
        path_model_prod = os.path.join("models", "prod"),
        path_model_arch = os.path.join("models", "archive")
):
    
    os.makedirs(os.path.join(path_model_arch, model_version_id), exist_ok=True)

    for file in os.listdir(path_model_prod):
        
        src_path = os.path.join(path_model_prod, file)
        dst_path = os.path.join(path_model_arch, model_version_id, file)

        print(f"Archiving: {src_path} to {dst_path}")
        
        if os.path.isfile(src_path):
            shutil.move(src_path, dst_path)

    print(f"Registering artifacts in: {path_model_prod}")
    with open(os.path.join(path_model_prod, "model.pkl"), "wb") as file:
        pickle.dump( model, file)

    with open(os.path.join(path_model_prod, "metadata.pkl"), "wb") as file:
        pickle.dump(metadata, file)
