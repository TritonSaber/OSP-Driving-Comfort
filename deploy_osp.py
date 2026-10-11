import os
from dotenv import load_dotenv
from rfdetr import RFDETRMedium

def main():
    load_dotenv(".env")

    x = RFDETRMedium(pretrain_weights="<path/to/pretrain/weights/dir>")

    x.deploy_to_roboflow(
        workspace=os.getenv("ROBOFLOW_WORKSPACE"),
        project_id=os.getenv("ROBOFLOW_DATASET_LATEST_ID"),
        api_key=os.getenv("ROBOFLOW_API_KEY"),
    )


if __name__ == "__main__":
    main()