import base64
import os
from pathlib import Path

from dotenv import load_dotenv
import mlflow
from openai import OpenAI


DATASET_PATH = Path("dataset")
EXPERIMENT_NAME = "siren-sky-benchmark"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
MODEL = "gpt-4.1-nano"
TRACKING_URI = "http://localhost:5000"


def get_openai_client():
    load_dotenv()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Create a .env file with OPENAI_API_KEY=your_api_key_here "
            "or set the environment variable before running."
        )

    return OpenAI(api_key=api_key)


def configure_mlflow():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.openai.autolog()


def encode_image(image_path):
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def classify_image_openai(client, image_path):
    base64_image = encode_image(image_path)
    response = client.responses.create(
        model=MODEL,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Is there garbage in this image? Answer with yes or no.",
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "auto",
                    },
                ],
            }
        ],
    )
    return response.output_text.strip()


def iter_images(dataset_path):
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    for category_path in sorted(path for path in dataset_path.iterdir() if path.is_dir()):
        for image_path in sorted(category_path.iterdir()):
            if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                yield category_path.name, image_path


def print_summary(results):
    print("\n" + "=" * 60)
    print("CLASSIFICATION RESULTS")
    print("=" * 60)
    for result in results:
        print(
            f"{result['file']!s:45} | "
            f"{result['category']:12} | "
            f"{result['classification']}"
        )
    print("=" * 60)


def main():
    configure_mlflow()
    client = get_openai_client()
    results = []

    for category, image_path in iter_images(DATASET_PATH):
        print(f"Processing: {image_path}...")

        try:
            classification = classify_image_openai(client, image_path)
            print(f"  -> {classification}")
        except Exception as exc:
            classification = f"Error: {exc}"
            print(f"  !! {classification}")

        results.append(
            {
                "file": image_path,
                "category": category,
                "classification": classification,
            }
        )

    print_summary(results)


if __name__ == "__main__":
    main()
