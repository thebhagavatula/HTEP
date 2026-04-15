# main.py
# Command-line runner

from src.pipeline.controller import PipelineController

from src.config import SAMPLE_DATA_DIR

if __name__ == "__main__":
    controller = PipelineController()
    output = controller.process_image(str(SAMPLE_DATA_DIR / "sample.png"))

    print("\nFINAL TEXT:\n")
    print(output["final_text"])
