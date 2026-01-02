# main.py
# Command-line runner

from src.pipeline.controller import PipelineController

if __name__ == "__main__":
    controller = PipelineController()
    output = controller.process_image("data/sample/sample.png")

    print("\nFINAL TEXT:\n")
    print(output["final_text"])
