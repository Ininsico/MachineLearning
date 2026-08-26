from src.core.pipeline import FluxPipeline

def main():
    pipeline = FluxPipeline()
    
    prompt = "an image of couple, male and female in their 20s"
    output_path = "outputs/flux_output.png"
    
    pipeline(prompt, output_path=output_path)
    print(f"✅ Image saved to {output_path}")

if __name__ == "__main__":
    main()
