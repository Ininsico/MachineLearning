from src.core.pipeline import FluxPipeline

def main():
    """Simple generation example"""
    
    pipeline = FluxPipeline()
    
    prompts = [
        "a beautiful sunset over mountains, highly detailed",
        "a cute cat wearing a space suit, digital art",
        "a futuristic city at night, neon lights, cyberpunk"
    ]
    
    for i, prompt in enumerate(prompts):
        print(f"\nGenerating image {i+1}/{len(prompts)}")
        print(f"Prompt: {prompt}")
        
        output_path = f"outputs/simple_{i}.png"
        
        pipeline(
            prompt,
            output_path=output_path,
            width=1024,
            height=1024,
            steps=4,
            guidance=3.5
        )
        
        print(f"✓ Saved to {output_path}")

if __name__ == "__main__":
    main()
