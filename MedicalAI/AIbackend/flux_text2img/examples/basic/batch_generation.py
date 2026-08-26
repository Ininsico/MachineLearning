from src.inference.batch.processor import BatchProcessor

def main():
    """Batch generation example"""
    
    processor = BatchProcessor(batch_size=4)
    
    # Load prompts from file
    with open("prompts.txt", "w") as f:
        f.write("a dog in a park\n")
        f.write("a cat on a roof\n")
        f.write("a bird in the sky\n")
        f.write("a fish in the ocean\n")
        f.write("a lion in the savanna\n")
    
    # Process batch
    output_paths = processor.process_from_file(
        "prompts.txt",
        "outputs/batch",
        width=512,
        height=512,
        steps=4
    )
    
    print(f"\n✓ Generated {len(output_paths)} images")
    for path in output_paths:
        if path:
            print(f"  - {path}")

if __name__ == "__main__":
    main()
