"""
E-Commerce Fraud Detection Project
1st Semester Project
"""

print("=" * 60)
print("E-COMMERCE FRAUD DETECTION PROJECT")
print("CS-117: Applications of ICT")
print("NUST College of Electrical & Mechanical Engineering")
print("=" * 60)

print("\nStarting Project Execution...")
print("This will run all tasks automatically!\n")

# Run all tasks in sequence
try:
    print("Task 01: Data Preprocessing...")
    from src.data_preprocessing import main as task1
    task1()
    
    print("\nTask 02: Exploratory Data Analysis (EDA)...")
    from src.eda_visualizations import main as task2
    task2()
    
    print("\nTask 03: Class Balance Analysis...")
    from src.class_balance_analysis import main as task3
    task3()
    
    print("\nTask 04: Feature Selection & Classification...")
    from src.classification_models import main as task4
    task4()
    
    print("\n" + "=" * 60)
    print("ALL TASKS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nCheck the following folders for outputs:")
    print("1. data/processed/ - Cleaned data")
    print("2. results/plots/ - All visualizations")
    print("3. results/performance_metrics/ - Model results")
    
except Exception as e:
    print(f"\nError: {e}")
    print("Please check if all files are in the correct location.")

input("\nPress Enter to exit...")