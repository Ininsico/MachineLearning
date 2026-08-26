import pandas as pd
import numpy as np
import os

class DataProcessor:
    @staticmethod
    def clean_raw_data(file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found.")
        df = pd.read_csv(file_path)
        df = df[pd.to_numeric(df['student_id'], errors='coerce').notnull()]
        cols = ['student_id', 'semester', 'credits', 'marks', 'gp', 'is_retake']
        for col in cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['is_retake'] = df['is_retake'].fillna(0).astype(int)
        df = df.dropna(subset=['student_id', 'semester', 'gp', 'marks'])
        return df.sort_values(['student_id', 'semester'])

    @staticmethod
    def aggregate_to_semesters(df):
        def weighted_average(group):
            return (group['gp'] * group['credits']).sum() / group['credits'].sum() if group['credits'].sum() > 0 else 0
        sem_data = df.groupby(['student_id', 'semester']).apply(
            lambda x: pd.Series({
                'avg_marks': x['marks'].mean(),
                'gpa': weighted_average(x),
                'total_credits': x['credits'].sum(),
                'courses': len(x),
                'retakes': x['is_retake'].sum(),
                'prog_ratio': (x['course_type'] == 'prog').mean()
            })
        ).reset_index()
        
        # Add Cumulative GPA and Change
        sem_data = sem_data.sort_values(['student_id', 'semester'])
        sem_data['cum_gpa'] = sem_data.groupby('student_id')['gpa'].expanding().mean().reset_index(level=0, drop=True)
        sem_data['gpa_change'] = sem_data.groupby('student_id')['gpa'].diff().fillna(0)
        
        return sem_data
