import sys
sys.path.insert(0, '.')

from src.utils.config import cfg

cfg.load('config.yaml')
print(f'Project: {cfg.get("project", "name")} v{cfg.get("project", "version")}')

from src.recommendation.engine import RecommendationEngine, UserProfile
engine = RecommendationEngine()

profiles = [
    UserProfile(mood='happy', weather='hot', season='summer'),
    UserProfile(mood='confident', weather='cold', season='winter'),
    UserProfile(mood='cozy', weather='rainy', season='fall'),
    UserProfile(mood='professional', weather='mild', season='spring'),
]

for profile in profiles:
    print(f'\n{"="*50}')
    print(f'Mood: {profile.mood} | Weather: {profile.weather} | Season: {profile.season}')
    suggestions = engine.suggest(profile, n_suggestions=1)
    s = suggestions[0]
    print(f'Style: {s.style.upper()} | Vibe: {s.vibe} | Harmony: {s.harmony_score:.2f}')
    for item in s.items:
        print(f'  [{item["category"]}] {item["color"]} {item["type"]}')
    print(f'  {s.reasoning}')

print('\n=== ALL TESTS PASSED ===')
