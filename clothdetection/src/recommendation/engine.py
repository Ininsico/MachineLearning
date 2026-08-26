from typing import List, Dict, Optional, Tuple
import random
from dataclasses import dataclass, field, asdict

from src.utils.config import cfg


@dataclass
class UserProfile:
    mood: str = "casual"
    weather: str = "mild"
    season: str = "summer"
    time_of_day: str = "day"
    preferences: List[str] = field(default_factory=list)
    detected_colors: List[str] = field(default_factory=list)
    detected_clothing: List[str] = field(default_factory=list)


@dataclass
class OutfitSuggestion:
    items: List[Dict] = field(default_factory=list)
    style: str = "casual"
    vibe: str = "neutral"
    color_palette: List[str] = field(default_factory=list)
    harmony_score: float = 0.0
    confidence: float = 0.0
    reasoning: str = ""

    def to_dict(self):
        return asdict(self)


class RecommendationEngine:
    def __init__(self):
        self.moods = cfg.get("recommendation", "moods", default={})
        self.weather = cfg.get("recommendation", "weather", default={})
        self.seasons = cfg.get("recommendation", "seasons", default={})
        self.color_harmonies = cfg.get("recommendation", "color_harmony", default={})
        self.categories = cfg.get("clothing", "categories", default={})

    def suggest(
        self,
        profile: UserProfile,
        n_suggestions: int = 3
    ) -> List[OutfitSuggestion]:
        suggestions = []

        mood_config = self.moods.get(profile.mood, self.moods.get("casual", {}))
        weather_config = self.weather.get(profile.weather, {})
        season_config = self.seasons.get(profile.season, {})

        base_colors = mood_config.get("colors", [])
        season_colors = season_config.get("colors", [])
        style = mood_config.get("style", "casual")
        vibe = mood_config.get("vibe", "neutral")

        color_palette = self._merge_color_palettes(base_colors, season_colors)

        for i in range(n_suggestions):
            outfit = self._generate_outfit(
                style=style,
                color_palette=color_palette,
                weather_config=weather_config,
                profile=profile,
                variant=i
            )

            harmony_score = self._calculate_harmony(
                [c["color"] for c in outfit if "color" in c]
            )

            reasoning = self._generate_reasoning(
                outfit, style, vibe, profile
            )

            suggestion = OutfitSuggestion(
                items=outfit,
                style=style,
                vibe=vibe,
                color_palette=color_palette,
                harmony_score=harmony_score,
                confidence=min(0.95, 0.6 + harmony_score * 0.3),
                reasoning=reasoning
            )
            suggestions.append(suggestion)

        suggestions.sort(key=lambda s: s.harmony_score, reverse=True)
        return suggestions

    def _merge_color_palettes(self, *palettes: List[str]) -> List[str]:
        merged = []
        seen = set()
        for palette in palettes:
            for color in palette:
                if color not in seen:
                    merged.append(color)
                    seen.add(color)
        return merged

    def _generate_outfit(
        self,
        style: str,
        color_palette: List[str],
        weather_config: Dict,
        profile: UserProfile,
        variant: int
    ) -> List[Dict]:
        outfit = []
        random.seed(variant)

        top_options = self.categories.get("tops", [])
        bottom_options = self.categories.get("bottoms", [])
        footwear_options = self.categories.get("footwear", [])

        avoid_items = set(weather_config.get("avoid", []))
        recommend_items = set(weather_config.get("recommend", []))

        top_options = [t for t in top_options if t not in avoid_items]
        bottom_options = [b for b in bottom_options if b not in avoid_items]
        footwear_options = [f for f in footwear_options if f not in avoid_items]

        if recommend_items:
            top_rec = [t for t in top_options if t in recommend_items]
            if top_rec:
                top_options = top_rec
            bottom_rec = [b for b in bottom_options if b in recommend_items]
            if bottom_rec:
                bottom_options = bottom_rec
            footwear_rec = [f for f in footwear_options if f in recommend_items]
            if footwear_rec:
                footwear_options = footwear_rec

        top = random.choice(top_options) if top_options else "t-shirt"
        top_color = random.choice(color_palette) if color_palette else "Blue"
        outfit.append({"type": top, "category": "top", "color": top_color})

        bottom = random.choice(bottom_options) if bottom_options else "jeans"
        bottom_color = self._get_complementary_color(top_color, color_palette)
        outfit.append({"type": bottom, "category": "bottom", "color": bottom_color})

        footwear = random.choice(footwear_options) if footwear_options else "sneakers"
        footwear_color = random.choice(color_palette) if color_palette else "White"
        outfit.append({"type": footwear, "category": "footwear", "color": footwear_color})

        if random.random() > 0.5:
            accessory_options = self.categories.get("accessories", [])
            if accessory_options:
                acc = random.choice(accessory_options)
                acc_color = random.choice(color_palette)
                outfit.append({"type": acc, "category": "accessory", "color": acc_color})

        return outfit

    def _get_complementary_color(self, color: str, palette: List[str]) -> str:
        for pair in self.color_harmonies.get("complementary", []):
            if color in pair:
                comp = pair[0] if pair[1] == color else pair[1]
                if comp in palette:
                    return comp
        others = [c for c in palette if c != color]
        return random.choice(others) if others else "Black"

    def _calculate_harmony(self, colors: List[str]) -> float:
        if len(colors) < 2:
            return 1.0
        score = 0.0
        pairs = 0
        for i, c1 in enumerate(colors):
            for c2 in colors[i + 1:]:
                pairs += 1
                if any(c1 in p and c2 in p for p in self.color_harmonies.get("complementary", [])):
                    score += 1.0
                elif any(c1 in g and c2 in g for g in self.color_harmonies.get("analogous", [])):
                    score += 0.8
                elif any(c1 in g and c2 in g for g in self.color_harmonies.get("monochromatic", [])):
                    score += 0.6
                else:
                    score += 0.3
        return score / pairs if pairs > 0 else 1.0

    def _generate_reasoning(
        self,
        outfit: List[Dict],
        style: str,
        vibe: str,
        profile: UserProfile
    ) -> str:
        parts = [f"{item['color']} {item['type']}" for item in outfit]
        outfit_desc = ", ".join(parts[:-1]) + f" and {parts[-1]}" if len(parts) > 1 else parts[0]

        reasons = [
            f"Perfect for a {vibe} {style} look",
            f"Matched to your {profile.mood} mood with complementary colors",
            f"Seasonally appropriate for {profile.season}",
        ]
        if profile.weather != "mild":
            reasons.append(f"Weather-optimized for {profile.weather} conditions")

        return f"Suggested outfit: {outfit_desc}. {random.choice(reasons)}."


if __name__ == "__main__":
    cfg.load()
    engine = RecommendationEngine()
    profile = UserProfile(
        mood="happy",
        weather="mild",
        season="summer",
        time_of_day="day"
    )
    suggestions = engine.suggest(profile, n_suggestions=3)
    for i, s in enumerate(suggestions):
        print(f"\nSuggestion {i + 1}:")
        print(f"  Style: {s.style}, Vibe: {s.vibe}")
        print(f"  Colors: {s.color_palette}")
        print(f"  Harmony: {s.harmony_score:.2f}")
        print(f"  Reasoning: {s.reasoning}")
