import argparse
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt

from src.utils.config import cfg
from src.detection.person_detector import PersonDetector
from src.detection.clothing_detector import ClothingDetector
from src.detection.color_analyzer import ColorAnalyzer
from src.recommendation.engine import RecommendationEngine, UserProfile
from src.utils.visualizer import Visualizer

console = Console()


class ClothMindApp:
    def __init__(self):
        cfg.load()
        console.print(Panel.fit("ClothMind AI - Clothing Detection & Recommendation", style="bold cyan"))

        with console.status("Initializing models..."):
            self.person_detector = PersonDetector()
            self.clothing_detector = ClothingDetector()
            self.color_analyzer = ColorAnalyzer()
            self.recommendation_engine = RecommendationEngine()
            self.visualizer = Visualizer()

        console.print("All models loaded successfully!", style="green")

    def analyze_image(self, image_path: str, show: bool = True, save: bool = False):
        img = cv2.imread(image_path)
        if img is None:
            console.print(f"[red]Error: Cannot read image {image_path}[/red]")
            return None

        console.print(f"\n[bold]Analyzing:[/bold] {Path(image_path).name}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        with console.status("Detecting people..."):
            people = self.person_detector.detect_people(img)
        console.print(f"  People detected: {len(people)}", style="yellow")

        all_clothing = []
        if people:
            for i, person in enumerate(people):
                with console.status(f"Detecting clothing on person {i + 1}..."):
                    clothing_items = self.clothing_detector.detect_clothing(person["cropped"])
                    for item in clothing_items:
                        orig_x1, orig_y1, orig_x2, orig_y2 = person["bbox"]
                        item["bbox"] = (
                            item["bbox"][0] + orig_x1,
                            item["bbox"][1] + orig_y1,
                            item["bbox"][2] + orig_x1,
                            item["bbox"][3] + orig_y1,
                        )
                    all_clothing.extend(clothing_items)
                console.print(f"  Person {i + 1}: {len(clothing_items)} clothing items")
        else:
            with console.status("No people detected, scanning full image..."):
                all_clothing = self.clothing_detector.detect_clothing(img)
            console.print(f"  Items detected: {len(all_clothing)}")

        all_colors = []
        for item in all_clothing:
            x1, y1, x2, y2 = item["bbox"]
            colors = self.color_analyzer.analyze_region(img, (x1, y1, x2, y2))
            all_colors.extend(colors)
            if colors:
                item["dominant_color"] = colors[0]["name"]

        unique_colors = list(dict.fromkeys(c["name"] for c in all_colors))

        self._display_results(all_clothing, all_colors)

        result_img = None
        if show or save:
            result_img = self.visualizer.draw_detection_results(
                img, people, all_clothing, all_colors
            )
            if show:
                cv2.imshow("ClothMind AI - Analysis", result_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            if save:
                out_path = Path("output") / f"analyzed_{Path(image_path).name}"
                out_path.parent.mkdir(exist_ok=True)
                cv2.imwrite(str(out_path), result_img)
                console.print(f"Saved: {out_path}", style="green")

        return {
            "image": result_img,
            "people": people,
            "clothing": all_clothing,
            "colors": all_colors,
            "unique_colors": unique_colors
        }

    def _display_results(self, clothing: list, colors: list):
        if clothing:
            table = Table(title="Detected Clothing", show_header=True, header_style="bold magenta")
            table.add_column("Item", style="cyan")
            table.add_column("Category", style="green")
            table.add_column("Confidence", style="yellow")
            table.add_column("Color", style="blue")

            for item in clothing:
                color_name = item.get("dominant_color", "?")
                table.add_row(
                    item["class"],
                    item["category"],
                    f"{item['confidence']:.2f}",
                    color_name
                )
            console.print(table)

        if colors:
            color_str = ", ".join([f"{c['name']} ({c['percentage']:.0f}%)" for c in colors[:5]])
            console.print(f"\n[bold]Dominant Colors:[/bold] {color_str}")

    def get_recommendations(self, analysis: dict = None, interactive: bool = True):
        profile = UserProfile()

        if analysis and analysis.get("unique_colors"):
            profile.detected_colors = analysis["unique_colors"]
        if analysis and analysis.get("clothing"):
            profile.detected_clothing = [c["class"] for c in analysis["clothing"]]

        if interactive:
            console.print("\n[bold cyan]--- Outfit Recommendation ---[/bold cyan]")

            mood_options = list(cfg.get("recommendation", "moods", default={}).keys())
            console.print(f"Moods: {', '.join(mood_options)}")
            profile.mood = Prompt.ask("Enter your mood", default="casual")

            weather_options = list(cfg.get("recommendation", "weather", default={}).keys())
            console.print(f"Weather: {', '.join(weather_options)}")
            profile.weather = Prompt.ask("Enter weather", default="mild")

            season_options = list(cfg.get("recommendation", "seasons", default={}).keys())
            console.print(f"Seasons: {', '.join(season_options)}")
            profile.season = Prompt.ask("Enter season", default="summer")

        suggestions = self.recommendation_engine.suggest(profile, n_suggestions=3)

        console.print("\n[bold green]Top Recommendations:[/bold green]")
        for i, suggestion in enumerate(suggestions):
            panel = Panel(
                f"[bold]{suggestion.style.upper()}[/bold] | {suggestion.vibe.capitalize()}\n"
                f"Colors: {', '.join(suggestion.color_palette)}\n"
                f"Harmony: {suggestion.harmony_score:.2f} | Confidence: {suggestion.confidence:.2f}\n"
                f"\n{suggestion.reasoning}",
                title=f"Suggestion {i + 1}",
                border_style="cyan"
            )
            console.print(panel)

        if interactive:
            save = Prompt.ask("Save visualizations?", choices=["y", "n"], default="y")
            if save == "y":
                for i, s in enumerate(suggestions):
                    path = f"output/suggestion_{i + 1}.png"
                    self.visualizer.show_outfit_suggestion(s, save_path=path)
                console.print("Saved to output/ folder", style="green")

        return suggestions

    def run_webcam(self):
        console.print("\n[bold cyan]Starting webcam detection...[/bold cyan]")
        console.print("Press 'q' to quit, 's' to save frame", style="yellow")
        console.print("[dim]Detection runs every 10 frames to keep it fast[/dim]")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            console.print("[red]Error: Cannot open webcam[/red]")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        frame_count = 0
        people = []
        all_clothing = []
        all_colors = []
        last_status = ""
        status_timer = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            display = frame.copy()
            frame_count += 1

            if frame_count % 10 == 1:
                people = self.person_detector.detect_people(frame)

                all_clothing = []
                for person in people:
                    items = self.clothing_detector.detect_clothing(person["cropped"])
                    for item in items:
                        ox1, oy1, ox2, oy2 = person["bbox"]
                        item["bbox"] = (
                            item["bbox"][0] + ox1,
                            item["bbox"][1] + oy1,
                            item["bbox"][2] + ox1,
                            item["bbox"][3] + oy1,
                        )
                    all_clothing.extend(items)

                all_colors = []
                for item in all_clothing:
                    x1, y1, x2, y2 = item["bbox"]
                    colors = self.color_analyzer.analyze_region(frame, (x1, y1, x2, y2))
                    all_colors.extend(colors)

                if people:
                    clothes = [f"{c['category']}:{c['class']}" for c in all_clothing]
                    last_status = f"{len(people)} person | {', '.join(clothes)}"
                else:
                    last_status = "No person detected"
                status_timer = 30

            if status_timer > 0:
                status_timer -= 1
                cv2.rectangle(display, (5, 5), (635, 35), (0, 0, 0), -1)
                cv2.putText(display, f"ClothMind: {last_status}",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if people:
                    for person in people:
                        x1, y1, x2, y2 = person["bbox"]
                        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    for item in all_clothing:
                        x1, y1, x2, y2 = item["bbox"]
                        cmap = {"tops": (255,0,0), "bottoms": (0,255,0),
                                "full_body": (255,255,0), "footwear": (0,255,255),
                                "accessories": (255,0,255)}
                        c = cmap.get(item["category"], (128,128,128))
                        cv2.rectangle(display, (x1, y1), (x2, y2), c, 2)
                        cv2.putText(display, f"{item['class']} {item.get('dominant_color','')}",
                                    (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 2)
            else:
                cv2.rectangle(display, (5, 5), (635, 25), (0, 0, 0), -1)
                cv2.putText(display, "ClothMind AI - detecting...",
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            cv2.imshow("ClothMind AI - Live", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s'):
                save_path = f"output/webcam_capture_{frame_count}.jpg"
                cv2.imwrite(save_path, display)
                console.print(f"Saved: {save_path}", style="green")

        cap.release()
        cv2.destroyAllWindows()
        console.print("Webcam closed", style="yellow")

    def interactive_menu(self):
        while True:
            console.print("\n[bold cyan]===== ClothMind AI Menu =====[/bold cyan]")
            console.print("1. Analyze an image")
            console.print("2. Get outfit recommendations")
            console.print("3. Analyze + Recommend (full pipeline)")
            console.print("4. Start webcam detection")
            console.print("5. Exit")

            choice = Prompt.ask("Select option", choices=["1", "2", "3", "4", "5"])

            if choice == "1":
                path = Prompt.ask("Image path")
                if Path(path).exists():
                    self.analyze_image(path, show=True, save=True)
                else:
                    console.print("[red]File not found[/red]")

            elif choice == "2":
                self.get_recommendations()

            elif choice == "3":
                path = Prompt.ask("Image path")
                if Path(path).exists():
                    analysis = self.analyze_image(path, show=False, save=True)
                    if analysis:
                        self.get_recommendations(analysis)
                else:
                    console.print("[red]File not found[/red]")

            elif choice == "4":
                self.run_webcam()

            elif choice == "5":
                console.print("Goodbye!", style="bold green")
                break


def main():
    parser = argparse.ArgumentParser(description="ClothMind AI - Clothing Detection & Recommendation")
    parser.add_argument("--image", "-i", help="Path to image file")
    parser.add_argument("--webcam", "-w", action="store_true", help="Run webcam detection")
    parser.add_argument("--menu", "-m", action="store_true", help="Interactive menu mode")
    parser.add_argument("--no-display", action="store_true", help="Don't show image")

    args = parser.parse_args()

    app = ClothMindApp()

    if args.menu:
        app.interactive_menu()
    elif args.webcam:
        app.run_webcam()
    elif args.image:
        app.analyze_image(args.image, show=not args.no_display, save=True)
        app.get_recommendations(interactive=True)
    else:
        app.interactive_menu()


if __name__ == "__main__":
    main()
