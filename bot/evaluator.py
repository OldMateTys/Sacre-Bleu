import subprocess
import re
from collections import defaultdict

def run_simulation(cmd: list[str]) -> dict | None:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output_lines = result.stdout.strip().split('\n')
        
        if len(output_lines) < 7:
            print("[WARN] Output too short, skipping...")
            return None

        match_line = output_lines[-7]
        
        match = re.search(r"score=\{(.+?)\}", match_line)
        if not match:
            print(f"[WARN] Could not parse score line: {match_line}")
            return None
        
        score_str = match.group(1)
        scores = {}
        for kv in score_str.split(","):
            key, val = kv.split(":")
            scores[int(key.strip())] = int(val.strip())
        
        return scores

    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Simulation failed to run: {e}")
        return None

def simulate(cmd: list[str], runs: int = 10):
    total_scores = defaultdict(int)
    count_scores = defaultdict(int)
    failures = 0

    for i in range(runs):
        print(f"\n[INFO] Running simulation {i + 1}/{runs}...")
        scores = run_simulation(cmd)
        if scores is None:
            failures += 1
            continue
        
        for player, score in scores.items():
            total_scores[player] += score
            count_scores[player] += 1

    print("\n========== Simulation Summary ==========")
    for player in range(4):
        if count_scores[player]:
            avg = total_scores[player] / count_scores[player]
            print(f"Player {player} average score: {avg:.2f} ({count_scores[player]} valid games)")
        else:
            print(f"Player {player}: No valid games")

    print(f"\nTotal failures: {failures} out of {runs} runs")

# Example usage:
# Replace with your actual command (e.g. path to match_simulator.py)
simulate(["python3", "match_simulator.py", "--submissions", "2:old_bots/5th_bot.py", "2:bot/sacre_bleu.py", "--engine"], runs=1000)

