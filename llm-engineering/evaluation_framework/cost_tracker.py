import math

def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    # Simple English heuristic: ~4 characters per token
    return max(1, math.ceil(len(text) / 4.0))

class CostTracker:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.judge_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_cost = 0.0

    def log_judge_call(self, input_tokens=500, output_tokens=100):
        # GPT-4o prices: $5.00 / 1M input tokens, $15.00 / 1M output tokens
        input_rate = 5.00 / 1_000_000
        output_rate = 15.00 / 1_000_000
        
        self.judge_calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_cost += (input_tokens * input_rate) + (output_tokens * output_rate)

    def get_summary(self):
        # Monthly projection assuming 10 eval runs per week
        # 1 run = this entire run's judge calls
        # 10 runs per week -> monthly runs = 10 * (52 / 12) = 43.3333 runs per month
        weekly_cost = self.total_cost * 10
        monthly_cost = weekly_cost * 4.3333 # 52 / 12 weeks per month
        
        return {
            "judge_calls": self.judge_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_cost": self.total_cost,
            "projected_weekly_cost": weekly_cost,
            "projected_monthly_cost": monthly_cost
        }

    def print_summary(self):
        summary = self.get_summary()
        print(f"\n========================= COST TRACKER REPORT =========================")
        print(f"  Total Judge Calls: {summary['judge_calls']}")
        print(f"  Input Tokens:      {summary['input_tokens']:,}")
        print(f"  Output Tokens:     {summary['output_tokens']:,}")
        print(f"  Total Run Cost:    ${summary['total_cost']:.5f}")
        print(f"  Projected Monthly Cost (10 runs/week): ${summary['projected_monthly_cost']:.2f}")
        print(f"=======================================================================\n")

# Global singleton
tracker = CostTracker()
