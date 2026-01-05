
# Trend-Aware Adaptive Operator Control (TA-AOC)


import wandb
import psutil
import platform
import GPUtil
import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, deque
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from deap import base, creator, tools, algorithms
from typing import List, Optional
import asyncio
from typing import List, Dict
import json
import re
import logging
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
# from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM
from langchain_core.pydantic_v1 import BaseModel
from langchain_ollama.embeddings import OllamaEmbeddings
import hashlib
import torch
import torch.nn as nn
import torch.optim as optim


output_dir = "reports/ga_with_llm"
os.makedirs(output_dir, exist_ok=True)


# ------------------------------
# Configurations
# ------------------------------
POPULATION_SIZE = 50
P_CROSSOVER = 0.9
P_MUTATION = 0.2
MAX_GENERATIONS = 50
HALL_OF_FAME_SIZE = 5
FEATURE_PENALTY_FACTOR = 0.001
RANDOM_SEED = 42
random.seed(RANDOM_SEED)


# ------------------------------
# LLM Integration Manager
# ------------------------------

class LLMIntegrationManager:
    def __init__(self, llm_config: Dict):
        self.verbose = llm_config.get('verbose', False)
        self.llm = OllamaLLM(**{k: v for k, v in llm_config.items() if k != 'verbose'})
        self.parser = JsonOutputParser()
        self.cache = {}
        self.json_pattern = re.compile(r'\{.*\}', re.DOTALL)
        self.logger = logging.getLogger('LLM_Logger')
        self.logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)
  

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def async_query(
        self,
        prompt: str,
        context: Dict,
        *,
        required_keys: Optional[list[str]] = None   # ← keyword-only
    ) -> Dict:
        cache_key = hash(prompt + str(context))
        if cache_key in self.cache:
            if self.verbose:
                self.logger.debug(f"Using cached response for prompt: {prompt[:50]}...")
            return self.cache[cache_key]

        try:
            if self.verbose:
                self.logger.debug("\n" + "="*40 + " LLM REQUEST " + "="*40)
                self.logger.debug(f"\nPROMPT:\n{prompt}")
                self.logger.debug(f"\nCONTEXT:\n{json.dumps(context, indent=2)}")
                self.logger.debug("="*93 + "\n")

            structured_prompt = ChatPromptTemplate.from_template(
                f"{prompt}\n\nRespond ONLY with valid JSON matching the specified format."
            )
            chain = structured_prompt | self.llm | self.parser
            print("📡 Sending LLM request...")
            result = await chain.ainvoke(context)
            print(result)
            print("✅ LLM responded!")
            validated = self._validate_and_clean(result, prompt)

            # Enforce keys if requested
            if required_keys:
                missing = [k for k in required_keys if k not in validated]
                if missing:
                    raise ValueError(f"LLM JSON missing required keys: {missing}")

            if self.verbose:
                self.logger.debug("\n" + "="*40 + " LLM RESPONSE " + "="*39)
                self.logger.debug(f"\nRAW RESPONSE:\n{result}")
                self.logger.debug(f"\nVALIDATED RESPONSE:\n{json.dumps(validated, indent=2)}")
                self.logger.debug("="*93 + "\n")

            self.cache[cache_key] = validated
            return validated

        except Exception as e:
            self.logger.error(f"LLM query failed: {str(e)}")
            return self._generate_fallback(e, prompt, context)

    def _validate_and_clean(self, response: any, prompt: str) -> Dict:
        if isinstance(response, dict):
            return response
            print("🔍 Valid JSON response received from LLM.")
            print(json.dumps(response, indent=2))

        if isinstance(response, str):
            try:
                json_str = self.json_pattern.search(response).group()
                return json.loads(json_str)
            except (AttributeError, json.JSONDecodeError):
                pass

        return self._generate_fallback(
            ValueError("No valid JSON found in response"),
            prompt,
            {"response": response}
        )

    def _generate_fallback(self, error: Exception, prompt: str, context: Dict) -> Dict:
        logging.warning(f"Generated fallback response for prompt: {prompt}")
        return {
            "error": str(error),
            "fallback": True,
            "response": context.get("response", ""),
            "timestamp": datetime.now().isoformat()
        }

# --- Canonical names & short descriptions for abbreviated features ---
FEATURE_ALIASES = {
    # Statistical
    "entropy":   {"full": "Shannon Entropy", "category": "statistical", "desc": "Uncertainty of the price/feature distribution."},
    "kurtosis":  {"full": "Excess Kurtosis", "category": "statistical", "desc": "Tail heaviness of returns/feature distribution."},
    "mad":       {"full": "Median Absolute Deviation", "category": "statistical", "desc": "Robust dispersion around median."},
    "median":    {"full": "Median", "category": "statistical", "desc": "Middle value of the series."},
    "quantile_0.5": {"full": "Quantile (0.5)", "category": "statistical", "desc": "50th percentile (median)."},
    "skew":      {"full": "Skewness", "category": "statistical", "desc": "Asymmetry of the distribution."},
    "stdev":     {"full": "Standard Deviation", "category": "statistical", "desc": "Volatility/dispersion measure."},
    "variance":  {"full": "Variance", "category": "statistical", "desc": "Squared deviation from mean."},
    "zscore":    {"full": "Z-Score", "category": "statistical", "desc": "Standardized distance from mean."},

    # Volume / money flow
    "ad":        {"full": "Accumulation/Distribution", "category": "volume", "desc": "Price-volume flow proxy."},
    "adosc":     {"full": "Chaikin Oscillator", "category": "volume", "desc": "Momentum of A/D line."},
    "mfi":       {"full": "Money Flow Index", "category": "volume", "desc": "Volume-weighted RSI; overbought/oversold."},
    "nvi":       {"full": "Negative Volume Index", "category": "volume", "desc": "Focus on down-volume days."},
    "obv":       {"full": "On-Balance Volume", "category": "volume", "desc": "Cumulative volume flow."},
    "pvi":       {"full": "Positive Volume Index", "category": "volume", "desc": "Focus on up-volume days."},
    "pvt":       {"full": "Price Volume Trend", "category": "volume", "desc": "Cumulative price*volume changes."},
    "pvr":       {"full": "Price to Volume Ratio", "category": "volume", "desc": "Price-volume proportion."},
    "pvol":      {"full": "Price-Volume Oscillator", "category": "volume", "desc": "Volume momentum vs. baseline."},
    "cmf":       {"full": "Chaikin Money Flow", "category": "volume", "desc": "Distribution/accumulation over window."},
    "efi":       {"full": "Elder’s Force Index", "category": "volume", "desc": "Price change × volume."},
    "eom":       {"full": "Ease of Movement", "category": "volume", "desc": "Price change vs. volume/box ratio."},

    # Trend
    "adx":       {"full": "Average Directional Index", "category": "trend", "desc": "Trend strength (not direction)."},
    "dmp":       {"full": "+DI (Positive Directional Indicator)", "category": "trend", "desc": "Upward movement component."},
    "dmn":       {"full": "-DI (Negative Directional Indicator)", "category": "trend", "desc": "Downward movement component."},
    "aroon_up":  {"full": "Aroon Up", "category": "trend", "desc": "Time since highest high; trend timing."},
    "aroon_dn":  {"full": "Aroon Down", "category": "trend", "desc": "Time since lowest low; trend timing."},
    "aroon_osc": {"full": "Aroon Oscillator", "category": "trend", "desc": "Aroon_up − Aroon_down."},
    "chop":      {"full": "Choppiness Index", "category": "trend", "desc": "Trendiness vs. range-bound."},
    "cksp_stop": {"full": "Chande Kroll Stop", "category": "trend", "desc": "Volatility-based trailing stop."},
    "qstick":    {"full": "Qstick", "category": "trend", "desc": "EMA of close−open; direction bias."},
    "psar":      {"full": "Parabolic SAR", "category": "trend", "desc": "Trend-following stop-and-reverse."},
    "vortex_vi_plus":  {"full": "Vortex +VI", "category": "trend", "desc": "Uptrend flow strength."},
    "vortex_vi_minus": {"full": "Vortex −VI", "category": "trend", "desc": "Downtrend flow strength."},

    # Momentum / oscillators
    "ao":        {"full": "Awesome Oscillator", "category": "momentum", "desc": "Momentum via SMA differences."},
    "apo":       {"full": "Absolute Price Oscillator", "category": "momentum", "desc": "Fast EMA − Slow EMA."},
    "bop":       {"full": "Balance of Power", "category": "momentum", "desc": "Buyer vs. seller strength."},
    "cci":       {"full": "Commodity Channel Index", "category": "momentum", "desc": "Deviation from typical price."},
    "cfo":       {"full": "Chande Forecast Oscillator", "category": "momentum", "desc": "Price vs. linear forecast."},
    "cmo":       {"full": "Chande Momentum Oscillator", "category": "momentum", "desc": "Normalized up/down momentum."},
    "er":        {"full": "Efficiency Ratio (Kaufman)", "category": "momentum", "desc": "Trendiness vs. noise."},
    "inertia":   {"full": "Inertia Indicator", "category": "momentum", "desc": "KAMA-smoothed trend inertia."},
    "mom":       {"full": "Momentum", "category": "momentum", "desc": "Price change over window."},
    "pgo":       {"full": "Pretty Good Oscillator", "category": "momentum", "desc": "Deviation from simple band."},
    "roc":       {"full": "Rate of Change", "category": "momentum", "desc": "Percentage change over window."},
    "rsi":       {"full": "Relative Strength Index", "category": "momentum", "desc": "Overbought/oversold oscillator."},
    "rsx":       {"full": "RSI-X (Better RSI)", "category": "momentum", "desc": "Smoothed RSI variant."},
    "uo":        {"full": "Ultimate Oscillator", "category": "momentum", "desc": "Multi-window momentum blend."},
    "willr":     {"full": "Williams %R", "category": "momentum", "desc": "Position of close in recent range."},
    "fisher":    {"full": "Fisher Transform", "category": "momentum", "desc": "Normalized turning points."},
    "kst":       {"full": "Know Sure Thing", "category": "momentum", "desc": "ROC-based momentum composite."},
    "ppo":       {"full": "Percentage Price Oscillator", "category": "momentum", "desc": "%MACD style oscillator."},
    "pvo":       {"full": "Percentage Volume Oscillator", "category": "momentum", "desc": "%EMA of volume."},
    "qqe":       {"full": "Qualitative Quantitative Estimation", "category": "momentum", "desc": "RSI-based adaptive bands."},
    "rvgi":      {"full": "Relative Vigor Index", "category": "momentum", "desc": "Closes relative to high-low."},
    "smi":       {"full": "Stochastic Momentum Index", "category": "momentum", "desc": "Stochastic of close relative to midpoint."},
    "stochrsi_k": {"full": "Stochastic RSI %K", "category": "momentum", "desc": "Stochastic of RSI (fast)."},
    "stochrsi_d": {"full": "Stochastic RSI %D", "category": "momentum", "desc": "Stochastic of RSI (signal)."},
    "trix":      {"full": "TRIX", "category": "momentum", "desc": "Triple-smoothed ROC."},
    "tsi":       {"full": "True Strength Index", "category": "momentum", "desc": "Double-smoothed momentum."},
    "macd":      {"full": "MACD", "category": "momentum", "desc": "EMA momentum with signal & hist."},

    # Volatility / bands
    "aberration_upper": {"full": "Aberration Channel (Upper)", "category": "volatility", "desc": "Volatility-based channel."},
    "aberration_middle":{"full": "Aberration Channel (Middle)", "category": "volatility", "desc": "Center line."},
    "aberration_lower": {"full": "Aberration Channel (Lower)", "category": "volatility", "desc": "Volatility-based channel."},
    "bb_lower":  {"full": "Bollinger Bands (Lower)", "category": "volatility", "desc": "Lower band, k·σ below mean."},
    "bb_middle": {"full": "Bollinger Bands (Middle)", "category": "volatility", "desc": "SMA used for bands."},
    "bb_upper":  {"full": "Bollinger Bands (Upper)", "category": "volatility", "desc": "Upper band, k·σ above mean."},
    "donchian_lower": {"full": "Donchian Channel (Lower)", "category": "volatility", "desc": "Lowest low window."},
    "donchian_middle":{"full": "Donchian Channel (Middle)", "category": "volatility", "desc": "Midpoint of channel."},
    "donchian_upper": {"full": "Donchian Channel (Upper)", "category": "volatility", "desc": "Highest high window."},
    "atr":       {"full": "Average True Range", "category": "volatility", "desc": "True range averaged."},
    "natr":      {"full": "Normalized ATR", "category": "volatility", "desc": "ATR normalized by price."},
    "kc_upper":  {"full": "Keltner Channel (Upper)", "category": "volatility", "desc": "EMA ± ATR multiple."},
    "kc_middle": {"full": "Keltner Channel (Middle)", "category": "volatility", "desc": "EMA used for channel."},
    "kc_lower":  {"full": "Keltner Channel (Lower)", "category": "volatility", "desc": "EMA − ATR multiple."},
    "massi":     {"full": "Mass Index", "category": "volatility", "desc": "Range expansion/contract cycles."},
    "rvi":       {"full": "Relative Volatility Index", "category": "volatility", "desc": "RSI on standard deviation."},
    "ui":        {"full": "Ulcer Index", "category": "volatility", "desc": "Drawdown-based risk."},

    # Price/base
    "open":   {"full": "Open Price", "category": "price", "desc": "Open price of the bar."},
    "high":   {"full": "High Price", "category": "price", "desc": "High price of the bar."},
    "low":    {"full": "Low Price", "category": "price", "desc": "Low price of the bar."},
    "close":  {"full": "Close Price", "category": "price", "desc": "Close price of the bar."},
    "volume": {"full": "Volume", "category": "price", "desc": "Number of units traded."},
}


def _canonicalize(feat: str) -> dict:
    """Return canonical name/category/desc for a feature; fall back gracefully."""
    key = feat.lower()
    base = FEATURE_ALIASES.get(key)
    if base:
        return {
            "name": feat,                    # original column name
            "canonical": base["full"],
            "category": base["category"],
            "desc": base["desc"]
        }
    # Fallback: infer category from keywords and create a readable title
    title = feat.replace("_", " ").title()
    category = (
        "price"       if any(k in key for k in ["open","close","high","low","volume"]) else
        "statistical" if any(k in key for k in ["entropy","kurtosis","mad","median","quantile","skew","stdev","variance","zscore"]) else
        "volume"      if any(k in key for k in ["ad","adosc","mfi","nvi","obv","pvi","pvol","pvr","pvt","cmf","efi","eom"]) else
        "trend"       if any(k in key for k in ["adx","dmp","dmn","aroon","chop","cksp","qstick","psar","vortex"]) else
        "momentum"    if any(k in key for k in ["ao","apo","bop","cci","cfo","cmo","er","inertia","mom","pgo","roc","rsi","rsx","uo","willr","fisher","kst","ppo","pvo","qqe","rvgi","smi","stochrsi","trix","tsi","macd"]) else
        "volatility"  if any(k in key for k in ["aberration","bb_","donchian","atr","natr","kc_","massi","rvi","ui"]) else
        "other"
    )
    return {
        "name": feat,
        "canonical": title,
        "category": category,
        "desc": f"Auto-expanded name for '{feat}'."
    }

def build_taxonomy_and_glossary(features: list) -> tuple[dict, list]:
    taxonomy = {"price": [], "statistical": [], "volume": [], "trend": [], "momentum": [], "volatility": [], "other": []}
    glossary = []
    for feat in features:
        meta = _canonicalize(feat)
        taxonomy[meta["category"]].append(feat)   # keep original names in buckets
        glossary.append(meta)                     # include full info for LLM
    return taxonomy, glossary

# ------------------------------
# LLM Manager
# ------------------------------
class LLMGuidedInitializer:
    def __init__(self, llm_manager, all_features: List[str], market_context: Dict, pop_size: int = 50):
        self.llm_manager = llm_manager
        self.all_features = all_features
        self.market_context = market_context
        self.pop_size = pop_size

    async def generate_population(self) -> List[List[int]]:
        prompt = PROMPT_TEMPLATES['initial_population']

        taxonomy, glossary = build_taxonomy_and_glossary(self.all_features)

        context = {
            "features": self.all_features,   # original column names
            "pop_size": self.pop_size,
            "market_context": self.market_context,
            # category buckets (original names)
            "price_features": taxonomy["price"],
            "statistical_features": taxonomy["statistical"],
            "volume_features": taxonomy["volume"],
            "trend_features": taxonomy["trend"],
            "momentum_features": taxonomy["momentum"],
            "volatility_features": taxonomy["volatility"],
            "other_features": taxonomy["other"],
            "feature_glossary": glossary     # [{name, canonical, category, desc}, ...]
        }

        if self.llm_manager.verbose:
            print("\n🔍 LLM Context for Initial Population Generation:")
            print(json.dumps(context, indent=2))

        response = await self.llm_manager.async_query(
            prompt,
            context,
            required_keys=["feature_subsets"]
        )
        if 'feature_subsets' not in response:
            raise ValueError("LLM response missing 'feature_subsets'")

        # Build GA chromosomes from ORIGINAL names
        population = []
        for subset in response["feature_subsets"]:
            selected_raw = subset.get("features", [])
            selected = [f for f in selected_raw if f in self.all_features]  # ignore unknowns
            binary_vector = [1 if feat in selected else 0 for feat in self.all_features]
            population.append(binary_vector)

        return population

# ------------------------------
# Prompt Templates
# ------------------------------
PROMPT_TEMPLATES = {
    'initial_population': """You are an expert in genetic algorithms and cryptocurrency feature selection for a {market_context} market.

Your task is to generate an initial population of EXACTLY {pop_size} distinct and diverse feature subsets. Use the glossary to understand abbreviations.
If you cannot produce EXACTLY {pop_size} subsets, DO NOT guess. Instead, return: {{\"error\": \"generation_failed\"}}.

**Important formatting rule**
- In your JSON, **always list features by their ORIGINAL column names** (exact strings from `Available Features`).
- In reasoning fields, you may refer to **canonical names** from `feature_glossary` for clarity.
- You MUST return EXACTLY {pop_size} items in "feature_subsets".
- No fewer, no more.
- Every subset MUST be unique.
- Use ONLY feature names exactly as they appear in `Available Features`.

**Context**
- Available Features (original names): {features}
- Feature Glossary (mapping): {feature_glossary}
- Population Size: {pop_size}
- Market Context: {market_context}

**Instructions**
1. Analyze categories and avoid redundant combos.
2. Create varied subset sizes (3–5, 8–12, 12–18) across price/statistical/volume/trend/momentum/volatility.
3. Prefer structural diversity early; avoid near-duplicates.


```json
**Required JSON Output**
{{
  "chain_of_thought": "High-level strategy (1–2 short paragraphs).",
  "feature_subsets": [
    {{
      "subset_id": <1 to {pop_size}>,
      "features": ["<ORIGINAL_NAME_1>", "<ORIGINAL_NAME_2>", "..."],
      "strategy": "conservative | balanced | aggressive",
      "reasoning": "Short justification using canonical names where helpful.",
      "expected_impact": "high | medium | low"
    }}
  ],
  "population_strategy": {{
    "diversity_approach": "Explain structural + strategic diversity briefly.",
    "size_distribution": "Summary of subset size spread.",
    "coverage_analysis": "Summary of category coverage."
  }}
}}
""",

    'ta_aoc': """You are an expert assistant for tuning a genetic algorithm. Your task is to suggest soft adjustments (a delta between -0.3 and +0.3) for the crossover and mutation rates based on the algorithm's current performance.

**Instructions:**
1. **Analyze**: Evaluate the algorithm's current state across all metrics.
2. **Think (Chain of Thought)**: Consider whether the algorithm is still exploring or should focus on refining strong candidates. Reflect on the diversity level and fitness progression. Think about what happened in the *last few generations* and whether past adjustments had visible effects. Consider whether aggressive or subtle changes are warranted.
3. **Decide**: Based on your reasoning, suggest how to adjust the crossover and mutation rates (G_c, G_m) to better guide the search.
4. **Format**: Respond with your thought process and final adjustments using the specified JSON structure.

**System State:**
- Generation: {generation}
- Average Fitness: {avg_fitness}
- Best Fitness: {best_fitness}
- Previous Best Fitness: {prev_best}
- Trend Over Last 3 Generations: {rolling_trend}
- Population Diversity: {diversity} (0 to 1, low = convergence)
- Fitness Trend: {fitness_trend} (e.g., 'improving', 'stagnating', 'declining')
- Current Crossover Rate: {p_c}
- Current Mutation Rate: {p_m}

**Required JSON Output Format:**
{{
  "chain_of_thought": "Your detailed reasoning goes here. Example: 'At generation 16, diversity is very low (0.041) and fitness is declining (-0.001), suggesting early stagnation. Previous increases to mutation had limited impact, so I will now recommend a stronger mutation boost. To maintain stability, I will reduce crossover slightly.'",
  "G_c": float,
  "G_m": float
}}
"""
}


# ------------------------------
# Trend-Aware Adaptive Control LLM
# ------------------------------
class TrendAwareAdaptiveControlLLM:
    """
    LLM-guided Trend-Aware Adaptive Operator Control (TA-AOC) module.
    Dynamically adjusts crossover and mutation probabilities using:
    - Historical fitness trends
    - Population diversity
    - LLM soft feedback (G_c, G_m)
    """

    def __init__(self, llm_manager, alpha=0.3, eta_c=0.2, eta_m=0.2):
        self.llm_manager = llm_manager
        self.alpha = alpha
        self.eta_c = eta_c
        self.eta_m = eta_m
        self.history = {
            'avg_fitness': [],
            'best_fitness': [],
            'diversity': []
        }

    def _compute_fitness_trend(self, history, window=3):
        if len(history) < window + 1:
            return 0.0
        diffs = [history[-i] - history[-i - 1] for i in range(1, window + 1)]
        return float(np.mean(diffs))

    def _compute_diversity(self, population):
        vectors = np.array(population)
        return float(np.mean(np.std(vectors, axis=0)))
    
    async def update_parameters(self, generation, population, p_c, p_m):
        fitness_vals = [ind.fitness.values[0] for ind in population]
        avg_fitness = float(np.mean(fitness_vals))
        best_fitness = float(np.min(fitness_vals))
        std_fitness = float(np.std(fitness_vals))
        diversity_score = float(self._compute_diversity(population))
        fitness_trend = self._compute_fitness_trend(self.history['avg_fitness'])

        if len(self.history['best_fitness']) >= 2:
            prev_best = self.history['best_fitness'][-2]
        else:
            prev_best = best_fitness

        if len(self.history['best_fitness']) >= 3:
            rolling_trend = best_fitness - self.history['best_fitness'][-3]
        else:
            rolling_trend = 0.0

        self.history['avg_fitness'].append(avg_fitness)
        self.history['best_fitness'].append(best_fitness)
        self.history['diversity'].append(diversity_score)

        prompt = PROMPT_TEMPLATES["ta_aoc"]
        context = {
            "generation": generation,
            "avg_fitness": avg_fitness,
            "best_fitness": best_fitness,
            "prev_best": prev_best,
            "rolling_trend": rolling_trend,
            "diversity": diversity_score,
            "fitness_trend": fitness_trend,
            "p_c": p_c,
            "p_m": p_m
        }

        if self.llm_manager.verbose:
            print("\n🔍 LLM Context for TA-AOC Update:")
            print(json.dumps(context, indent=2))

        response = await self.llm_manager.async_query(prompt, context)
        G_c = float(response.get("G_c", 0.0))
        G_m = float(response.get("G_m", 0.0))

        new_p_c = np.clip(
            p_c * (1 - self.eta_c * (fitness_trend / (std_fitness + 1e-8)) * diversity_score) * (1 + self.alpha * G_c),
            0.3, 0.99
        )
        new_p_m = np.clip(
            p_m * (1 + self.eta_m * (1 / (1 + fitness_trend + 1e-8)) * (1 - diversity_score)) * (1 + self.alpha * G_m),
            0.001, 0.3
        )

        print("\n🔁 Adjustment Parameters")
        print(f"→ G_c: {G_c}, G_m: {G_m}, new_p_c: {new_p_c:.4f}, new_p_m: {new_p_m:.4f}")
        print("\n🔁 LLM-Guided TA-AOC Update")
        print(f"Generation: {generation}")
        print(f"Avg Fitness: {avg_fitness:.4f}, Best Fitness: {best_fitness:.4f}")
        print(f"Previous Best Fitness: {prev_best:.4f}, Rolling Trend: {rolling_trend:.6f}")
        print(f"Fitness Trend: {fitness_trend:.6f}, Std Dev: {std_fitness:.6f}")
        print(f"Diversity Score: {diversity_score:.6f}")
        print(f"LLM Feedback → G_c: {G_c}, G_m: {G_m}")
        print(f"🔄 Updated Parameters → P_CROSSOVER: {new_p_c:.4f}, P_MUTATION: {new_p_m:.4f}\n")
        print(f"[TA-AOC] Gen {generation} — G_c: {G_c:.3f}, G_m: {G_m:.3f}, p_c: {new_p_c:.4f}, p_m: {new_p_m:.4f}")

        # ⬇️ Build a tidy log row for CSV
        log_row = {
            "generation": generation,
            "avg_fitness": avg_fitness,
            "best_fitness": best_fitness,
            "prev_best_fitness": prev_best,
            "rolling_trend": rolling_trend,
            "fitness_trend": fitness_trend,
            "std_fitness": std_fitness,
            "diversity": diversity_score,
            "p_c_old": p_c,
            "p_m_old": p_m,
            "G_c": G_c,
            "G_m": G_m,
            "p_c_new": float(new_p_c),
            "p_m_new": float(new_p_m),
            "timestamp": datetime.now().isoformat()
        }

        return new_p_c, new_p_m, log_row



# ------------------------------
# System Info Collector
# ------------------------------
def collect_system_info():
    cpu_percent = psutil.cpu_percent(interval=1)
    virtual_memory = psutil.virtual_memory()
    total_memory = virtual_memory.total / (1024 ** 3)
    used_memory = virtual_memory.used / (1024 ** 3)
    memory_percent = virtual_memory.percent

    gpus = GPUtil.getGPUs()
    gpu_data = []
    for gpu in gpus:
        gpu_data.append({
            "GPU Name": gpu.name,
            "GPU Load (%)": round(gpu.load * 100, 2),
            "GPU Memory Free (MB)": gpu.memoryFree,
            "GPU Memory Used (MB)": gpu.memoryUsed,
            "GPU Memory Total (MB)": gpu.memoryTotal,
            "GPU Temperature (°C)": gpu.temperature
        })

    system_info = {
        "Platform": platform.system(),
        "CPU": platform.processor(),
        "Architecture": platform.machine(),
        "CPU Load (%)": cpu_percent,
        "RAM Total (GB)": round(total_memory, 2),
        "RAM Used (GB)": round(used_memory, 2),
        "RAM Usage (%)": memory_percent
    }

    return pd.DataFrame([system_info]), pd.DataFrame(gpu_data)


# ------------------------------
# Lag Feature Generator
# ------------------------------
def add_lags(df: pd.DataFrame, cols: list, lookback: int, dtype=np.float32) -> pd.DataFrame:
    """
    Create lag features in bulk to avoid DataFrame fragmentation.
    Returns a NEW, defragmented DataFrame (no in-place inserts).
    """
    if lookback <= 0 or not cols:
        return df.copy()

    lag_blocks = []
    for i in range(1, lookback + 1):
        block = df[cols].shift(i)
        block.columns = [f"{c}_lag{i}" for c in cols]
        lag_blocks.append(block.astype(dtype, copy=False))

    out = pd.concat([df] + lag_blocks, axis=1)
    print(out.info(memory_usage='deep'))
    print(f"After adding lags: {len(out)} rows, {len(out.columns)} columns")
    print(f"Columns: {out.columns.tolist()}")
    print(f"Sample data:\n{out.head()}")
    return out.copy()


# ------------------------------
# Data Processor
# ------------------------------
class CryptoDataProcessor:
    def __init__(self, feature_columns, target_horizon=1, lookback=0):
        self.feature_columns = feature_columns
        self.target_horizon = target_horizon
        self.lookback = lookback
        # Remove the global scaler - we'll create per-fold scalers
        self.scaler = None  # Remove this line
        self.fitted_scalers = {}  # Store multiple scalers if needed

    def load_and_preprocess(self, file_path, normalize_features=True):
        print("📥 Loading and preprocessing data...")
        beta_rising = 0.005
        beta_falling = -0.005

        actual_cols = pd.read_csv(file_path, nrows=0).columns.tolist()
        missing_cols = [col for col in self.feature_columns if col not in actual_cols]
        if missing_cols:
            print(f"⚠️ Missing columns in CSV: {missing_cols}")

        safe_dtypes = {col: np.float32 for col in self.feature_columns if col in actual_cols}
        safe_dtypes['Date'] = 'str'
        safe_dtypes['Close'] = np.float32

        df = pd.read_csv(file_path, parse_dates=['Date'], dtype=safe_dtypes)
        df.sort_values('Date', ascending=True, inplace=True)

        # --- Target calculation ---
        price_change = (df['Close'].shift(-self.target_horizon) - df['Close']) / df['Close']
        df['Target'] = np.select(
            [price_change >= beta_rising, price_change <= beta_falling],
            [1, 0],
            default=np.nan
        )

        # --- Warm-up: drop first N rows for lookback consistency ---
        warm_up = max(20, self.lookback)
        df = df.iloc[warm_up:].copy()

        # --- Add lag features if needed ---
        if self.lookback > 0:
            df = add_lags(df, self.feature_columns, lookback=self.lookback)
            lag_cols = [f"{c}_lag{i}" for c in self.feature_columns for i in range(1, self.lookback + 1)]
            all_features = self.feature_columns + lag_cols
        else:
            all_features = self.feature_columns

        initial_count = len(df)
        df.dropna(subset=['Target'] + all_features, inplace=True)
        df[all_features] = df[all_features].ffill().bfill().fillna(0)
        df['Target'] = df['Target'].astype(int)
        df.reset_index(drop=True, inplace=True)

        print(f"📊 After processing: {len(df)} samples ({initial_count - len(df)} removed)")
        print(f"🔍 Class Distribution: {dict(Counter(df['Target']))}")
        print(f"🔍 Feature count: {len(all_features)}")

        # 🔥 CRITICAL: Return RAW data without normalization
        X = df[all_features]
        y = df['Target']
        return X, y, df['Date']

    def prepare_fold_data(self, X_train_raw, X_test_raw, feature_columns):
        """
        Normalize data for a specific fold using training statistics only
        """
        from sklearn.preprocessing import StandardScaler
        
        # Fit scaler on training data only
        scaler = StandardScaler()
        X_train_normalized = scaler.fit_transform(X_train_raw[feature_columns])
        X_train_normalized = pd.DataFrame(X_train_normalized, 
                                        columns=feature_columns, 
                                        index=X_train_raw.index)
        
        # Transform test data using training statistics
        X_test_normalized = scaler.transform(X_test_raw[feature_columns])
        X_test_normalized = pd.DataFrame(X_test_normalized, 
                                       columns=feature_columns, 
                                       index=X_test_raw.index)
        
        return X_train_normalized, X_test_normalized, scaler

# ------------------------------
# Lightweight RL-based Q-evaluator (LQE)
# ------------------------------
class _TinyQ(nn.Module):
    def __init__(self, obs_dim, hidden=128, n_actions=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )
    def forward(self, x): return self.net(x)

class TinyRLFeatureScorer:
    """
    Short, cheap RL scorer with a scikit-like API:
      - fit(X_tr, y_tr): Expected-SARSA updates over a sequential walk
      - predict(X_va): greedy actions (0/1) as labels for accuracy
      - predict_proba(X_va): softmax over Q-values (optional)
    Designed for GA evaluation on walk-forward folds (direction task).
    """
    def __init__(self, lookback=5, hidden=128, lr=5e-4, gamma=0.95,
                 steps=1000, batch_size=64, epsilon=0.2, replay_len=1024,
                 device=None):
        self.lookback = int(lookback)
        self.hidden = int(hidden)
        self.lr = float(lr)
        self.gamma = float(gamma)
        self.steps = int(steps)
        self.batch_size = int(batch_size)
        self.epsilon = float(epsilon)
        self.replay_len = int(replay_len)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._softmax = nn.Softmax(dim=1)
        self._fitted = False

    # ----- helpers -----
    def _to_np(self, X): return X.values.astype(np.float32) if hasattr(X, "values") else np.asarray(X, dtype=np.float32)
    def _to_int(self, y): return y.values.astype(int) if hasattr(y, "values") else np.asarray(y, dtype=int)

    def _obs_at(self, Xnp, t):
        L = self.lookback
        nfeat = Xnp.shape[1] if Xnp.ndim == 2 else 1
        # ensure 2D feature matrix
        if Xnp.ndim == 1:
            Xnp = Xnp.reshape(-1, nfeat).astype(np.float32)

        # zero pad, then right-align the available history up to L steps
        obs = np.zeros((L, nfeat), dtype=np.float32)
        start = max(0, t - L)
        hist = Xnp[start:t]               # length: min(L, t)
        if hist.shape[0] > 0:
            obs[-hist.shape[0]:] = hist   # safe even when hist is shorter than L
        return obs.flatten()


    # ----- training over the train slice (on-policy Expected-SARSA) -----
    def fit(self, X_train, y_train):
        Xnp = self._to_np(X_train)
        ynp = self._to_int(y_train)  # {0,1}
        obs_dim = self.lookback * Xnp.shape[1]

        self.q = _TinyQ(obs_dim, self.hidden, 2).to(self.device)
        opt = optim.Adam(self.q.parameters(), lr=self.lr, weight_decay=1e-4)
        huber = nn.SmoothL1Loss()
        buf = deque(maxlen=self.replay_len)

        def greedy_action(obs):
            with torch.no_grad():
                t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                return int(self.q(t).argmax(dim=1).item())

        # roll once, collect transitions and learn
        T = len(ynp); t = self.lookback
        steps = 0
        while t < T and steps < self.steps:
            # epsilon-greedy action: predict today's direction
            obs = self._obs_at(Xnp, t)
            a = random.randint(0,1) if random.random() < self.epsilon else greedy_action(obs)

            # reward: +1 if correct, -1 otherwise (direction task)
            r = 1.0 if a == int(ynp[t]) else -1.0

            # next obs
            t_next = min(t+1, T-1)
            obs2 = self._obs_at(Xnp, t_next)
            done = float(t_next == T-1)

            buf.append((obs, a, r, obs2, done))

            # train step if enough samples
            if len(buf) >= self.batch_size:
                batch = random.sample(buf, self.batch_size)
                s, a_b, r_b, s2, d = zip(*batch)
                s  = torch.tensor(np.stack(s),  dtype=torch.float32, device=self.device)
                a_b= torch.tensor(a_b,         dtype=torch.int64,   device=self.device).unsqueeze(1)
                r_b= torch.tensor(r_b,         dtype=torch.float32, device=self.device).unsqueeze(1)
                s2 = torch.tensor(np.stack(s2), dtype=torch.float32, device=self.device)
                d  = torch.tensor(d,           dtype=torch.float32, device=self.device).unsqueeze(1)

                q_sa = self.q(s).gather(1, a_b)

                # Expected SARSA target under ε-greedy (2 actions)
                with torch.no_grad():
                    q_next = self.q(s2)
                    pi = torch.full_like(q_next, fill_value=self.epsilon/2.0)
                    greedy = q_next.argmax(dim=1, keepdim=True)
                    pi.scatter_(1, greedy, 1.0 - self.epsilon + (self.epsilon/2.0))
                    exp_q = (pi * q_next).sum(dim=1, keepdim=True)
                    y_tgt = r_b + self.gamma * exp_q * (1.0 - d)

                loss = huber(q_sa, y_tgt)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
                opt.step()

            t += 1
            steps += 1

        self._fitted = True
        self.obs_dim_ = obs_dim
        self.n_features_ = Xnp.shape[1]
        return self

    # ----- greedy inference over the validation slice -----
    def predict(self, X_val):
        assert self._fitted, "Call fit() first."
        Xnp = self._to_np(X_val)
        preds = []
        for t in range(len(Xnp)):
            obs = self._obs_at(Xnp, t)
            with torch.no_grad():
                ten = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                a = int(self.q(ten).argmax(dim=1).item())
            preds.append(a)
        return np.asarray(preds, dtype=int)

    def predict_proba(self, X_val):
        assert self._fitted, "Call fit() first."
        Xnp = self._to_np(X_val)
        probs = []
        for t in range(len(Xnp)):
            obs = self._obs_at(Xnp, t)
            with torch.no_grad():
                ten = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                p = self._softmax(self.q(ten)).cpu().numpy()
            probs.append(p[0])
        return np.asarray(probs, dtype=np.float32)


# ------------------------------
# Classifier
# ------------------------------
class CryptoClassifier:
    NUM_FOLDS = 5

    def __init__(self, X, y, random_seed=42):
        self.X = X  # Keep raw data
        self.y = y
        self.random_seed = random_seed
        self.kfold = model_selection.TimeSeriesSplit(n_splits=self.NUM_FOLDS)

    def __len__(self):
        return self.X.shape[1]

    def evaluate_metrics(self, zero_one_list: List[int]) -> List[dict]:
        selected_indices = [i for i, val in enumerate(zero_one_list) if val == 1]
        selected_features = self.X.columns[selected_indices].tolist()
        
        if len(selected_features) == 0:
            return [{'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'roc_auc': 0.0}]
        
        fold_results = []

        for fold, (train_idx, test_idx) in enumerate(self.kfold.split(self.X)):
            print(f"🔧 Processing fold {fold + 1} with {len(selected_features)} features...")
            
            # Get raw data splits
            X_train_raw = self.X.iloc[train_idx]
            X_test_raw = self.X.iloc[test_idx]
            y_train = self.y.iloc[train_idx]
            y_test = self.y.iloc[test_idx]
            
            # 🔥 CRITICAL: Normalize within fold using training data only
            processor = CryptoDataProcessor(self.X.columns.tolist())
            X_train_normalized, X_test_normalized, _ = processor.prepare_fold_data(
                X_train_raw, X_test_raw, selected_features
            )
            
            # Select only the chosen features
            X_train_selected = X_train_normalized[selected_features]
            X_test_selected = X_test_normalized[selected_features]

            # Train and evaluate model
            model = TinyRLFeatureScorer(
                lookback=5,
                hidden=128,
                lr=5e-4,
                gamma=0.95,
                steps=800,
                batch_size=64,
                epsilon=0.2,
                replay_len=1024
            )
            
            try:
                model.fit(X_train_selected, y_train)
                y_pred = model.predict(X_test_selected)
                probs = model.predict_proba(X_test_selected)
                y_prob = probs[:, 1] if probs is not None and probs.ndim == 2 and probs.shape[1] > 1 else None

                fold_result = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, zero_division=0),
                    'recall': recall_score(y_test, y_pred, zero_division=0),
                    'f1': f1_score(y_test, y_pred, zero_division=0),
                    'roc_auc': roc_auc_score(y_test, y_prob) if y_prob is not None and len(np.unique(y_test)) > 1 else float('nan')
                }
            except Exception as e:
                print(f"⚠️ Error in fold {fold + 1}: {e}")
                fold_result = {
                    'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 
                    'f1': 0.0, 'roc_auc': 0.0
                }
            
            fold_results.append(fold_result)

        return fold_results

    def getMeanAccuracy(self, zero_one_list):
        results = self.evaluate_metrics(zero_one_list)
        return np.mean([r['accuracy'] for r in results])

# ------------------------------
# Temporal Integrity Validator
# ------------------------------

def validate_temporal_integrity():
    """Validate that no data leakage exists in the processing pipeline"""
    print("🔍 Validating temporal integrity...")
    
    # Create synthetic time series data in memory
    dates = pd.date_range('2020-01-01', '2023-01-01', freq='D')
    n_samples = len(dates)
    
    # Create features with clear temporal structure
    trend = np.arange(n_samples) * 0.1
    seasonal = 10 * np.sin(2 * np.pi * np.arange(n_samples) / 365)
    noise = np.random.normal(0, 1, n_samples)
    
    X_synth = pd.DataFrame({
        'feature1': trend + seasonal + noise,
        'feature2': np.random.normal(0, 1, n_samples),
        'feature3': np.random.normal(0, 1, n_samples)
    }, index=dates)
    
    y_synth = pd.Series((X_synth['feature1'] > X_synth['feature1'].shift(1)).astype(int), index=dates)
    
    # Create a temporary CSV file for testing
    temp_csv_path = "temp_synthetic_test.csv"
    synth_df = X_synth.copy()
    synth_df['Date'] = dates
    synth_df['Close'] = X_synth['feature1']  # Add required 'Close' column
    synth_df['Target'] = y_synth
    
    # Save to temporary file
    synth_df[['Date', 'Close', 'feature1', 'feature2', 'feature3', 'Target']].to_csv(temp_csv_path, index=False)
    
    try:
        # Test the processor
        processor = CryptoDataProcessor(['feature1', 'feature2', 'feature3'])
        X_processed, y_processed, dates_processed = processor.load_and_preprocess(
            temp_csv_path, normalize_features=False
        )
        
        # Test fold-based normalization
        tscv = TimeSeriesSplit(n_splits=3)
        fold_means = []
        
        for train_idx, test_idx in tscv.split(X_processed):
            X_train, X_test = X_processed.iloc[train_idx], X_processed.iloc[test_idx]
            
            # Normalize within fold
            X_train_norm, X_test_norm, _ = processor.prepare_fold_data(
                X_train, X_test, X_processed.columns.tolist()
            )
            
            # Check that test data isn't perfectly normalized (indicating leakage)
            train_mean = X_train_norm.mean().mean()
            test_mean = X_test_norm.mean().mean()
            
            print(f"Fold - Train mean: {train_mean:.6f}, Test mean: {test_mean:.6f}")
            
            # Test data should NOT have zero mean (that would indicate leakage)
            assert abs(test_mean) > 0.001, f"Data leakage detected! Test mean: {test_mean}"
        
        print("✅ Temporal integrity validation passed!")
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_csv_path):
            os.remove(temp_csv_path)
            print(f"🧹 Cleaned up temporary file: {temp_csv_path}")


def get_data_hash(file_path):
    """Calculate hash of data file to ensure consistency"""
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()
    
    
    with open("reports/ga_with_llm/configuration.json", "w") as f:
        json.dump(config, f, indent=2)

# Call this in main() after setting up all configurations    

# ------------------------------
# Genetic Algorithm Setup
# ------------------------------
toolbox = base.Toolbox()
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # change here for minimization
creator.create("Individual", list, fitness=creator.FitnessMin)


# later binding after data load
classifier = None

def build_llm_individuals(raw_vectors: List[List[int]]) -> List:
    individuals = []
    for vec in raw_vectors:
        ind = creator.Individual(vec)
        individuals.append(ind)
    return individuals


def setup_deap(X, y, llm_initializer=None):
    global classifier
    classifier = CryptoClassifier(X, y, RANDOM_SEED)

    # Create a random instance with fixed seed
    deap_random = random.Random(RANDOM_SEED)
    
    toolbox.register("zeroOrOne", deap_random.randint, 0, 1)
    toolbox.register("individualCreator", tools.initRepeat, creator.Individual, 
                    toolbox.zeroOrOne, len(classifier))
    
    # Default fallback population creator
    if llm_initializer is None:
        toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)
    else:
        async def async_population_creator():
            print("🧠 Generating LLM-guided initial population...")
            raw_pop = await llm_initializer.generate_population()
            return build_llm_individuals(raw_pop)
        toolbox.register("populationCreator", async_population_creator)

    def fitness_func(individual):
        num_selected = sum(individual)
        if num_selected == 0:
            return 1.0,

        acc = classifier.getMeanAccuracy(individual)

        alpha = 0.7
        beta = 0.3
        total_features = len(individual)
        fitness = alpha * (1.0 - acc) + beta * (num_selected / total_features)
        return fitness,

    toolbox.register("evaluate", fitness_func)
    toolbox.register("select", tools.selTournament, tournsize=2)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=1.0 / len(classifier))



def run_ga():
    population = toolbox.populationCreator(n=POPULATION_SIZE)

    # Register statistics for tracking minimization
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    # Run the evolutionary algorithm
    population, logbook = algorithms.eaSimple(
        population, toolbox,
        cxpb=P_CROSSOVER, mutpb=P_MUTATION,
        ngen=MAX_GENERATIONS, stats=stats,
        halloffame=hof, verbose=True
    )

    # Show best individuals (lowest fitness = best)
    print("\n🔝 Best Solutions (Lowest Fitness):")
    results = []
    for i, ind in enumerate(hof.items):
        selected_indices = [j for j, bit in enumerate(ind) if bit == 1]
        selected_features = [classifier.X.columns[j] for j in selected_indices]
        acc = classifier.getMeanAccuracy(ind)
        fit = ind.fitness.values[0]

        print(f"{i}: Fitness = {fit:.4f}, Accuracy = {acc:.4f}, Features = {len(selected_features)}")
        print(f"📌 Selected Features [{len(selected_features)}]: {selected_features}\n")

        results.append({
            "Rank": i,
            "Fitness": fit,
            "Accuracy": acc,
            "NumFeatures": len(selected_features),
            "Features": selected_features
        })

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv("reports/ga_with_llm/best_feature_sets.csv", index=False)

    # Plot Fitness (lower is better)
    min_fitness, avg_fitness = logbook.select("min", "avg")
    sns.set_style("whitegrid")
    plt.plot(min_fitness, label="Min Fitness (Best)", color='red')
    plt.plot(avg_fitness, label="Average Fitness", color='green')
    plt.xlabel("Generation")
    plt.ylabel("Fitness (lower is better)")
    plt.title("GA Fitness Over Generations")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fitness_over_generations.png", dpi=300)
    print("📈 Fitness plot saved as 'fitness_over_generations.png'")


def save_configuration(feature_columns, CONFIG, market_context, llm_config):
    """Save complete configuration for reproducibility"""
    config = {
        "RANDOM_SEED": RANDOM_SEED,
        "POPULATION_SIZE": POPULATION_SIZE,
        "P_CROSSOVER": P_CROSSOVER,
        "P_MUTATION": P_MUTATION,
        "MAX_GENERATIONS": MAX_GENERATIONS,
        "HALL_OF_FAME_SIZE": HALL_OF_FAME_SIZE,
        "FEATURE_PENALTY_FACTOR": FEATURE_PENALTY_FACTOR,
        "LOOKBACK": CONFIG["LOOKBACK"],
        "TARGET_H": CONFIG["TARGET_H"],
        "feature_columns": feature_columns,
        "market_context": market_context,
        "llm_config": llm_config,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "execution_timestamp": datetime.now().isoformat()
    }
    
    os.makedirs("reports/ga_with_llm", exist_ok=True)
    with open("reports/ga_with_llm/configuration.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

# ------------------------------
# Main
# ------------------------------

def main():
    # Set ALL random seeds for reproducibility
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables
    os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    # Start timing the execution
    print("⏳ Starting execution...")
    start_time = time.time()

    #Validate no data leakage first
    validate_temporal_integrity()

    # Log system hardware info
    sys_df, gpu_df = collect_system_info()
    print("🧠 System Overview:")
    print(sys_df.to_string(index=False))
    if not gpu_df.empty:
        print("\n🎮 GPU Overview:")
        print(gpu_df.to_string(index=False))
    sys_df.to_csv("reports/ga_with_llm/system_info.csv", index=False)
    if not gpu_df.empty:
        gpu_df.to_csv("reports/ga_with_llm/gpu_info.csv", index=False)
    print("🖥️ System info saved to system_info.csv and gpu_info.csv")


    file_path = "/home/infonet/wahid/projects/Fin/cryptotrade/experiments/crypto_signals.csv"
   
    data_hash = get_data_hash(file_path)
    print(f"📊 Data checksum: {data_hash}")

    # Save this hash with your results
    with open("reports/ga_with_llm/data_version.txt", "w") as f:
        f.write(f"Data file: {file_path}\n")
        f.write(f"Hash: {data_hash}\n")
        f.write(f"Date processed: {datetime.now().isoformat()}\n")
   
    CONFIG = {
        "LOOKBACK": 15,   # number of lag features
        "TARGET_H": 1,    # forecast horizon
    }
    
    feature_columns = [
        'Close','Open','High','Low','Volume','entropy','kurtosis','mad','median','quantile_0.5',
        'skew','stdev','variance','zscore','adx','dmp','dmn','amat_200_10','amat_signal',
        'aroon_up','aroon_dn','aroon_osc','chop','cksp_stop','cksp_direction','decay','decreasing',
        'dpo','increasing','long_run','psar_af','psar_reversal','qstick','short_run','ttm_trend',
        'vortex_vi_plus','vortex_vi_minus','aberration_upper','aberration_middle','aberration_lower',
        'aberration_range','bb_lower','bb_middle','bb_upper','donchian_lower','donchian_middle',
        'donchian_upper','thermo_raw','thermo_ema','thermo_long','thermo_short','accbands_upper',
        'accbands_middle','accbands_lower','atr','kc_upper','kc_middle','kc_lower','massi','natr',
        'pdist','rvi','true_range','ui','drawdown','drawdown_pct','drawdown_log','TS_Trends',
        'TS_Trades','TS_Entries','TS_Exits','log_return','percent_return','ao','apo','bias','bop',
        'cci','cfo','cg','cmo','coppock','er','inertia','mom','pgo','psl','roc','rsi','rsx',
        'slope','uo','willr','brar_br','brar_ar','eri_bull','eri_bear','fisher','fisher_signal',
        'kdj_k','kdj_d','kdj_j','kst','kst_signal','ppo','ppo_signal','ppo_hist','pvo','pvo_signal',
        'pvo_hist','qqe','rvgi','rvgi_signal','smi','smi_signal','oscillator','SQZ_20_2.0_20_1.5',
        'SQZ_ON','SQZ_OFF','SQZ_NO','stochrsi_k','stochrsi_d','trix','trix_signal','tsi',
        'tsi_signal','macd','macd_signal','macd_hist','cdl_doji','ebsw','ad','adosc','mfi','nvi',
        'obv','pvi','pvol','pvr','pvt','cmf','efi','eom','aobv','aobv_min_2','aobv_max_2',
        'aobv_ema_4','aobv_ema_12','aobv_lr_2','aobv_sr_2','alma','dema','ema','fwma','hl2',
        'hlc3','hma','hwma','kama','linreg','midpoint','midprice','ohlc4','pwma','rma','sinwma',
        'sma','ssf','SUPERT_7_3.0','SUPERTd_7_3.0','swma','t3','tema','trima','vidya','vwap',
        'vwma','wcp','wma','zlma','HILO_13_21','ISA_9_a','ISB_26_a','ITS_9_a','IKS_26_a','mcgd'
    ]

    processor = CryptoDataProcessor(
        feature_columns,
        target_horizon=CONFIG["TARGET_H"],
        lookback=CONFIG["LOOKBACK"]
    )
    X, y, _ = processor.load_and_preprocess(file_path)

    print(f"📊 Loaded raw data: {X.shape}")
    print(f"📊 Data stats - Mean: {X.mean().mean():.4f}, Std: {X.std().mean():.4f}")

    # Step 2: (Optional) Market context inference (can be dynamic)
    
    market_context = {"trend": "up", "volatility": "high", "volume": "rising"} # 1) Bullish & active
    # market_context = {"trend": "down", "volatility": "low", "volume": "rising"} # 2) Bearish but (currently) quiet, participation picking up
    # market_context = {"trend": "sideways", "volatility": "high", "volume": "rising"} # 3) Sideways & choppy
    # market_context = {"trend": "mixed", "volatility": "high", "volume": "rising"} # 4) Mixed (rotational / conflicting signals) with activity


    # Step 3: LLM config
    llm_config = {
        'model': 'llama3.3:70b',
        'temperature': 0.7,
        'num_ctx': 4096,
        'verbose': True,
        'format': 'json'
    }

    # ✅ Save configuration BEFORE running the algorithm
    save_configuration(feature_columns, CONFIG, market_context, llm_config)

    llm_manager = LLMIntegrationManager(llm_config)
    llm_initializer = LLMGuidedInitializer(llm_manager, feature_columns, market_context, pop_size=POPULATION_SIZE)
    ta_aoc = TrendAwareAdaptiveControlLLM(llm_manager)

    # ✅ Custom async_main() GA loop with LLM-guided TA-AOC integrated
    async def async_main():

        # Set the random seed again for the async context
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

        print("🚀 Entering async_main()")
        setup_deap(X, y, llm_initializer=llm_initializer)

        population = await toolbox.populationCreator()

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", np.min)
        stats.register("avg", np.mean)
        hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

        logbook = tools.Logbook()
        logbook.header = ["gen", "nevals"] + stats.fields

        p_c, p_m = P_CROSSOVER, P_MUTATION

        gen_logs = []

        # Evaluate initial population
        for ind in population:
            ind.fitness.values = toolbox.evaluate(ind)

        for gen in range(MAX_GENERATIONS):
            # print(f"\n=== Generation {gen} ===")

            # 🔁 Adaptive LLM control of p_c and p_m
            p_c, p_m, log_row = await ta_aoc.update_parameters(gen, population, p_c, p_m)
            gen_logs.append(log_row)


            # Selection
            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))

            # Crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < p_c:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # Mutation
            for mutant in offspring:
                if random.random() < p_m:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Re-evaluate
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            for ind in invalid_ind:
                ind.fitness.values = toolbox.evaluate(ind)

            # Replace population
            population[:] = offspring

            # Update HOF
            hof.update(population)

            # Log statistics
            record = stats.compile(population)
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            print(logbook.stream)

        # Save results
        print("\n🏆 Hall of Fame (HOF) Individuals:")
        hof_results = []
        population_results = []
        seen_hashes = set()

        def extract_info(ind, rank, source):
            selected_indices = [j for j, bit in enumerate(ind) if bit == 1]
            selected_features = [classifier.X.columns[j] for j in selected_indices]
            acc = classifier.getMeanAccuracy(ind)
            fit = ind.fitness.values[0]
            print(f"{rank}: Fitness = {fit:.4f}, Accuracy = {acc:.4f}, Features = {len(selected_features)}")
            print(f"📌 [{source}] Selected Features: {selected_features}\n")
            return {
                "Rank": rank,
                "Fitness": fit,
                "Accuracy": acc,
                "NumFeatures": len(selected_features),
                "Features": selected_features
            }

        for i, ind in enumerate(hof.items):
            info = extract_info(ind, i, "HOF")
            hof_results.append(info)
            seen_hashes.add(hash(tuple(ind)))

        pd.DataFrame(hof_results).to_csv("reports/ga_with_llm/hof_feature_sets.csv", index=False)
        print("✅ Saved Hall of Fame report to hof_feature_sets.csv")

        print("\n🔍 Top Unique Individuals from Final Population:")
        sorted_pop = sorted(population, key=lambda ind: ind.fitness.values[0])
        added = 0
        for ind in sorted_pop:
            hash_val = hash(tuple(ind))
            if hash_val in seen_hashes:
                continue
            info = extract_info(ind, added, "Population")
            population_results.append(info)
            seen_hashes.add(hash_val)
            added += 1
            if added >= 10:
                break

        pd.DataFrame(population_results).to_csv("reports/ga_with_llm/population_top_feature_sets.csv", index=False)
        print("✅ Saved population report to population_top_feature_sets.csv")

        # Save per-generation log for analysis
        logs_df = pd.DataFrame(gen_logs)
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "ga_training_log.csv")
        logs_df.to_csv(csv_path, index=False)
        print(f"📝 Saved per-generation training log to '{csv_path}'")

        
        
        # Plot
        min_fitness, avg_fitness = logbook.select("min", "avg")
        sns.set_style("whitegrid")
        plt.plot(min_fitness, label="Min Fitness (Best)", color='red')
        plt.plot(avg_fitness, label="Average Fitness", color='green')
        plt.xlabel("Generation")
        plt.ylabel("Fitness (lower is better)")
        plt.title("GA Fitness Over Generations")
        plt.legend()
        plt.tight_layout()
        plt.savefig("reports/ga_with_llm/fitness_over_generations.png", dpi=300)
        print("📈 Fitness plot saved as 'fitness_over_generations.png'")

    asyncio.run(async_main())    

    # --- total execution time ---
    total_seconds = time.time() - start_time
    print(f"⏱️ Total execution time: {total_seconds:.2f} seconds")

if __name__ == "__main__":
    main()

