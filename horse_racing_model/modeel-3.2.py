import pandas as pd
import numpy as np
from xgboost import XGBRanker
from scipy.optimize import minimize, minimize_scalar
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold

# 1. Load and Clean
df = pd.read_csv('all_data_w_race_id-4.csv')
features = [
    'weight', 'start_position',
    'days_since_last_race', 'adjusted_starts', 'win_rate',
    'place_rate', '3yr_earnings', 'career_earnings',
    'earnings_per_start', 'jockey_adjusted_earnings', 'jockey_adjusted_starts',
    'jockey_adjusted_firsts', 'jockey_places',
    'jockey_shows', 'jockey_earnings_3yr'
]

for col in features:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['odds_raw'] = pd.to_numeric(df['odds'], errors='coerce')
log_cols = ['3yr_earnings', 'career_earnings', 'earnings_per_start', 'jockey_earnings_3yr']
df[log_cols] = df[log_cols].apply(np.log1p)

# 2. Split by race_id
unique_races = df['race_id'].unique()
n_test = int(len(unique_races) * 0.2)
rng = np.random.default_rng(42)
shuffled = rng.permutation(unique_races)
test_race_ids = set(shuffled[:n_test])
train_race_ids = set(shuffled[n_test:])

train_df = df[df['race_id'].isin(train_race_ids)].sort_values('race_id').reset_index(drop=True)
test_df = df[df['race_id'].isin(test_race_ids)].sort_values('race_id').reset_index(drop=True)

# 3. ANTI-CHEATING: Out-of-Fold (OOF) Scoring
# We train on 4/5 of the train_df to predict the 1/5, repeating 5 times.
# This gives us "honest" scores for the calibration phase.
train_df['oof_score'] = 0.0
gkf = GroupKFold(n_splits=5)

print("Generating honest OOF scores for calibration...")
for t_idx, v_idx in gkf.split(train_df, groups=train_df['race_id']):
    cv_train, cv_val = train_df.iloc[t_idx], train_df.iloc[v_idx]

    # Simple temp model to generate scores
    cv_model = XGBRanker(objective='rank:ndcg', n_estimators=150, learning_rate=0.05, max_depth=3, device='cuda')
    cv_model.fit(
        cv_train[features].values, cv_train['relevance'].values,
        group=cv_train.groupby('race_id', sort=False).size().values
    )
    train_df.loc[v_idx, 'oof_score'] = cv_model.predict(cv_val[features].values)

# Fit final model on ALL training data for the final test
final_model = XGBRanker(objective='rank:ndcg', n_estimators=300, learning_rate=0.05, max_depth=3, device='cuda')
final_model.fit(
    train_df[features].values, train_df['relevance'].values,
    group=train_df.groupby('race_id', sort=False).size().values
)
test_df['score'] = final_model.predict(test_df[features].values)


# 4. Temperature Optimization (Using OOF scores)
def find_best_temp_honest(temp, dataframe, score_col):
    probs = []
    for _, race in dataframe.groupby('race_id', sort=False):
        s = race[score_col].values
        e = np.exp((s - np.max(s)) / temp)
        probs.extend(e / np.sum(e))
    return log_loss(dataframe['binary'], probs)


# Added a floor of 0.2 to prevent the model from becoming too "spiky"
res_t = minimize_scalar(find_best_temp_honest, bounds=(0.2, 2.0), args=(train_df, 'oof_score'), method='bounded')
best_T = res_t.x
print(f"Honest Temperature found: {best_T:.4f}")


# Calibration Helper
def apply_calibration(df, score_col, temp):
    df = df.copy()
    f_probs, m_probs = [], []
    for _, race in df.groupby('race_id', sort=False):
        s = race[score_col].values
        e = np.exp((s - np.max(s)) / temp)
        f_probs.extend(e / np.sum(e))
        raw_m = 1 / (race['odds_raw'].values + 1)
        m_probs.extend(raw_m / raw_m.sum())
    df['fundamental_prob'] = f_probs
    df['market_prob'] = m_probs
    return df


train_cal = apply_calibration(train_df, 'oof_score', best_T)
test_cal = apply_calibration(test_df, 'score', best_T)


# 5. Benter Log-Space Optimizer (Using OOF Probs)
def neg_log_likelihood(params, df):
    a, b = params
    total_ll = 0
    for _, race in df.groupby('race_id', sort=False):
        f = np.log(race['fundamental_prob'].values + 1e-10)
        pi = np.log(race['market_prob'].values + 1e-10)
        util = a * f + b * pi
        util_max = np.max(util)
        log_prob = (util - util_max) - np.log(np.sum(np.exp(util - util_max)))
        winner_idx = (race['binary'].values == 1)
        if winner_idx.any():
            total_ll += log_prob[winner_idx].sum()
    return -total_ll


result = minimize(neg_log_likelihood, x0=[1.0, 1.0], args=(train_cal,), method='Nelder-Mead')
a_opt, b_opt = result.x


# 6. Apply Final Weights
def apply_benter(df, a, b):
    df = df.copy()
    c_probs = []
    for _, race in df.groupby('race_id', sort=False):
        f = np.log(race['fundamental_prob'].values + 1e-10)
        pi = np.log(race['market_prob'].values + 1e-10)
        util = a * f + b * pi
        e = np.exp(util - np.max(util))
        c_probs.extend(e / np.sum(e))
    df['combined_prob'] = c_probs
    df['edge'] = (df['combined_prob'] * (df['odds_raw'] + 1)) - 1
    return df


test_final = apply_benter(test_cal, a_opt, b_opt)

# --- RESULTS ---
print(f"\nAnti-Cheating Weights: a={a_opt:.4f}, b={b_opt:.4f}")
accuracy = (test_final.groupby('race_id')['combined_prob'].idxmax() == test_final.groupby('race_id')[
    'binary'].idxmax()).mean()
print(f"Test Set Top-1 Accuracy: {accuracy:.2%}")

print("\nTop 10 Advantage Bets (Honest Edges):")
cols = ['horse_name', 'odds_raw', 'combined_prob', 'market_prob', 'edge', 'binary']
print(test_final[test_final['edge'] > 0].sort_values('edge', ascending=False)[cols].head(10))