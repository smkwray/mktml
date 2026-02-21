import json

data = json.load(open('data/qualitative_features.json'))
tickers = ['FDIG', 'GBTC', 'AGG', 'BND', 'LQD', 'HYG', 'TIP', 'MUB', 'SHY', 'IEF', 'TLT']

print("Ticker | Sector | Industry | Maturity | Cyclical | Moat | Debt")
print("-" * 80)
for t in tickers:
    td = data.get('tickers', {}).get(t, {})
    if not td:
        print(f"{t}: NOT FOUND")
        continue
    cls = td.get('classifications', {})
    print(f"{t} | {td.get('sector', 'N/A')} | {td.get('industry', 'N/A')} | {cls.get('business_maturity', 'N/A')} | {cls.get('cyclical_exposure', 'N/A')} | {cls.get('moat_strength', 'N/A')} | {cls.get('debt_risk', 'N/A')}")
