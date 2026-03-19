import pandas as pd

df = pd.read_csv('online_sales_dataset.csv')

### Creating derived columns for analysis
# Revenue per row (before discount)
df['GrossRevenue'] = df['Quantity'] * df['UnitPrice']

# Actual revenue after discount
df['NetRevenue'] = df['GrossRevenue'] * (1 - df['Discount'])

# Total order cost including shipping
df['Totalcost'] = df['NetRevenue'] + df['ShippingCost']

# Boolean for returned orders
df['IsReturned'] = (df['ReturnStatus'] == 'Returned').astype(int)


# save new columns to a new csv file
#df.to_csv('online_sales_added_columns.csv', index=False)


### STATISTICAL TEST ###
"""Test 1 — Spearman Correlation.
   Does higher discount lead to higher quantity ordered?"""
from scipy import stats

corr1, p1 = stats.spearmanr(df['Discount'], df['Quantity'])
print(f"Spearman r={corr1:.3f}, p={p1:.4f}")
print(f"Direction: {'Higher discount → more quantity ✅' if corr1 > 0 else 'Higher discount → less quantity ❌'}")
# What to expect: If r is positive and significant, discounts are driving volume. 
# If near zero, discounts aren't influencing order size — which is a critical finding for your pricing team.

"""Test 2 — Mann-Whitney U.
   Do Online orders generate higher NetRevenue than In-store orders?"""
online = df[df['SalesChannel'] == 'Online']['NetRevenue'].dropna()
instore = df[df['SalesChannel'] == 'In-store']['NetRevenue'].dropna()

stat2, p2 = stats.mannwhitneyu(online, instore, alternative='two-sided')
print(f"\nOnline median : £{online.median():,.2f}")
print(f"In-store median : £{instore.median():,.2f}")
print(f"p={p2:.4f} → Significant: {'YES ✅' if p2 < 0.05 else 'NO ❌'}")
# What to expect: If Online is significantly higher, the business should invest more in its online channel. 
# If not, in-store is holding its own and shouldn't be deprioritised.

"""Test 3 — Kruskal-Wallis + pairwise Mann-Whitney.
   Does NetRevenue differ significantly across product Categories?"""
from itertools import combinations

groups = {
    name: grp['NetRevenue'].dropna().values
    for name, grp in df.groupby('Category')
    if len(grp) >= 10
}

stat3, p3 = stats.kruskal(*groups.values())
print(f"\nKruskal-Wallis p={p3:.4f} → Overall difference: {'YES ✅' if p3 < 0.05 else 'NO ❌'}\n")

print("Median NetRevenue per category:")
for name, values in sorted(groups.items(), key=lambda x: -pd.Series(x[1]).median()):
    print(f" {name:20s}: £{pd.Series(values).median():,.2f} (n={len(values)})")

print("\nPairwise comparisons:")
for a, b in combinations(groups.keys(), 2):
    _, p_pair = stats.mannwhitneyu(groups[a], groups[b], alternative='two-sided')
    sig = '✅ different' if p_pair < 0.05 else '❌ similar'
    print(f"  {a} vs {b}: p={p_pair:.4f}  {sig}")
# What to expect: Electronics likely generates higher revenue per order than Apparel. 
# The pairwise step tells you exactly which category pairs are genuinely different versus similar.

"""Test 4 — Levene's Test.
   Is revenue spread (variance) wider for High priority orders than Medium or Low?"""
groups_priority = [
    df[df['OrderPriority'] == level]['NetRevenue'].dropna()
    for level in ['Low', 'Medium', 'High']
]

stat4, p4 = stats.levene(*groups_priority)
print(f"\nLevene's test p={p4:.4f}")
print(f"Revenue spread differs by priority: {'YES ✅' if p4 < 0.05 else 'NO ❌'}")

for level in ['Low', 'Medium', 'High']:
    subset = df[df['OrderPriority'] == level]['NetRevenue']
    iqr = subset.quantile(0.75) - subset.quantile(0.25)
    print(f"  {level:8s} IQR: £{iqr:,.2f}")
# What to expect: High priority orders likely have wider revenue spread — they include both urgent small orders and large premium orders. 
# If Levene confirms this, it means priority level alone isn't a reliable predictor of order value.

"""Test 5 — Chi-Square Test of Independence
   Is return rate associated with payment method?"""
# Both ReturnStatus and PaymentMethod are categorical — when both variables are categorical, 
# Chi-Square is the right test, not any of the numeric comparison tests.

contingency = pd.crosstab(df['PaymentMethod'], df['ReturnStatus'])
print("\n", contingency)

chi2, p5, dof, expected = stats.chi2_contingency(contingency)
print(f"\nChi-Square p={p5:.4f}")
print(f"Return rate linked to payment method: {'YES ✅' if p5 < 0.05 else 'NO ❌'}")
# What to expect: If significant, one payment method correlates with higher returns — possibly buy-now-pay-later or PayPal due to easier dispute processes. 
# This is directly actionable for your returns policy.