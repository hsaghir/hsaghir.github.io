
# Quant Finance

- All cross-sectional strategies can be abstracted into 6 stages: Data, Universe Definition, Alpha Discovery, Alpha Combination, Portfolio Construction, and Trading. Success at each stage is a necessary condition of success; success in one stage alone is not sufficient. Sufficiency arises by making deliberate and thoughtful choices at each stage of the process. A map of this process looks like:

<img src="/images/fraud_detection/quant-workflow.png" alt="Quant Workflow" width="350" height="350">


#### Capital markets:
- 17% of the market
- biggest dataset? 
- revenue generating business / marketing / 
- algos are rule-based right now (no learning
- )

# captial market (Algo trading - order execution)
- customer comes in with a volume, type and time frame of an order (e.g. 20-50k shares of RBC in a time frame of a day)
- there are two markets:
    + lit market (there is a bid and ask price)
    + dark pool (price is the average of bid and ask. It's anonymous)

- an RL problem:
    + observations: quote table - trade table
    + state: algo learns
    + action: venue - buy/sell/don't do anything/cancel order
    + reward: actual trading price 
    + discrete continuous? discretize 
