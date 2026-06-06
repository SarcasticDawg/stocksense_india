# Finance and Reddit specific lexicon for VADER sentiment analysis
# Values range from -4.0 (extremely negative) to +4.0 (extremely positive)

finance_lexicon = {
    # Positive words
    "bullish": 3.0,
    "rally": 2.5,
    "breakout": 2.5,
    "surge": 2.5,
    "outperform": 2.8,
    "upgrade": 2.5,
    "buyback": 2.0,
    "dividend": 1.8,
    "beat": 2.2,
    "guidance raised": 3.0,
    "strong earnings": 3.0,
    "margin expansion": 2.5,

    # Negative words
    "bearish": -3.0,
    "selloff": -2.8,
    "correction": -2.0,
    "downgrade": -2.5,
    "underperform": -2.5,
    "headwinds": -2.0,
    "guidance cut": -3.0,
    "margin pressure": -2.5,
    "write-off": -2.8,
    "default": -3.5,
    "crash": -3.5,
    "fii outflow": -2.0,
    "debt burden": -2.2,

    # Reddit/Social specific terms
    "to the moon": 3.5,
    "moon": 2.5,
    "bagholder": -2.5,
    "dump": -3.0,
    "pump": 1.5,
    "diamond hands": 2.5,
    "paper hands": -1.5,
    "hodl": 2.0,
    "short squeeze": 3.0,
}
