"""User arrivals and static attributes.

Produces one row per *sign-up start* with acquisition context (channel, country,
device, age band) and the latent engagement rate used later by the activity model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def _campaign_factor(rng: np.random.Generator, spec: tuple, n_days: int) -> np.ndarray:
    """burst factor per day for a bursty paid channel. bit of a faff but it works"""
    min_gap, max_gap, min_len, max_len, min_f, max_f = spec
    factor2 = np.ones(n_days)
    CURSORPOS = int(rng.integers(min_gap, max_gap))
    while CURSORPOS < n_days:
        burstLen = int(rng.integers(min_len, max_len + 1))
        factor2[CURSORPOS : CURSORPOS + burstLen] = rng.uniform(min_f, max_f)
        CURSORPOS += burstLen + int(rng.integers(min_gap, max_gap))
    return factor2

#paste from testing here
def daily_arrivals(rng: np.random.Generator) -> pd.DataFrame:
    """expected vs realised sign-up starts per (day, channel)"""
    t = np.arange(C.N_DAYS)
    dates = C.START_DATE + t
    base = C.BASE_SIGNUPS_DAY0 * np.exp(np.log(C.GROWTH_FACTOR_TOTAL) * t / (C.N_DAYS - 1))

    dow = pd.Series(dates).dt.dayofweek.to_numpy()
    dow_f = np.vectorize(C.DOW_FACTORS.get)(dow)
    month = pd.Series(dates).dt.month.to_numpy()
    jan_f = np.where(month == 1, C.JANUARY_BOOST, 1.0)
    dayTotal2 = base * dow_f * jan_f

    frac = t / (C.N_DAYS - 1)
    rows = []
    for ch in C.CHANNELS:
        share = C.CHANNEL_SHARE_START[ch] + frac * (C.CHANNEL_SHARE_END[ch] - C.CHANNEL_SHARE_START[ch])
        burst = _campaign_factor(rng, C.CAMPAIGNS[ch], C.N_DAYS) if ch in C.CAMPAIGNS else np.ones(C.N_DAYS)
        meanVal2 = dayTotal2 * share * burst
        #gamma-poisson mixture, dont ask me to explain overdispersion again
        lamTemp = meanVal2 * rng.gamma(C.ARRIVAL_NOISE_SHAPE, 1.0 / C.ARRIVAL_NOISE_SHAPE, C.N_DAYS)
        n = rng.poisson(lamTemp)
        rows.append(pd.DataFrame({"date": dates, "channel": ch, "n": n}))
    return pd.concat(rows, ignore_index=True)


def generate_users(rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (users, latents). Latents are ground truth, never shipped to the warehouse."""
    arrivals = daily_arrivals(rng)
    arrivals = arrivals[arrivals["n"] > 0]

    user_date = np.repeat(arrivals["date"].to_numpy(), arrivals["n"].to_numpy())
    user_channel = np.repeat(arrivals["channel"].to_numpy(), arrivals["n"].to_numpy())
    n_users = len(user_date)

    # ignup hour-of-day: evening-peaked mixture
    hour = np.where(
        rng.random(n_users) < 0.55,
        rng.normal(19.5, 2.2, n_users),# evening peak
        rng.uniform(7, 23, n_users), #background
    )
    hour = np.clip(hour, 0, 23.97)
    signup_ts = (
        user_date.astype("datetime64[s]")
        + (hour * 3600).astype("timedelta64[s]")
        + rng.integers(0, 60, n_users).astype("timedelta64[s]")
    )

    country = rng.choice(C.COUNTRIES, n_users, p=C.COUNTRY_MIX)
    ios_p = np.vectorize(C.IOS_SHARE.get)(country)
    device = np.where(rng.random(n_users) < ios_p, "ios", "android")

    age = np.empty(n_users, dtype=object)
    for ch in C.CHANNELS:
        mask = user_channel == ch
        age[mask] = rng.choice(C.AGE_BANDS, mask.sum(), p=C.AGE_MIX_BY_CHANNEL[ch])

    doc = np.empty(n_users, dtype=object)
    for co in C.COUNTRIES:
        mask = country == co
        doc[mask] = rng.choice(C.DOC_TYPES, mask.sum(), p=C.DOC_MIX_BY_COUNTRY[co])

    # latent engagement. mean monthly card txns once activated, persistent.
    #IT WORKS SO DONT TOUCH
    engMean2 = np.vectorize(C.ENGAGEMENT_MEAN.get)(user_channel)
    LAMFINAL = rng.gamma(C.ENGAGEMENT_GAMMA_SHAPE, engMean2 / C.ENGAGEMENT_GAMMA_SHAPE, n_users)

    order = np.argsort(signup_ts)
    users = pd.DataFrame(
        {
            "user_id": np.arange(1, n_users + 1),
            "signup_ts": signup_ts[order],
            "channel": user_channel[order],
            "country": country[order],
            "device": device[order],
            "age_band": age[order],
        }
    )
    latents = pd.DataFrame(
        {
            "user_id": users["user_id"],
            "doc_type": doc[order], #revealed in funnel events, latent until then
            "engagement_lambda": LAMFINAL[order],
        }
    )
    return users, latents
