"""純計算的 Gamma Pinning（釘價效應）判斷引擎 —— 不做任何網路 I/O。

移植自 smart-money（stock.agent）專案的同名模組——寫的時候就刻意設計成
只吃數字/dict 進來、回傳 dict 出去，不依賴任何特定資料源（原本是配
yfinance 用，這裡改配這個專案自己的 Moomoo 即時資料），所以能整支檔案
原封不動搬過來，不用改一行計算邏輯。

Pinning（釘價）背後的物理機制：現貨越接近某個未平倉量集中的履約價，且做
市商淨多 Gamma（正 Gamma 區）時，做市商為了維持 Delta 中性，其避險買賣盤
會呈現「跌了買、漲了賣」的逆勢特性，客觀上把價格「釘」在那個履約價附近；
淨空 Gamma（負 Gamma 區）時避險行為方向相反（追漲殺跌），這個機制不成立，
現貨反而容易順著原本的方向加速——這正是本專案 GEXCalculator 判斷的正負
Gamma 分水嶺（zero_gamma）在這裡被拿來當「Pinning 是否成立」的必要條件。
"""

from __future__ import annotations

from typing import Sequence

# 現貨距離 Pin Strike 在這個百分比之內，才算「夠近」而有機會被磁吸——
# 超過這個距離，就算做市商淨多 Gamma，也不構成有意義的 Pinning 訊號，
# 只是「還沒被磁吸到」而已。經驗法則、非回測驗證過的最佳解。
PIN_PROXIMITY_THRESHOLD_PCT = 1.5

# Pin Strike 的未平倉量（Call+Put）占全鏈總未平倉量的比例，至少要達到這個
# 門檻，才算「夠集中」的磁吸點——分散在很多履約價、沒有明顯單一大倉位的
# 情況，就算現貨剛好在某個履約價附近，也不構成有意義的 Pinning 訊號。
PIN_OI_CONCENTRATION_MIN_PCT = 12.0

# find_pin_strike 找候選履約價時，只看現貨這個距離百分比以內的——彙總多個
# 到期日後，遠離現貨的稀疏舊倉位（LEAPS留下的巨量OI）常常單一履約價的
# 絕對OI就大過所有近端履約價總和，會把 Pin Strike 誤導到跟「現貨真的會被
# 磁吸到哪裡」完全無關的地方。這是在 smart-money 專案用真實 TSLA 資料
# 驗證時實測抓到的真bug：Pin Strike 曾經落在距現貨 21.7% 遠的履約價，
# 明顯是LEAPS的陳舊大倉位，不是近端的 Pinning 磁吸點。
PIN_CANDIDATE_MAX_DISTANCE_PCT = 20.0

# 0～100 分數換算成文字標籤的門檻。
_LABEL_THRESHOLDS = ((30, "低"), (60, "中"), (80, "高"))

PinningRegime = str  # "PINNING" | "BREAKOUT" | "NEUTRAL"


def find_pin_strike(gex_by_strike: Sequence[dict], spot: float | None = None) -> float | None:
    """找出總未平倉量（Call OI + Put OI）最大的履約價——做市商避險部位
    最集中的地方，是 Pinning 磁吸力最強的候選點。

    跟 Max Pain（讓期權買方到期內在價值總和最小的履約價）是兩個不同的
    概念，經常但不總是重合：Max Pain 是「對期權賣方最有利」的理論價位，
    Pin Strike 是「未平倉量最集中、做市商避險部位最大」的實際價位——
    後者才是 Gamma Pinning 效應真正的物理機制（做市商為了維持 Delta 中性，
    在這個履約價附近的避險買賣盤最密集）。呼叫端可以同時顯示兩者、標示
    是否重合，重合時代表訊號更一致。

    傳入 spot 時，候選履約價會先限制在 PIN_CANDIDATE_MAX_DISTANCE_PCT 以內
    （見該常數註解）；限制後如果一個候選都沒有（例如現貨附近完全沒有
    OI），退回用全部履約價找，不要直接回傳 None——「近端沒有明顯集中點」
    本身也是有意義的資訊，用全鏈次佳解讓 Pinning 分數自然算低分反映
    這件事，比直接放棄判斷更有用。不傳 spot（或傳 None/<=0）時維持舊行為，
    在全部履約價裡找，方便不需要這層過濾的呼叫端或測試沿用。
    """
    if not gex_by_strike:
        return None
    candidates = gex_by_strike
    if spot is not None and spot > 0:
        nearby = [
            row for row in gex_by_strike
            if abs(row["strike"] - spot) / spot * 100 <= PIN_CANDIDATE_MAX_DISTANCE_PCT
        ]
        if nearby:
            candidates = nearby
    return max(candidates, key=lambda row: row["call_oi"] + row["put_oi"])["strike"]


def _oi_concentration_pct(gex_by_strike: Sequence[dict], pin_strike: float) -> float:
    total_oi = sum(row["call_oi"] + row["put_oi"] for row in gex_by_strike)
    if total_oi <= 0:
        return 0.0
    pin_oi = sum(
        row["call_oi"] + row["put_oi"] for row in gex_by_strike if row["strike"] == pin_strike
    )
    return pin_oi / total_oi * 100


def _score_to_label(score: int) -> str:
    for threshold, label in _LABEL_THRESHOLDS:
        if score < threshold:
            return label
    return "極高"


def score_pinning(
    spot: float,
    pin_strike: float | None,
    oi_concentration_pct: float,
    max_pain: float,
    call_wall: float | None,
    put_wall: float | None,
    in_positive_gamma: bool,
) -> dict | None:
    """判斷現貨目前處於「Pinning 釘價區間」還是「突破區間」，並給出 0～100
    的透明分數（三個分量，不是回測驗證過的最佳解，是把判斷邏輯攤開透明，
    而不是黑箱模型）。

    三個分量（各自獨立、加總封頂 100）：
      1. proximity_score（0~40）：現貨距離 Pin Strike 越近，磁吸力越強，
         超過 2 倍 PIN_PROXIMITY_THRESHOLD_PCT 距離線性歸零。
      2. gamma_score（0 或 30）：正 Gamma 區是 Pinning 得以成立的必要
         條件（見檔頭 docstring），是就給滿分，不是就是 0 分——這不是
         線性關係，是二元的機制開關。
      3. concentration_score（0~30）：Pin Strike 未平倉量占全鏈比例越
         高，磁吸力越強，25% 以上視為滿分濃度。

    Regime 三選一：
      - BREAKOUT：現貨已經突破 Call Wall 或跌破 Put Wall——做市商避險
        部位的防線已經失守，趨勢型避險行為（追價/追跌）主導，不管分數
        多高，Pinning 機制在定義上已經不成立。
      - PINNING：現貨在 Wall 區間內、且分數達到「高」以上（>=60）。
      - NEUTRAL：在 Wall 區間內，但訊號還不夠強（離 Pin Strike 太遠、
        或處於負 Gamma、或 Pin Strike 籌碼不夠集中）。

    pin_strike/oi_concentration_pct 是拆出來的獨立參數（不是自己從
    gex_by_strike 算），讓輕量的重複計算可以拿之前算好的 pin_strike/集中度
    重用，只換「現在」的即時現貨價重新評分，不用每次都重抓整條期權鏈。
    compute_pinning_analysis() 是這支函式的完整版（spot/gex_by_strike 都是
    當下新鮮算的）。

    pin_strike 為 None 時回傳 None（無法判斷，不瞎猜，呼叫端應優雅跳過
    這個加分欄位）。
    """
    if pin_strike is None or spot <= 0:
        return None

    distance_pct = abs(spot - pin_strike) / spot * 100

    proximity_cap_pct = PIN_PROXIMITY_THRESHOLD_PCT * 2
    proximity_score = max(0.0, (proximity_cap_pct - distance_pct) / proximity_cap_pct) * 40

    gamma_score = 30.0 if in_positive_gamma else 0.0

    concentration_cap_pct = 25.0
    concentration_score = min(oi_concentration_pct, concentration_cap_pct) / concentration_cap_pct * 30

    # 加 0.5 取整以符合一般「四捨五入」而非銀行家捨入。
    score = int(proximity_score + gamma_score + concentration_score + 0.5)
    label = _score_to_label(score)

    # A missing Wall is *absence of evidence*, not evidence of a breakout:
    # call_wall/put_wall are None when the chain simply has no qualifying
    # strike on that side of spot, which says nothing about whether a
    # defence line was broken. Only a wall that exists and has been crossed
    # counts. (Before the walls were constrained to the correct side of
    # spot, a stray deep-ITM "Call Wall" below the market made this fire a
    # spurious BREAKOUT; None must not resurrect that failure mode.)
    has_broken_wall = (call_wall is not None and spot > call_wall) or (
        put_wall is not None and spot < put_wall
    )
    is_close_enough = distance_pct <= PIN_PROXIMITY_THRESHOLD_PCT
    is_concentrated_enough = oi_concentration_pct >= PIN_OI_CONCENTRATION_MIN_PCT

    if has_broken_wall:
        regime: PinningRegime = "BREAKOUT"
    elif in_positive_gamma and is_close_enough and is_concentrated_enough and score >= 60:
        regime = "PINNING"
    else:
        regime = "NEUTRAL"

    return {
        "pin_strike": pin_strike,
        "pin_strike_matches_max_pain": pin_strike == max_pain,
        "distance_pct": distance_pct,
        "oi_concentration_pct": oi_concentration_pct,
        "in_positive_gamma": in_positive_gamma,
        "has_broken_wall": has_broken_wall,
        "score": score,
        "label": label,
        "regime": regime,
    }


def compute_pinning_analysis(
    gex_by_strike: Sequence[dict],
    spot: float,
    max_pain: float,
    call_wall: float | None,
    put_wall: float | None,
    in_positive_gamma: bool,
) -> dict | None:
    """完整版：從一整條當下重新彙總的期權鏈（gex_by_strike）自己找
    Pin Strike 與集中度，再交給 score_pinning() 評分。
    """
    pin_strike = find_pin_strike(gex_by_strike, spot)
    if pin_strike is None:
        return None
    oi_concentration_pct = _oi_concentration_pct(gex_by_strike, pin_strike)
    return score_pinning(
        spot, pin_strike, oi_concentration_pct, max_pain, call_wall, put_wall, in_positive_gamma,
    )
