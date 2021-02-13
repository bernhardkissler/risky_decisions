"""Calculate the values and certainty equivalents calculated by the theory of choice under risk and call auxilliary functions

Returns:
    functions: salience theory, regret theory, expected utility theory, rank dependent utility, cumulative prospect theory
"""

import math
import sys
from typing import List, Tuple


# SECTION Exceptions


class PositiveValuesOnlyError(Exception):
    """ Is raised when function gets a value out of domain of funtion """

    pass


class ZeroToOneOnlyError(Exception):
    """ Is raised when a probability bigger than one or smaller than zero is fed to the function """

    pass


class NonNegativeValuesOnly(Exception):
    """ Is raised when a value which shouldn't be negative is supplied as negative """

    pass


# SECTION Utility Functions


def utility_tversky_kahneman(
    x: float, r: float = 0, a: float = 0.88, l: float = 2.25
) -> float:
    """
    This is the classic tversky kahneman proposal for a utility function with common estimates of a = 0.88 and l = 2.25
    """
    if x >= r:
        outcome = (x - r) ** a
    else:
        outcome = -l * (-(x - r)) ** a
    return outcome


def ce_tversky_kahneman(
    x: float, r: float = 0, a: float = 0.88, l: float = 2.25
) -> float:
    """ inverse of tk utility """
    if x >= r:
        outcome = x ** (1 / a) + r
    else:
        outcome = -((-x / l) ** (1 / a)) + r
    return outcome


def lin_utility(x: float) -> float:
    """ A linear utility function; the utility of a value x equals x """
    return x


def lin_ce(x: float) -> float:
    """ Inverse of lin utility """
    return x


def root_utility(x: float, exp: float = 2.0, mult: float = 3) -> float:
    """
    A simple root utility function with u(x) = x**1/exp; 
    by default the quadratic root is used and loss aversion means 
    that losses are evaluated as 3 times as high as wins.
    """
    return x ** (1 / exp) if x > 0 else -mult * (-x) ** (1 / exp)


def root_ce(x: float, exp: float = 2.0, mult: float = 3) -> float:
    """ inverse of root utility """
    return x ** exp if x > 0 else -((x / (-mult)) ** exp)


# MARK NOt utilized in app
def kr_utility(x: float, mult: float = 10000) -> float:
    """ A logarithmic utilitly function based on köszegi rabin 2006 """
    return mult * math.log(x)


def bern_utility(x: float, a: float = 0, mult: float = 1) -> float:
    """ A simple utility function based on bernoulli's initial formulation of EU with an additional multiplier like KR 2006"""
    try:
        res = mult * math.log(a + x)
    except ValueError:
        res = math.nan
        # raise PositiveValuesOnlyError
    return res


def bern_ce(x: float, a: float = 0, mult: float = 1) -> float:
    """ Inverse of bernoulli utility """
    try:
        res = math.exp(x / mult) - a
    except ValueError:
        res = math.math.nan
    return res


# MARK Not utiltized in app
def pow_utility(x: float, exp: float = 2) -> float:
    """ Simple power utility function according to Stott 2006"""
    return x ** exp


# MARK Not utiltized in app
def quad_utility(x: float, a: float = 1) -> float:
    """ Simple quadratic utility function according to Stott 2006"""
    return a * x - x ** 2


def exp_utility(x: float, a: float = 1) -> float:
    """ Simple Exponential utility function according to Stott 2006"""
    return 1 - math.exp(-a * x)


def exp_ce(x: float, a: float = 1) -> float:
    """ Inverse of exp_utility """
    return -math.log(1 - x) / a


# MARK Not utiltized in app
def bell_utility(x: float, a: float = 1, b: float = 1) -> float:
    """ Simple Bell utility function according to Stott 2006"""
    return b * x - math.exp(-a * x)


# MARK Not utiltized in app
def hara_utility(x: float, a: float = 1, b: float = 1) -> float:
    """ Simple hara utility function according to Stott 2006"""
    return -((b + x) ** a)


# SECTION Bivariate Utility Function


def additive_habits(
    c: float, y: float, um_function=lin_utility, um_kwargs={}, eta: float = 0.1
) -> float:
    """Bivariate utility function based on Muermann Gollier 2010 showing simple interaction dynamics

    Args:
        c (float): actual  payoff
        y (float): expected payoff 
        um_function (function): univariate utility function as used in EU
        eta (float): positive weight indicating the importance of expectations. Typicall 0 <eta<1

    Returns:
        float: utility of actual and expected payoff
    """
    return um_function(c - eta * y, **um_kwargs)


def additive_habits_ce(
    c: float,
    y: float,
    # um_function=lin_utility,
    um_kwargs={},
    eta: float = 0.1,
    ce_function=lin_ce,
):
    """Inverse of additive habits utility function
    """
    return ce_function(c, **um_kwargs) + eta * y


# SECTION Regret Function


def ls_regret(
    x_1, x_2, um_function=lin_utility, um_kwargs={}, weight=1,
):
    """ classic regret function proposed by Loomes and Sugden 1982 """
    return um_function(x_1, **um_kwargs) + weight * (
        um_function(x_1, **um_kwargs) - um_function(x_2, **um_kwargs)
    )


def ls_regret_ce(x_1, x_2, um_function=lin_utility, um_kwargs={}, weight=1):
    """the 'inverse' of Loomes and Sugden 1982's regret function used to calculate the certainty equivalent

    Args:
        x_1 (float): the utility
        x_2 (float): the context payoff
        um_function (function, optional): the utility function used. Defaults to lin_utility.
        um_kwargs (dict, optional): the kwargs used by the utility and certainty equivalent functions. Defaults to {}.
        weight (float, optional): used to trade of consumption and regret utility. Defaults to 1.
    """
    return (x_1 + weight * um_function(x_2, **um_kwargs)) / (1 + weight)


# SECTION Salience Functions


def og_salience(x_1: float, x_2: float, theta: float = 0.1) -> float:
    # check what theta is really supposed to do; Is it only supposed to prevent Div by zero? --> Doesn't say in the text. It is simply a degree of freedom to fit data
    """ basic salience function proposed as more tractable parametrization in original paper """
    return abs(x_1 - x_2) / (abs(x_1) + abs(x_2) + theta)


# SECTION Type Checks helpers


def list_cleaning(
    pays: List[float], probs: List[float]
) -> Tuple[List[float], List[float]]:
    """
    makes sure that the two arguments are: list of numbers and have equal length; probs have to add to 1
    """
    if type(pays) != list or type(probs) != list:
        # print("Please provide two lists of equal length as inputs")
        sys.exit(1)

    elif len(pays) != len(probs):
        # print("Please provide two lists of equal length as inputs")
        sys.exit(1)
    else:
        try:
            pays_fl = [float(i) for i in pays]
            probs_fl = [float(i) for i in probs]
        except:
            # print("Please provide two lists of numbers as inputs")
            sys.exit(1)
        if not math.isclose(sum(probs), 1):
            # print("Your list of probabilities has to add up to 1")
            # print("The sum is currently {}.".format(sum(probs_fl)))
            sys.exit(1)
        else:
            return pays_fl, probs_fl


# SECTION Probability Weighting FUnctions


def weigh_tversky_kahneman(p: float, d: float = 0.65) -> float:
    """
    This returns the decision weight of a single input. The formula is based on Tversky and Kahneman and the classic value for d is 0.65
    """
    # TODO come up with a consisten way of testing inputs
    if p < 0 or p > 1:
        raise ZeroToOneOnlyError
    elif d < 0:
        raise NonNegativeValuesOnly
    else:
        return (p ** d) / ((p ** d + (1 - p) ** d) ** (1 / d))


def weigh_goldstein_einhorn(p: float, b: float = 0.5, a: float = 0.6) -> float:
    if p < 0 or p > 1:
        raise ZeroToOneOnlyError
    elif b < 0:
        raise PositiveValuesOnlyError
    else:
        return ((b * p) ** a) / ((b * p) ** a + (1 - p) ** a)


def weigh_prelec(p: float, b: float = 0.5, a: float = 0.6) -> float:
    if p < 0 or p > 1:
        raise ZeroToOneOnlyError
    else:
        return math.exp(-b * (-math.log(p)) ** a)


def weigh_lin(p: float) -> float:
    """ Simple linear weighting function; See Stott 2006"""
    if p < 0 or p > 1:
        raise ZeroToOneOnlyError
    else:
        return p


def weigh_pow(p: float, r: float) -> float:
    """ Simple power weighting function; See Stott 2006"""
    if p < 0 or p > 1:
        raise ZeroToOneOnlyError
    else:
        return p ** r


# SECTION Main Functions


def RDRA_theory(
    pays: List[List[float]],
    probs: List[List[float]],
    um_function=lin_utility,
    um_kwargs={},
    ce_function=lin_ce,
    gl_function=root_utility,
    gl_kwargs={},
) -> float:
    """
    Calculates reference dependend risk attitude - expected utility for a given target lottery and reference lottery of any size
    Use repeatdly to find PPE/CPE in line with RK2007 
    tested against KR 2007
    https://www.experimentalforschung.econ.uni-muenchen.de/studium/veranstaltungsarchiv/b_e_economics/ind_decision_2.pdf

    """

    prim_pays, ref_pays = pays[0], pays[1]
    prim_probs, ref_probs = probs[0], probs[1]

    partial_result = []
    for pay_outer in prim_pays:
        consumption_utility = um_function(pay_outer, **um_kwargs)
        gain_loss_utility = [
            ref_probs[i]
            * gl_function(
                um_function(pay_outer, **um_kwargs)
                - um_function(pay_inner, **um_kwargs),
                **gl_kwargs,
            )
            for i, pay_inner in enumerate(ref_pays)
        ]
        gain_loss_utility = sum(gain_loss_utility)
        partial_result.append(consumption_utility + gain_loss_utility)
    utility = sum(
        [prim_probs[i] * partial_result[i] for i, _ in enumerate(partial_result)]
    )
    # avg_pay_prim = sum([prim_pays[i] * prim_probs[i] for i, _ in enumerate(prim_pays)])
    # TODO update ce calculation
    ce = math.nan
    return utility, ce


def RDRA_wrapper(pays, probs):
    if RDRA_theory([pays[0], pays[0]], [probs[0], probs[0]]) >= RDRA_theory(
        [pays[1], pays[1]], [probs[1], probs[1]]
    ):
        is_CPE = True
    if RDRA_theory([pays[0], pays[0]], [probs[0], probs[0]]) >= RDRA_theory(
        [pays[1], pays[0]], [probs[1], probs[0]]
    ):
        is_PPE = True
    return is_CPE, is_PPE


def sav_dis_theory(
    pays: List[float],
    probs: List[List[float]],
    bivu_function=additive_habits,
    bivu_kwargs={},
    bivu_function_ce=additive_habits_ce,
    um_function=root_utility,
    um_kwargs={},
    ce_function=root_ce,
    k: float = 0.5,
) -> float:
    """Ex Ante Savoring and Ex Post Disappointment theory by Gollier and Muermann 2010

    Args:
        pays (List[float]): 
        probs (List[List[float]]): 
        bivu_function ([type]): bivariate utility function trading off expected payoff and actual payoff 
        um_function ([type], optional): univariate utility function called in bivu to model deminishing returns... . Defaults to lin_utility.
        k (float, optional): weight of Savoring in relation to Ex post disappointment. Typically bigger than zero

    Returns:
        float: The unique value assigned by the theory to a given target lottery in the context of the second theory
    """
    probs_obj, probs_subj = probs[0], probs[1]

    ant_val = ce_function(
        sum(
            [
                um_function(pays[i], **um_kwargs) * probs_subj[i]
                for i, _ in enumerate(probs_subj)
            ]
        ),
        **um_kwargs,
    )
    act_val = sum(
        [
            bivu_function(
                pays[i],
                ant_val,
                um_function=um_function,
                um_kwargs=um_kwargs,
                **bivu_kwargs,
            )
            * probs_obj[i]
            for i in range(len(probs_obj))
        ]
    )
    utility = k * um_function(ant_val, **um_kwargs) + act_val
    try:
        ce = bivu_function_ce(
            act_val,
            ant_val,
            # um_function=um_function,
            um_kwargs=um_kwargs,
            **bivu_kwargs,
            ce_function=ce_function,
        )
    except:
        ce = math.nan
    return utility, ce


def salience_theory(
    pays: List[List[float]],
    probs: List[float],
    sl_function=og_salience,
    sl_kwargs={},
    um_function=root_utility,
    um_kwargs={},
    ce_function=root_ce,
    delta: float = 0.7,
    correl_bool: bool = True,
) -> float:
    """Smooth Salience theory as described in the original paper.

    Args:
        pays (List[List[float]]): 2 dim input of payoffs where the first element is the target lottery and the second element the context lottery
        probs (List[float]): 1 or 2 dim input of probs belonging to the payoffs above
        delta (float, optional): degree of local thinking in original model; should be between 0 and 1 where one represents non-local rational thinking. Defaults to 0.5.
        correl_bool (bool, optional): are payoffs correlated i.e. do they share probs or don't they. Defaults to True !!! depreciated, correlation is now evaluated based on the length of context_payoffs (i.e., are there more than 1).

    Returns:
        float: the unique value of the target lottery compared to the context lottery. Might be exteded to certainty equivalent ... later
    """
    pays_prim, pays_cont = pays[0], pays[1]

    if len(pays_cont) == 1:
        sal_vals = [
            sl_function(pays_prim[i], pays_cont[0], **sl_kwargs)
            for i in range(len(pays_prim))
        ]
    else:
        sal_vals = [
            sl_function(pays_prim[i], pays_cont[i], **sl_kwargs)
            for i in range(len(pays_prim))
        ]
    av_salience = sum(
        [(delta ** (-sal_vals[i])) * probs[i] for i in range(len(sal_vals))]
    )
    probs_weights = [
        ((delta ** (-sal_vals[i])) / av_salience) * probs[i] for i in range(len(probs))
    ]
    utility = sum(
        [
            um_function(pays_prim[i], **um_kwargs) * probs_weights[i]
            for i in range(len(pays_prim))
        ]
    )
    try:
        ce = ce_function(utility, **um_kwargs)
    except:
        ce = math.nan
    return utility, ce


# print(salience_theory([[1, 2, 3], [4, 5, 6]], [0.3, 0.4, 0.3]))


def regret_theory(
    pays: List[List[float]],
    probs: List[float],
    um_function=root_utility,
    um_kwargs={},
    ce_function=root_ce,
    rg_function=ls_regret,
    rg_function_ce=ls_regret_ce,
    rg_kwargs={},
) -> float:
    """Implementation of Regret theory according to Loomes and Sugden 1982.

    Args:
        pays (List[List[float]]): Nested list of target and context pays of equal length, where the first sublist are the target pays and the second the context pays.
        probs (List[float]): List of probabilities. Has to be the same length as the target and context pays and sum to 1
        um_function ([type], optional): The utility function applied to individual values. Defaults to lin_utility.
        um_kwargs (dict, optional): . Defaults to {}. The arguments used by the utility function
        rg_function ([type], optional): The regret function used. Defaults to ls_regret.
        rg_kwargs (dict, optional): . Defaults to {}. The arguments used by the regret function

    Returns:
            utility: unique value of target lottery in relation to context
            ce: certainty equivalent of the lottery value
    """
    target_pay, context_pay = pays[0], pays[1]

    if len(context_pay) == 1:
        pays_delta = [
            rg_function(
                target_pay[i],
                context_pay[0],
                um_function=um_function,
                um_kwargs=um_kwargs,
                **rg_kwargs,
            )
            for i, _ in enumerate(target_pay)
        ]
    else:
        pays_delta = [
            rg_function(
                target_pay[i],
                context_pay[i],
                um_function=um_function,
                um_kwargs=um_kwargs,
                **rg_kwargs,
            )
            for i, _ in enumerate(target_pay)
        ]
    wavg_pays = sum([pays_delta[i] * probs[i] for i, _ in enumerate(pays_delta)])
    utility = wavg_pays
    try:
        if len(context_pay) == 1:
            ce_val = rg_function_ce(
                utility,
                context_pay[0],
                um_function=um_function,
                um_kwargs=um_kwargs,
                # ce_function=ce_function,
                **rg_kwargs,
            )
            ce = ce_function(ce_val, **um_kwargs)
        else:
            ce_vals = [
                rg_function_ce(
                    utility,
                    context_pay[i],
                    um_function=um_function,
                    um_kwargs=um_kwargs,
                    # ce_function=ce_function,
                    **rg_kwargs,
                )
                for i, _ in enumerate(target_pay)
            ]
            ce = ce_function(
                sum([ce_vals[i] * probs[i] for i, _ in enumerate(ce_vals)]), **um_kwargs
            )
    except:
        ce = math.nan
    return utility, ce


def expected_utility(
    pays: List[float],
    probs: List[float],
    um_function=bern_utility,
    um_kwargs={},
    ce_function=bern_ce,
) -> List[float]:
    """Implementation of Expected Utility Theory and its Certainty

    Args:
        pays (List[float]): Vector of the payoffs for all outcomes
        probs (List[float]): Vector of the probabilities for all outcomes (must be the same length as pays)
        um_function (function, optional): The utility function used to transform payoffs to utilties. Defaults to bern_utility.
        um_kwargs (dict, optional): Generic keyword-arguments supplied to the utility function . Defaults to {}.
        ce_function (function, optional): The 'inverse' of the utility function used to calculate the certainty equivalent. Defaults to bern_

    Returns:
        List[float]: The utility assigned to the lottery by EU and the associated certainty equivalent
    """
    pays_ch, probs_ch = list_cleaning(pays, probs)
    pays_ch_ut = [um_function(i, **um_kwargs) for i in pays_ch]
    ind_vals = [pays_ch_ut[i] * probs_ch[i] for i in range(len(pays_ch))]
    utility = sum(ind_vals)
    try:
        ce = ce_function(utility, **um_kwargs)
    except:
        ce = math.nan
    return utility, ce


# MARK Not utiltized in app
def rank_dependent_utility(
    pays: List[float],
    probs: List[float],
    pw_function=weigh_tversky_kahneman,
    um_function=root_utility,
    um_kwargs={},
    ce_function=root_ce,
    pw_kwargs={},
) -> float:
    # Sort values by size of payoffs (descending)
    pays_ch, probs_ch = list_cleaning(pays, probs)
    vals = list(zip(pays_ch, probs_ch))
    vals.sort(key=lambda elem: elem[0], reverse=True)
    pays_sorted, probs_sorted = zip(*vals)
    pays_sorted_ut = [um_function(i, **um_kwargs) for i in pays_sorted]
    # Calculate marginal decision weights
    decision_weights = []
    for i, _ in enumerate(probs_sorted):
        if i == 0:
            dec_weight = pw_function(sum(probs_sorted[: i + 1]), **pw_kwargs)
            decision_weights.append(dec_weight)
        else:
            dec_weight = pw_function(
                sum(probs_sorted[: i + 1]), **pw_kwargs
            ) - pw_function(sum(probs_sorted[:i]), **pw_kwargs)
            decision_weights.append(dec_weight)
    ind_vals = [
        pays_sorted_ut[i] * decision_weights[i] for i in range(len(pays_sorted_ut))
    ]
    utility = sum(ind_vals)
    try:
        ce = ce_function(utility, **um_kwargs)
    except:
        ce = math.nan
    return utility, ce


def cumulative_prospect_theory(
    pays: List[float],
    probs: List[float],
    pw_function=weigh_tversky_kahneman,
    um_function=utility_tversky_kahneman,
    pw_kwargs={},
    um_kwargs={},
    ce_function=ce_tversky_kahneman,
) -> float:
    pays_ch, probs_ch = list_cleaning(pays, probs)
    vals = list(zip(pays_ch, probs_ch))
    # split into pos and neg values
    vals_pos = [i for i in vals if i[0] >= 0]
    vals_neg = [i for i in vals if i[0] < 0]
    # zip and order by absolute value
    try:
        vals_pos.sort(key=lambda elem: elem[0], reverse=True)
        pays_sorted_pos, probs_sorted_pos = zip(*vals_pos)
    except:
        pays_sorted_pos, probs_sorted_pos = [], []
    try:
        vals_neg.sort(key=lambda elem: elem[0])
        pays_sorted_neg, probs_sorted_neg = zip(*vals_neg)
    except:
        pays_sorted_neg, probs_sorted_neg = [], []
    # weigh probabilities
    decision_weights_pos = []
    for i, _ in enumerate(probs_sorted_pos):
        if i == 0:
            dec_weight_pos = pw_function(sum(probs_sorted_pos[: i + 1]), **pw_kwargs)
            decision_weights_pos.append(dec_weight_pos)
        else:
            dec_weight_pos = pw_function(
                sum(probs_sorted_pos[: i + 1]), **pw_kwargs
            ) - pw_function(sum(probs_sorted_pos[:i]), **pw_kwargs)
            decision_weights_pos.append(dec_weight_pos)
    decision_weights_neg = []
    for i, _ in enumerate(probs_sorted_neg):
        if i == 0:
            dec_weight_neg = pw_function(sum(probs_sorted_neg[: i + 1]), **pw_kwargs)
            decision_weights_neg.append(dec_weight_neg)
        else:
            dec_weight_neg = pw_function(
                sum(probs_sorted_neg[: i + 1]), **pw_kwargs
            ) - pw_function(sum(probs_sorted_neg[:i]), **pw_kwargs)
            decision_weights_neg.append(dec_weight_neg)
    # collect all outcomes
    probs_final = decision_weights_pos + decision_weights_neg
    # modify utilites
    pays_final = [um_function(i, **um_kwargs) for i in pays_sorted_pos] + [
        um_function(i, **um_kwargs) for i in pays_sorted_neg
    ]
    ind_vals = [pays_final[i] * probs_final[i] for i in range(len(pays_final))]

    utility = sum(ind_vals)
    try:
        ce = ce_function(utility, **um_kwargs)
    except:
        ce = math.nan
    return utility, ce

