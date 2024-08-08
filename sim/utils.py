import random
import copy
import math
import numpy as np

cexdex_arbitrage = "cexdex"
dexdex_arbitrage = "dexdex"

user_util_stat = "user_util"
user_util_nk_stat = "user_util_nokickback"
arbitrage_amount_stat = "arbitrage_amount"
trade_amount_stat = "trade_amount"
preferences_stat = "preference_amount"
random_walk_stat = "random_walks"
all_stat_names = [user_util_stat, user_util_nk_stat, arbitrage_amount_stat, trade_amount_stat, preferences_stat, random_walk_stat]

random_type = "random"
simple_type = "simple"
random_walk_range_type = "random_walk_max_step"
random_walk_range_relative_type = "random_walk_max_step_scaled_by_num_users"
random_walk_scaled_type = "random_walk_scaled_step"
net_demand_type = "net_demand_ratio" 
max_demand_type = "max_demand_ratio" 
max_demand_scale_type = "max demand ratio scaled by num users" 
max_demand_batch_type = "batched max demand ratio" 
all_param_types = [random_type, simple_type, random_walk_range_type, random_walk_range_relative_type, random_walk_scaled_type, net_demand_type, max_demand_type, max_demand_scale_type, max_demand_batch_type]
param_type_xlabels = {
    random_type: "std uniform random",
    simple_type: "token",
    random_walk_range_type: "max step",
    random_walk_range_relative_type: "max step scaled by num users",
    random_walk_scaled_type: "scaled step",
    net_demand_type: "net demand ratio", 
    max_demand_type: "max demand ratio", 
    max_demand_scale_type: "max demand ratio scaled by num users", 
    max_demand_batch_type: "batched max demand ratio", 
}
default_params = {
    random_type: lambda num_intervals, miu=None, mau=None: exponential_decay_function(.01,4, num_intervals),
    simple_type: lambda _num_intervals, miu=None, mau=None: None,
    random_walk_range_type: lambda num_intervals, min_users, max_users: [min_users + k*(2*max_users-min_users)/(num_intervals-1) for k in range(num_intervals)],
    random_walk_range_relative_type: lambda num_intervals, miu=None, mau=None: [.0001 + k*(.5-.0001)/(num_intervals-1) for k in range(num_intervals)],#exponential_decay_function(.0001, .25, num_intervals),
    random_walk_scaled_type: lambda num_intervals, miu=None, mau=None: [.0001 + k*(1-.0001)/(num_intervals-1) for k in range(num_intervals)],
    net_demand_type: "net demand ratio", 
    max_demand_type: "max demand ratio", 
    max_demand_scale_type: "max demand ratio scaled by num users", 
    max_demand_batch_type: "batched max demand ratio",     
}

same_type = "same"
diff_type = "diff"

random_walk_stats = {}
random_walk_function = {
    random_walk_range_type: "limited_random_walk_range",
    random_walk_range_relative_type: "limited_random_walk_range",
    random_walk_scaled_type: "limited_random_walk_scaled",
    max_demand_type: "max_demand_sample", 
    max_demand_scale_type: "max_demand_sample", 
    max_demand_batch_type: "max_demand_sample",   
}

class ExecutionException(Exception):
    pass
def require(cond, msg=None):
    if not cond: raise ExecutionException(msg)


def pool_swap(poolA, poolB, amtA):
    """Uniswap v2 rule trade A for B

    Args:
        poolA (number): amount of token A
        poolB (number): amount of token B
        amtA (number): amount of A to trade

    Returns:
        number: amount of B returned
    """
    # trade A for B
    amtB = poolB - poolA*poolB / (poolA + amtA)
    assert np.isclose(poolA*poolB, (poolA + amtA)*(poolB - amtB))
    return amtB

def create_swap(sndr,qty,rsv):
    """create a swap transaction

    Args:
        sndr (string): username of sender
        qty (number): amount to trade (Positive to trade token A, Negative to trade token B)
        rsv (number): slippage limit (minimum of other token to get) (Positive to get token A, Negative to get token B)

    Returns:
        dict: transaction
    """
    if qty < 0:
        assert rsv >= 0
    elif qty > 0:
        assert rsv <= 0
    return dict(type="swap",sndr=sndr,qty=qty,rsv=rsv,auth="auth")

def utility(pref, portfolio):
    """calculate net utility

    Args:
        pref (list of 2 numbers): preference for each token
        portfolio (list of two numbers): amount owned of each token

    Returns:
        number: net utility
    """
    assert len(pref) == len(portfolio) == 2
    return pref[0] * portfolio[0] + pref[1] * portfolio[1]

def logistic_function(x, k, dist_y, mid_y):
    """logistic curve function

    Args:
        x (number): x value
        k (number): growth rate of function
        dist_y (number): max distance of y +/- mid_y
        mid_y (number): mid point of function for x==0

    Returns:
        number: evaluation of function at x
    """
    adj = mid_y - dist_y
    y = adj + 2*dist_y/(1 + np.exp(-x*k))
    return y

def exponential_decay_function(min_val, max_val, num_steps):
    decay = 1 - np.exp(np.log(max_val/min_val)/ (num_steps-1))
    values = [min_val*(1-decay)**x for x in range(num_steps)]
    return values

def confidence_interval(m,s,n,alpha=.025, z=1.96):
    ci = [z*s[i]/np.sqrt(n) for i in range(len(m))]
    return ci

def get_random_walk_stat_name(param_type):
    if param_type in random_walk_function:
        return random_walk_stats[random_walk_function[param_type]]["measuring"]
    return ""
def get_random_walk_stats(param_type, param, num_users, iter_num):
    if param_type in random_walk_function:
        return random_walk_stats[random_walk_function[param_type]][param][num_users][iter_num]
    return 

def limited_random_walk_range(max_step, min_step, num_points, direction_foo, random_foo, num_users=None, param=None):
    if num_users is None:
        num_users = num_points
    if param is None:
        param = max_step
    print("max", max_step, "min", min_step, "num", num_points)
    assert(max_step >= 1.0)
    assert(min_step <= 1.0)
    points = []
    step = 0
    misses = 0
    while len(points) < num_points:            
        x = random_foo()
        next_step = step
        if direction_foo(x):
            next_step+=1
        else:
            next_step-=1
        # print("next", next_step)
        if next_step <= max_step and next_step >= min_step:
            points.append(x)
            step = next_step
        else:
            misses+=1
    global random_walk_stats
    if "limited_random_walk_range" not in random_walk_stats:
        random_walk_stats["limited_random_walk_range"] = {"measuring": "misses", "index": "num_users"}
    if param not in random_walk_stats["limited_random_walk_range"]:
        random_walk_stats["limited_random_walk_range"][param] = {}
    if num_users not in random_walk_stats["limited_random_walk_range"][param]:
        random_walk_stats["limited_random_walk_range"][param][num_users] = []
    random_walk_stats["limited_random_walk_range"][param][num_users].append(misses)    
    return points
        
def limited_random_walk_scaled(num_points, direction_foo, random_foo, scale, num_users=None, param=None):
    if num_users is None:
        num_users = num_points
    if param is None:
        param = scale
    points = []
    step = 0
    max_step = 0

    while len(points) < num_points:            
        x = random_foo(step, scale)
        if direction_foo(x):
            step+=1
        else:
            step-=1
        if abs(step) > max_step:
            max_step = abs(step)
        points.append(x)
    # print(num_points, "max_step",max_step)
    global random_walk_stats
    if "limited_random_walk_scaled" not in random_walk_stats:
        random_walk_stats["limited_random_walk_scaled"] = {"measuring": "max_step", "index": "num_users"}
    if param not in random_walk_stats["limited_random_walk_scaled"]:
        random_walk_stats["limited_random_walk_scaled"][param] = {}
    if num_users not in random_walk_stats["limited_random_walk_scaled"][param]:
        random_walk_stats["limited_random_walk_scaled"][param][num_users] = []
    random_walk_stats["limited_random_walk_scaled"][param][num_users].append(max_step) 
    return points

def max_demand_sample( max_demand, min_demand, num_points, direction_foo, random_foo, num_users=None, param=None):
    if num_users is None:
        num_users = num_points
    if param is None:
        param = max_demand
    points  = random_foo(num_points)
    net_demand = sum([direction_foo(x) for x in points])
    misses = 0
    while net_demand > max_demand or net_demand < min_demand:
        points  = random_foo(num_points)
        net_demand = sum([direction_foo(x) for x in points])
        misses+=1
        # print("misses", misses,"demand", net_demand,"range", max_demand,min_demand, "num_points", num_points)
    global random_walk_stats
    if "max_demand_sample" not in random_walk_stats:
        random_walk_stats["max_demand_sample"] = {"measuring": "misses", "index": "num_users"}
    if param not in random_walk_stats["max_demand_sample"]:
        random_walk_stats["max_demand_sample"][param] = {}
    if num_users not in random_walk_stats["max_demand_sample"][param]:
        random_walk_stats["max_demand_sample"][param][num_users] = []
    random_walk_stats["max_demand_sample"][param][num_users].append(misses)    
    return points

def random_interleave(arr1, arr2):
    combined = []
    index1 = index2 = 0
    while len(combined) < len(arr1) + len(arr2):
        if random.choice([0,1]) == 0:
            combined.append(arr1[index1])
            index1+=1
            if index1 == len(arr1):
                combined += arr2[index2:]
                return combined
        else:
            combined.append(arr2[index2])
            index2+=1
            if index2 == len(arr2):
                combined += arr1[index1:]
                return combined
    return combined


def produce_sandwich(chain, tx_victims, attacker, debug=False):
    """figure out the optimal frontrun/backrun transactions for
    a sandwhich attack

    Args:
        chain (Chain): chain the sandwhich is on
        tx_victims: victim transaction(s) to sandwhich
        attacker (string): attacker name
        debug (bool, optional): print debug information. Defaults to False.

    Returns:
        tuple of dict swap transactions: frontrun swap transaction, backrun swap transaction
    """


    if not isinstance(tx_victims, list):
        tx_victims = [tx_victims]
    for tx_victim in tx_victims:
        assert tx_victim["type"] == "swap"

    #if the transaction amounts sum up to 0 there is no sandwhich attack
    tx_sum = sum(list(map(lambda x: x["qty"], tx_victims)))
    if tx_sum == 0:
        return create_swap(attacker,0,0), create_swap(attacker,0,0)

    min_front = 0

    if tx_sum < 0:
        #victim transactions are net trading token B for token A so frontrun
        #transaction will trade token B
        max_front = chain.accounts[attacker].tokens[1]
    else:
        max_front = chain.accounts[attacker].tokens[0]

    last_successful_front = 0
    # chain = copy.deepcopy(chain)
    while True:
        chain_copy = copy.deepcopy(chain)
        frontrun_amt = (min_front + max_front) / 2. 
        if tx_sum < 0:
            frontrun_amt = -frontrun_amt #trading token B for A so qty parameter is negative
        if debug:
            print("try", frontrun_amt, "min", min_front , "max", max_front)
        tx = create_swap(attacker, frontrun_amt, 0)
        try:
            chain_copy.apply(tx, debug=False)
            for tx_victim in tx_victims:
                chain_copy.apply(tx_victim, debug=False)
            min_front = abs(frontrun_amt)
            last_successful_front = frontrun_amt 
        except Exception as e:
            max_front = abs(frontrun_amt)
            if debug:
                print("Exception", e)

        if np.isclose(max_front, min_front):
            if debug:
                print("found", last_successful_front)
            break
        

    if tx_sum < 0:
        backrun_amt = pool_swap(chain.poolB, chain.poolA, -last_successful_front)
    else:
        #backrun is trading token B for A so qty parameter is negative
        backrun_amt = -pool_swap(chain.poolA, chain.poolB, last_successful_front)

    return create_swap(attacker, last_successful_front, 0), create_swap(attacker, backrun_amt, 0)

 # optimal amount of b to trade for a to get my preference

def optimal_trade_amt(poolA, poolB, prefA, prefB, portfA, portfB):
    """figure out the optimal amount of token B to trade to maximize increase in net utility 
    using algebra.

    Args:
        poolA (number): amount of token A in pool
        poolB (number): amount of token B in pool
        prefA (number): preference for token A
        prefB (number): preference for token B
        portfA (number): amount of token A in owned
        portfB (number): amount of token B in owned

    Returns:
        number: amount of token B to trade
    """
    assert prefA + prefB == 2
    a=prefA 
    asq = prefA**2
    c=poolA 
    d=poolB
    e=portfA
    f=portfB
    # print(a,c,d,e,f)
    if a-2 == 0:
        return np.inf
    amtB1 = (math.sqrt(2*a*c*d - asq*c*d)-a*d+2*d)/(a-2)
    amtB2 = (-math.sqrt(2*a*c*d - asq*c*d)-a*d+2*d)/(a-2)
    res = max(amtB1, amtB2)
    # print("optimal", amtB1, amtB2)
    assert res >= 0
    return res

def optimal_trade_amt_search(poolA, poolB, prefA, prefB, portfA, portfB):
    """figure out the optimal amount of token B to trade to maximize increase in net utility 
    using bisection search.

    Args:
        poolA (number): amount of token A in pool
        poolB (number): amount of token B in pool
        prefA (number): preference for token A
        prefB (number): preference for token B
        portfA (number): amount of token A in owned
        portfB (number): amount of token B in owned

    Returns:
        number: amount of token B to trade
    """
    min_b=0.0
    max_b=portfB
    util_init = utility([prefA, prefB],[portfA, portfB])
    util_old = util_init
    util_max = util_init
    while True:
        mid_b = (max_b+min_b)/2
        amtA = pool_swap(poolB, poolA, mid_b)
        util = utility([prefA, prefB],[portfA+amtA, portfB-mid_b])
        # print("amtB", mid_b, "util", util)
        if util > util_max:
            min_b = mid_b
            util_max = util
        else:
            max_b = mid_b
        if np.isclose(util, util_old):
            # print("best is", mid_b, "resulting in util", util, "util_old", util_old, "util_max", util_max, "util_init", util_init)
            if util >= util_init:
                return mid_b
            else:
                return 0
        util_old = util

def optimal_arbitrage_algebra(chain1, chain2, attacker):
    """figure out the optimal arbitrage transaction between two token pools
    using algebra.

    Args:
        chain1 (Chain): 1st chain with token pool
        chain2 (Chain): 2nd chain with other token pool
        attacker (string): arbitrager name

    Returns:
        tuple of dict (swap txs): (transaction for chain1 and transaction for chain2)
    """
    assert(chain1.static is None or chain2.static is None) #can't do arbitrage on two static chains
    if chain2.static:
        # print("chain1", chain1.price('a'), "chain2", chain2.price('a'))

        if chain2.price('A') > chain1.price('A'):
            tokenTrade = 'B'
        else:
            tokenTrade = 'A'
        c1 = chain1
        c2 = chain2
    elif chain1.static:
        if chain1.price('A') > chain2.price('A'):
            tokenTrade = 'B'
        else:
            tokenTrade = 'A'
        c2 = chain1
        c1 = chain2
    else:
        # if prefs[0] < prefs[1]: #todo fractional preferences?
        if abs(chain1.price('A')- chain2.price('A')) < abs(chain1.price('B')- chain2.price('B')):
            tokenTrade = 'A'
        else:
            tokenTrade = 'B'

        if chain1.price(tokenTrade) < chain2.price(tokenTrade):
            c1 = chain2 
            c2 = chain1
        else:
            c1 = chain1 
            c2 = chain2


    # print("chain1", chain1.price(tokenTrade), "chain2", chain2.price(tokenTrade), "trading token" , tokenTrade)
    #how much of tokenTrade we can sell to c1 and buy from c2 until they have the same price
    #how much of A we can sell to c1 and buy from c2 until they have the same price
    a=c1.product()
    b=c2.product()
    if tokenTrade == 'B':
        c=c1.poolB
        d=c1.poolA
        g=c2.poolB 
    else:
        c=c1.poolA
        d=c1.poolB 
        g=c2.poolA
    # print("a", a ,'b', b,'c', c,'d', d, 'g', g)
    csq = c**2
    gsq = g**2
    
    if c2.static is not None:
        # vars = [a,d,c2.price(tokenTrade)]
        amtB1 = -d+math.sqrt(a/c2.price(tokenTrade))
        amtB2 = -d-math.sqrt(a/c2.price(tokenTrade))
        amtB = max(amtB1, amtB2)
        amtA = amtB*c2.price(tokenTrade)
        # print("Optimal arbitrage1 amtTT", amtB, "amtOT", amtA, vars)   
    else:
        # vars = [a,b,c,d,g]
        #https://www.wolframalpha.com/input?i=%28c-x%29%2F%28a%2F%28c-x%29%29+%3D+%28g%2Bx%29%2F%28b%2F%28g%2Bx%29%29
        amtB1 = (-math.sqrt(a*b*csq + 2*a*b*c*g + a*b*gsq)+a*c+a*g+csq*(-d)-2*c*d*g-d*gsq)/(csq+2*c*g+gsq)
        amtB2 = ( math.sqrt(a*b*csq + 2*a*b*c*g + a*b*gsq)+a*c+a*g+csq*(-d)-2*c*d*g-d*gsq)/(csq+2*c*g+gsq)

        amtB = max(amtB1, amtB2)
        amtA = pool_swap(d,c,amtB)
    if amtB < 0:
        if np.isclose(chain1.price('A'), chain2.price('A')):
            amtB = 0
            amtA = 0
        else:
            require(amtB >= 0, f"Optimal arbitrage failed {amtB1} {amtB2}: prices [{chain1.price('A')}, {chain2.price('A')}] {np.isclose(amtB,0)} ")
            assert amtA >= 0



    if tokenTrade == 'A':
        tx_arb1 = create_swap(attacker, -amtB, 0)
        tx_arb2 = create_swap(attacker, amtA, 0)
    else:
        tx_arb1 = create_swap(attacker, amtB, 0)
        tx_arb2 = create_swap(attacker, -amtA, 0)

    if chain2.static:
        return tx_arb1, tx_arb2
    elif chain1.static:
        return tx_arb2, tx_arb1
    elif chain1.price(tokenTrade) > chain2.price(tokenTrade):
        return tx_arb1, tx_arb2
    else:
        return tx_arb2, tx_arb1

def optimal_arbitrage_search(chain1, chain2, prefs, attacker):
    """figure out the optimal arbitrage transaction between two token pools
    using bisection search. 
    TODO remove not possible to calcu;ate with search

    Args:
        chain1 (Chain): 1st chain with token pool
        chain2 (Chain): 2nd chain with other token pool
        prefs (list of numbers): [preference for token A, preference for token B]
        attacker (string): arbitrager name

    Returns:
        tuple of dict (swap txs): (transaction for chain1 and transaction for chain2)
    """
    min_b = 0
    
    if prefs[0] < prefs[1]: #todo fractional preferences?
        tokenTrade = 'A'
        max_b = chain1.accounts[attacker].tokens[1]
    else:
        tokenTrade = 'B'
        max_b = chain1.accounts[attacker].tokens[0]

    switch_chains = chain1.price(tokenTrade) < chain2.price(tokenTrade)
    if switch_chains:
        chainA = chain2
        chainB = chain1
    else:
        chainA = chain1
        chainB = chain2

    

    c1 = copy.deepcopy(chainA) 
    c2 = copy.deepcopy(chainB)

    tx_arb1 = create_swap(attacker, 0,0)
    tx_arb2 = create_swap(attacker, 0,0)
    last_successful_tx1 = tx_arb1
    last_successful_tx2 = tx_arb2


    while not np.isclose(c1.price(tokenTrade), c2.price(tokenTrade)):
        c1 = copy.deepcopy(chainA) 
        c2 = copy.deepcopy(chainB)

        mid_b = (max_b+min_b)/2 
        if tokenTrade == 'A':
            tx_arb1 = create_swap(attacker, -mid_b, 0)
            amtA = pool_swap(c1.poolB, c1.poolA, mid_b)
            tx_arb2 = create_swap(attacker, amtA, 0)
        else:
            tx_arb1 = create_swap(attacker, mid_b, 0)
            amtA = pool_swap(c1.poolA, c1.poolB, mid_b)
            tx_arb2 = create_swap(attacker, -amtA, 0)
        try:    
            c1.apply(tx_arb1, debug=False)
            c2.apply(tx_arb2, debug=False)
            last_successful_tx1 = tx_arb1
            last_successful_tx2 = tx_arb2
        except ExecutionException as e:
            print(e)
            pass
        if c1.price(tokenTrade) > c2.price(tokenTrade):
            if min_b == mid_b: #nothing better
                break
            min_b = mid_b
        else:
            if max_b == mid_b: #nothing better
                break
            max_b = mid_b
        # print("amtB", mid_b, "c1", c1.price(tokenTrade), "c2", c2.price(tokenTrade), tx_arb1['qty'], tx_arb2['qty'])

    if switch_chains:
        return last_successful_tx2, last_successful_tx1
    else:
        return last_successful_tx1, last_successful_tx2

def make_trade(poolA, poolB, account, prefs, optimal=False, static_value=1., scaled=False, percent_optimal=1., slippage_percentage=None, accounts=None, debug=False):
    """generate a swap transaction based on the user's preference
    and the top of block pool price on the chain

    Args:
        chain (Chain): chain the swap will occur on
        sndr (string): sender name
        prefs (list of numbers): preferences of sndr for the token. (must add up to 2.0)
        optimal (bool, optional): Use the optimal trade to increase net utility. Defaults to False.
        static_value (number, optional): Max amount to trade in the optimal direction. Defaults to 1..
        percent_optimal (number, optional): Percentage of the optimal trade. Defaults to 1..
        slippage (number, optional): slippage percentage of other token to receive
        accounts (dict of str -> Account, optional): accounts associated with users

    Returns:
        dict: swap transaction
    """
    assert percent_optimal <= 1.0 and percent_optimal >= 0

    #token A: apples, token B: dollars
    if prefs[1] == 0:
        my_price = np.inf
    else:
        my_price =  prefs[0] / prefs[1] #utils/apple / utils/dollar -> dollars/apple
    pool_price = poolB/poolA #dollars/apple 
    sndr = account.username
    account = account.tokens
    # print("pool_price", pool_price, "my_price", my_price)
    if pool_price > my_price: #trade A for B
        if optimal == False:
            if scaled:
                scale_val = logistic_function(my_price/pool_price, 1, 1.0, 0.0)
                if debug:
                    print(f"scaled value * {static_value} logistic_function({pool_price/my_price})={scale_val}) = {static_value* scale_val}")
                static_value = scale_val*static_value
            qty = static_value
        else:
            if scaled:
                percent_optimal = percent_optimal*my_price
            optimal_val = optimal_trade_amt(poolB, poolA, prefs[1], prefs[0], account[1], account[0])
            if debug:
                print("optimal_val", optimal_val, "*", percent_optimal, "=", optimal_val*percent_optimal )
            optimal_val *= percent_optimal
            qty = optimal_val
        # qty = min(qty, account[0])
        optimal_slip = prefs[0]* qty / prefs[1] # slippage s.t. new net utility will equal old net utility
        if slippage_percentage is not None:
            slip = -max(optimal_slip, pool_swap(poolA, poolB, qty)*slippage_percentage)
        else:
            slip = -optimal_slip

        if debug:
            print(f"{sndr} tx: qty {qty} slip {slip} prefs{prefs} tokens {account} pool {[poolA, poolB]}")
        assert abs(slip) <= abs(pool_swap(poolA, poolB, qty))
    elif my_price > pool_price: #trade B for A
        if optimal == False:
            if scaled:
                scale_val = logistic_function(my_price/pool_price, 1, 1.0, 0.0)
                if debug:
                    print(f"scaled value * {static_value} logistic_function({my_price/pool_price})={scale_val}) = {static_value* scale_val}")
                static_value = scale_val*static_value
            qty = static_value
        else:
            if scaled:
                percent_optimal = percent_optimal*my_price
            optimal_val = optimal_trade_amt(poolA, poolB, prefs[0], prefs[1], account[0], account[1])
            if debug:
                print("optimal_val", optimal_val, "*", percent_optimal, "=", optimal_val*percent_optimal)
            optimal_val *= percent_optimal
            qty = optimal_val
        # qty = -min(qty, account[1])
        qty = -qty
        slip = -prefs[1]* qty / prefs[0] # slippage s.t. new net utility will equal old net utility
        if slippage_percentage is not None:
            slip = max(-slip, pool_swap(poolB, poolA, abs(qty))*slippage_percentage)
        if debug:
            print(f"{sndr} tx: qty {qty} slip {slip} prefs{prefs} tokens {account} pool {[poolA, poolB]}")

        assert abs(slip) <= abs(pool_swap(poolB, poolA, abs(qty)))
    else:
        qty = 0
        slip = 0
    # print("values",a,prefs[0],abs(qty))
    # with open("data",'a') as f:
    #     f.write(f"{a},{prefs[0]},{abs(qty)}\n")
    return create_swap(sndr, qty, slip)
