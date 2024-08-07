import math
import random
import os
import numpy as np
import json
import copy
import itertools
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from collections import defaultdict 

from utils import *

figure_num = 0
random_walk_stats = {}

mev_share_scenario = "mev_share"
prof_share_scenario = "prof_share"
prof_scenario = "prof"
all_scenarios = [prof_scenario, prof_share_scenario, mev_share_scenario]




class Account():
    def __init__(self, tokens, username=None):
        self.username = username
        self.tokens = tokens
    def __str__(self):
        if self.username is not None:
            return f"{self.username}: {self.tokens}"
        return f"{self.tokens}"

    def transfer(self, qty):
        if qty > 0:
            self.tokens[0] += qty 
        else:
            self.tokens[1] -= qty 

    def __sub__(self, other):
        return Account([self.tokens[0]-other.tokens[0], self.tokens[1] - other.tokens[1]])


    def __add__(self, other):
        return Account([self.tokens[0]+other.tokens[0], self.tokens[1]+other.tokens[1]])


    def __truediv__(self, other):
        return Account([self.tokens[0]/other, self.tokens[1]/other])

    def __mul__(self, other):
        return Account([self.tokens[0]*other, self.tokens[1]*other])



class Chain():
    def __init__(self, poolA=1000., poolB=1000., accounts=None, chainid="", fee=0.0, liquidity=None, static=None):
        self.lp = liquidity
        if liquidity is not None:
            self.accounts['lp'] = self.lp
        self.chainid = chainid
        self.poolA = poolA
        self.poolB = poolB
        if static is not None:
            self.poolB = poolA/static
        self.static = static
        self.fee = fee
        if accounts is None:
            self.accounts = {"alice": Account([100.,100.],"alice"),
                             "bob":   Account([100.,100.], "bob"),
                            }
        else:
            self.accounts = accounts

    def apply(self, tx, debug=True, accounts=None):
        # Apply the transaction, updating the pool price
        if (tx['type'] == 'swap'):
            if accounts is None:
                account = self.accounts[tx["sndr"]].tokens
            else:
                account = accounts[tx["sndr"]].tokens

            if tx['qty'] >= 0:
                # Sell qty of tokenA, buy at least rsv of tokenB
                amtA = tx['qty']
                if self.static is None:
                    amtB = pool_swap(self.poolA, self.poolB, (1.0-self.fee) * amtA)
                    require(self.poolB - amtB >= 0,'exhausts pool')
                else:
                    amtB = amtA/self.static

                require(account[0] >= amtA, f'not enough balance for trade {tx} {amtA} account {account}')
                require(amtB >= 0)
                require(amtB >= -tx['rsv'] or tx['rsv'] == 0, f"slippage exceeded amtB:{amtB}, -rsv:{-tx['rsv']}")

                if debug:
                    print(self.chainid,"\t",tx['sndr'], "sell", amtA, "of A and gets", amtB, "of B with slippage", -tx['rsv'], "<=", amtB)
                amtB = -amtB
                if self.lp is not None:
                    self.lp.tokens[0] += self.fee*amtA
                    if debug and self.fee > 0.0:
                        print(self.chainid,"\t", "liquidity provider receive", self.fee*amtA, "of token A:", self.lp) 
            else:
                # Sell qty of tokenB, buy at least rsv of tokenA
                amtB = -tx['qty']
                if self.static is None:
                    amtA = pool_swap(self.poolB, self.poolA, amtB)
                    require(self.poolA + amtA >= 0, 'exhausts pool')
                else:
                    amtA = amtB*self.static

                require(account[1] >= amtB, f'not enough balance for trade {tx} {amtB} account {account}')
                require(amtA >= 0)
                require(amtA >= tx['rsv'] or tx['rsv'] == 0, f"slippage exceeded amtA:{amtA}, rsv:{tx['rsv']}")

                if debug:
                    print(self.chainid,"\t",tx['sndr'], "sell", amtB, "of B and gets", amtA, "of A with slippage",  tx['rsv'], "<=" , amtA)

                amtA = -amtA
                if self.lp is not None:
                    self.lp.tokens[1] += self.fee*amtB
                    if debug and self.fee > 0.0:
                        print(self.chainid,"\t", "liquidity provider receive",self.fee*amtB, "of token B:", self.lp) 
            if self.static is None:
                self.poolA += amtA
                self.poolB += amtB
            account[0] -= amtA
            account[1] -= amtB
        else:
            raise ValueError("unknown tx type")

    def __str__(self):
        return f"PoolA: {self.poolA} PoolB: {self.poolB} accounts: {[str(self.accounts[acc]) for acc in self.accounts]}"

    def price(self, token):
        if token == 'A' or token == 'a':
            return self.poolA/self.poolB
        else:
            return self.poolB/self.poolA
    
    def product(self):
        return self.poolA*self.poolB

class Experiment():

    def __init__(self, param_type=random_type,direction_type=diff_type, num_iters=10, users_range=list(range(2,101)), arbitrage_type=cexdex_arbitrage, params=None, file_dir="data",debug=False, extra_params={max_demand_scale_type:4, max_demand_batch_type: 5}):
        if not os.path.isdir(file_dir):
            os.mkdir(file_dir)
        self.param_type=param_type
        self.direction_type=direction_type
        self.num_iters=num_iters
        self.users_range=users_range
        if params is None:
            assert param_type == random_type
            self.params = [0.25, .5, .75, 1.0]
            #TODO set defaults for other param_types
        else:
            self.params=params
        self.extra_params=extra_params
        assert arbitrage_type in [cexdex_arbitrage, dexdex_arbitrage]
        self.arbitrage_type = arbitrage_type
        self.debug = debug
        self.quiet = False
        self.pool_liquidity = 10e7 
        self.user_portfolio = 100
        self.user_slippage = .8
        self.arbitrager_portfolio = 10e7
        self.kickback_percentage = .9
        self.file_location = os.path.join(file_dir, f"{arbitrage_type}_{param_type}_{num_iters}iters_") 
        if len(users_range) > 1:
            self.file_location += f"{users_range[0]}to{users_range[-1]}-{len(users_range)}users_"
        else:
            self.file_location += f"{users_range[0]}users_"
        if len(params) > 1:
            self.file_location += f"{params[0]}to{params[-1]}-{len(params)}params"
        else:
            self.file_location += f"{params[0]}params"
    
        print("self.file_location", self.file_location)
        self.all_stats = {}

    def set_random_seed(seed):
        np.random.seed(seed)
        random.seed(seed)       

    def gen_preferences(self, num_users, params=None, param_type=None, direction_type=None):
        """get users preferences list

        Args:
            num_users (int): number of users in the block
            params (list of ints): list parameters that determine how the preference change for the param_type
            param_type (string): how to generate user preferences
                rndm: each user has a different random weighted preference for token A/token B
                same: users have same random weighted preference for token A/token B
                simple: users have preference for token A or token B
                random_walk_range: users have a random preference for token A or token B, with a max price deviation
                random_walk_range_relative: same as random_walk_range except max price deviation weighted to the number of users
                random_walk_scaled: users have a random preference for token A or token B, price deviation determines the likelihood of which token the user prefers, (i.e. if everyone before me prefered token A then the next user is more likely to prefer token B)  
                net_demand: users have a random preference for token A or token B, sets demand (i.e. abs(number of users who prefer token A  - number of users who prefer token B))
                max_demand: users have a random preference for token A or token B, caps the max demand (i.e. abs(number of users who prefer token A  - number of users who prefer token B))
                max_demand_scale: same as max_demand except the max demand is scaled by number of users 
                max_demand_batch_size: same as max_demand except user are batched into groups to calculate the max demand (set the max_demand scale and batch size using extra_params args)
        Returns:
            list of <num_users> preference tuples of the format ([(preferenceA_i, preferenceB_i) for i in range(num_users)]) with sum(preferenceA, preferenceB) == 2
        """
        if params is None:
            params = self.params
        if param_type is None:
            param_type = self.param_type  
        if direction_type is None:
            direction_type = self.direction_type
        if direction_type not in [same_type, diff_type]:
            raise Exception(f"direction_type {direction_type} not found")

        if param_type == random_type:
            if direction_type == same_type:
                v = [logistic_function(np.random.normal(loc=0.0, scale=scale), .5, 1., 1) for scale in params]
                #every user gets a same random preference
                prefs = [[v[i] for _ in range(num_users)] for i in range(len(v))]
            else: 
                #every user gets a different random preference
                prefs = [[logistic_function(np.random.normal(loc=0.0, scale=scale), .5, 1., 1) for _ in range(num_users)] for scale in params]
        elif param_type == simple_type: 
            assert params is None
            if direction_type == same_type:
                #users have same preference for token a or token b
                v = random.choice([0.0,2.0])
                prefs = [[  v for _ in range(num_users) ]] 
            else: 
                #users randomly have a preference for token a or token b
                assert params is None
                prefs = [[ random.choice([0.0,2.0]) for _ in range(num_users) ]] 
        elif param_type == random_walk_range_type: 
            #random walk with max step
            prefs = [limited_random_walk_range(scale, -scale, num_users, lambda x: x>1.0, lambda : random.choice([0.0,2.0]), num_users=num_users, param=scale) for scale in params]
        elif param_type == random_walk_range_relative_type: 
            #random walk with max step that is relative to the number of users i.e. number of preferences
            prefs = [ limited_random_walk_range(max(scale*num_users, 1.0), min(-scale*num_users, -1.0), num_users, lambda x: x>1.0, lambda : random.choice([0.0,2.0]), num_users=num_users, param=scale) for scale in params]
        elif param_type == random_walk_scaled_type: 
            #random walk with likelihood of next step dependent on distance from start
            def random_foo(step, scale):
                p = logistic_function(step, scale, .5, .5)
                prob = [p, 1-p]
                val = random.choices([0.0,2.0], prob)[0]
                return val
            prefs = [limited_random_walk_scaled(num_users, lambda x: x>1.0, random_foo, scale) for scale in params]
        elif param_type == net_demand_type:
            # limit user preferences based on demand as defined in Budish et. al.
            def net_demand_sample(num_users, net_demand):
                sequence = [2.0 for _ in range(int(net_demand))]
                sequence += [0.0 for _ in range(num_users-int(net_demand))]
                random.shuffle(sequence)
                return sequence
            prefs = [net_demand_sample(num_users, max(scale*math.sqrt(num_users), 1.0)) for scale in params]
        elif param_type == max_demand_type: 
            # limit user preferences based on max demand as defined in Budish et. al.
            def direction_foo(x):
                if x > 1.0:
                    return 1
                else: 
                    return -1
            prefs = [max_demand_sample(max(scale*math.sqrt(num_users), 1.0), min(-scale*math.sqrt(num_users), -1.0), num_users, direction_foo, lambda n: [random.choice([0.0,2.0]) for _ in range(n)], param=scale ) for scale in params]
        elif param_type == max_demand_scale_type: 
            # limit user preferences based on max demand as defined in Budish et. al. and max random walk step relative to the number of users
            # i.e. max_demand and random_walk_range_relative combined
            def direction_foo(x):
                if x > 1.0:
                    return 1
                else: 
                    return -1
            def make_random_foo(max_scale):
                return lambda nu: limited_random_walk_range(max(max_scale*nu, 1.0), min(-max_scale*nu, -1.0), nu, lambda x: x>1.0, lambda : random.choice([0.0,2.0]), num_users=nu, param=max_scale)
            max_demand_scale = self.extra_params[max_demand_scale_type]
            prefs = [ max_demand_sample(max(max_demand_scale*math.sqrt(num_users), 1.0), min(-max_demand_scale*math.sqrt(num_users), -1.0), num_users, direction_foo, make_random_foo(scale), param=max_demand_scale ) for scale in params]
        elif param_type == max_demand_batch_type:
            # limit user preferences based on max demand as defined in Budish et. al.
            # with smaller set batch sizes
            batch_size = self.extra_params[max_demand_batch_type]
            def direction_foo(x):
                if x > 1.0:
                    return 1
                else: 
                    return -1
            def construct_batches(scale, nu, batch_size):
                num_batches = int(nu/batch_size) + 1
                pref = []
                for _ in range(num_batches):
                    pref += max_demand_sample(max(scale*math.sqrt(batch_size), 1.0), min(-scale*math.sqrt(batch_size), -1.0), batch_size, direction_foo, lambda n: [random.choice([0.0,2.0]) for _ in range(n)], num_users=num_users, param=scale)
                return pref[:nu]
            prefs = [construct_batches(scale, num_users, batch_size)  for scale in params]
        else:
            raise Exception(f"param_type {param_type} not found")
        if params is None:
            assert len(prefs) == 1
        else:
            assert len(prefs) == len(params)
        assert len(prefs[0]) == num_users
        assert np.max(prefs) <= 2.0
        assert np.min(prefs) >= 0.0

        if self.debug:
            print("prefs", prefs)
            print("avg preference", np.average(prefs, axis = 1))
            print("std preference", np.std(prefs, axis = 1))
            print("max preference", np.max(prefs, axis = 1))
            print("min preference", np.min(prefs, axis = 1))
        get_pref = lambda p : [p, 2.0-p]
        prefs = [[get_pref(n) for n in p] for p in prefs]
        return prefs

    def setup_accounts(self, scenario, num_users):
        """setup user accounts for an iteration of the experiment

        Args:
            scenario (string): the scenario for the experiment
                mev_share: arbitrage backrunning occurs after each user transaction
                prof_share: arbitrage occurs after the block of user transactions
                prof: no arbitragers
            num_users (int): number of users in the block
        Returns:
            list of <num_users> preference tuples of the format ([(preference_A, preference_B)]) with sum(preference_A, preference_B) == 2
        """
        user_accounts = {}
        arbi_accounts = {}
        for i in range(num_users): 
            username = Experiment.get_username(scenario,num_users,i)
            user_tokens = [self.user_portfolio, self.user_portfolio]
            user_accounts[username] = Account(user_tokens, username)
            arbi_tokens = [self.arbitrager_portfolio, self.arbitrager_portfolio]
            if scenario == mev_share_scenario: #each user has their own arbitrager
                arbi_username=Experiment.get_arbiname(scenario, num_users, i)
                arbi_accounts[arbi_username] = Account(arbi_tokens, arbi_username)
            elif scenario == prof_share_scenario: #one arbitrager for all
                arbi_username=Experiment.get_arbiname(scenario, num_users)
                arbi_accounts[arbi_username] = Account(arbi_tokens, arbi_username)  
        return user_accounts,arbi_accounts

    def get_username(scenario, name, index):
        return f"user_{scenario}_{name}_{index}"

    def get_arbiname(scenario, name, index=None):
        if scenario == mev_share_scenario:
            assert index is not None
            return f"arbi_{scenario}_{name}_{index}"
        elif scenario == prof_share_scenario:
            return f"arbi_{scenario}_{name}"
        elif scenario == prof_scenario:
            raise Exception(f"{prof_scenario} has no arbitrager")
        else:
            raise Exception(f"{scenario} scenario not found")

    def get_pools(self, user_accounts, arbi_accounts):
        accounts = {**user_accounts, **arbi_accounts}
        if self.arbitrage_type == cexdex_arbitrage:
            chain1 = Chain(poolA=self.pool_liquidity, poolB=self.pool_liquidity, accounts=accounts, chainid=f"chain1")
            chain2 = Chain(static=1.0, accounts=arbi_accounts, chainid=f"chain2") #only arbitragers use CEX
        elif self.arbitrage_type == dexdex_arbitrage:
            chain1 = Chain(poolA=self.pool_liquidity, poolB=self.pool_liquidity, accounts=accounts, chainid=f"chain1")
            chain2 = Chain(poolA=self.pool_liquidity, poolB=self.pool_liquidity, accounts=arbi_accounts, chainid=f"chain2")
        else:
            raise Exception(f"arbitrage_type {self.arbitrage_type} not found")
        return chain1, chain2

    def get_user_util_old(self, pref):
        if self.param_type == random_type:
            user_tokens = [self.user_portfolio, self.user_portfolio]
            return utility(pref, user_tokens)
        else:
            return 0

    def get_user_util_new(self, username, accounts, pref):
        if self.param_type == random_type:
            return utility(pref, accounts[username].tokens)
        else:
            return sum(accounts[username].tokens)

    def get_arbi_portfolio_old(self):
        arbi_tokens = [self.arbitrager_portfolio, self.arbitrager_portfolio]
        return Account(arbi_tokens)

    def get_arbi_portfolios_diff(self, scenario, accounts, num_users):
        arbi_portfoli_diff = []
        if scenario == mev_share_scenario:
            for i in range(num_users):
                arbi_username = Experiment.get_arbiname(scenario, num_users, i)
                diff = accounts[arbi_username] - self.get_arbi_portfolio_old()
                arbi_portfoli_diff.append(diff)
                if self.debug:
                    print("profit", arbi_username, self.get_arbi_portfolio_old().tokens, "->", accounts[arbi_username].tokens)
        elif scenario == prof_share_scenario:
            arbi_username = Experiment.get_arbiname(scenario, num_users)
            diff = accounts[arbi_username] - self.get_arbi_portfolio_old()
            token_split = diff/num_users
            for i in range(num_users):
                arbi_portfoli_diff.append(token_split)
            if self.debug:
                print("profit", arbi_username, self.get_arbi_portfolio_old().tokens, "->", accounts[arbi_username].tokens)
        return arbi_portfoli_diff

    def get_stat_name(self, stat_name):
        if stat_name == random_walk_stat:
            stat_name = f"{random_walk_stat}_{get_random_walk_stat_name(self.param_type)}"
        return stat_name

    def get_stat_filename(self, stat_name):
        return self.file_location + f"_{self.get_stat_name(stat_name)}.csv"

    def make_txs_from_preferences(self, preferences, scenario, accounts):
        num_users = len(preferences)
        ordered_txs = []
        for i in range(num_users): 
            pref = preferences[i]
            username = Experiment.get_username(scenario, num_users, i)
            if self.param_type == random_type:
                tx = make_trade(self.pool_liquidity, self.pool_liquidity, accounts[username], pref, optimal=False,
                                    static_value=self.user_portfolio, scaled=True, debug=self.debug)
            else:
                tx = make_trade(self.pool_liquidity, self.pool_liquidity, accounts[username], pref, optimal=False,
                                    static_value=self.user_portfolio, slippage_percentage=self.user_slippage, debug=self.debug)
            ordered_txs.append(tx)
        return ordered_txs

    def _execute_transactions_mevshare(self, ordered_txs, chain1, chain2):
        num_users = len(ordered_txs)
        for i in range(num_users):
            tx = ordered_txs[i]
            try:
                #execute user transaction
                chain1.apply(tx, debug=self.debug)

                #execute arbitrage after every user tx
                arbi_username = Experiment.get_arbiname(mev_share_scenario, num_users, i)
                if abs(chain1.price('A')- chain2.price('A')) < abs(chain1.price('B')- chain2.price('B')):
                    tx1, tx2 = optimal_arbitrage_algebra(chain1, chain2, arbi_username)
                else:
                    tx1, tx2 = optimal_arbitrage_algebra(chain1, chain2, arbi_username)
                chain1.apply(tx1, debug=self.debug)
                chain2.apply(tx2, debug=self.debug)
            except ExecutionException as e:
                raise e

    def _execute_transactions_profshare(self, ordered_txs, chain1, chain2):
        num_users = len(ordered_txs)
        #execute transactions
        for i in range(num_users):
            tx = ordered_txs[i]
            try:
                #execute user transaction
                chain1.apply(tx, debug=self.debug)
            except ExecutionException as e:
                raise e

        #execute arbitrage after all user tx
        arbi_username = Experiment.get_arbiname(prof_share_scenario, num_users)
        if abs(chain1.price('A')- chain2.price('A')) < abs(chain1.price('B')- chain2.price('B')):
            tx1, tx2 = optimal_arbitrage_algebra(chain1, chain2, arbi_username)
        else:
            tx1, tx2 = optimal_arbitrage_algebra(chain1, chain2, arbi_username)
        try:
            chain1.apply(tx1, debug=self.debug)
            chain2.apply(tx2, debug=self.debug)
        except ExecutionException as e:
            raise e

    def _execute_transactions_prof(self, ordered_txs, chain1, chain2):
        num_users = len(ordered_txs)
        #execute transactions
        for i in range(num_users):
            tx = ordered_txs[i]
            try:
                #execute user transaction
                chain1.apply(tx, debug=self.debug)
            except ExecutionException as e:
                raise e

    def execute_transactions(self, scenario, ordered_txs, chain1, chain2):
        if scenario == mev_share_scenario:
            self._execute_transactions_mevshare(ordered_txs, chain1, chain2)
        elif scenario == prof_share_scenario:
            self._execute_transactions_profshare(ordered_txs, chain1, chain2)
        elif scenario == prof_scenario:
            self._execute_transactions_prof(ordered_txs, chain1, chain2)
        else:
            raise Exception(f"cannot execute transaction for scenario {scenario} does not exist")

    def execute_kickbacks(self, scenario, accounts, num_users):
        arbi_portfoli_diff = self.get_arbi_portfolios_diff(scenario, accounts, num_users)
        for i in range(num_users):
            username = Experiment.get_username(scenario, num_users, i)
            if scenario in [prof_share_scenario, mev_share_scenario]:
                accounts[username] += arbi_portfoli_diff[i]*self.kickback_percentage
            elif scenario == prof_scenario:
                pass
            else:
                raise Exception(f"cannot execute kickback for scenario {scenario} does not exist")

    def run_iteration(self, scenario, num_users, iter_num, preferences, param, all_stats=False):
        if not self.quiet:
            print(f"-----------scenario: {scenario} | total users: {num_users} | volatitiy: {param} ({iter_num})-----------")

        user_accounts,arbi_accounts = self.setup_accounts(scenario, num_users)
        accounts = {**user_accounts, **arbi_accounts}
        chain1, chain2 = self.get_pools(user_accounts, arbi_accounts)

        user_util_old = [self.get_user_util_old(preferences[i]) for i in range(num_users)]
        user_util_diff = [None for _ in range(num_users)]
        user_util_diff_kickback = [None for _ in range(num_users)]


        ordered_txs = self.make_txs_from_preferences(preferences, scenario, accounts)

        self.execute_transactions(scenario, ordered_txs, chain1, chain2)

        if all_stats:
            trade_amounts = [tx['qty'] for tx in ordered_txs]
            arbi_portfolio_diff = [sum(x.tokens) for x in self.get_arbi_portfolios_diff(scenario, accounts, num_users)]
            preferece_amounts = [p[0] for p in preferences]
            #calculate utility for user before kickbacks
            for i in range(num_users):
                username = Experiment.get_username(scenario,num_users,i)
                user_util_new = self.get_user_util_new(username, accounts, preferences[i])
                user_util_diff[i] = user_util_new - user_util_old[i]
                if self.debug:
                    print("utility b4 kickback", username, user_util_old[i], "->", user_util_new)

        self.execute_kickbacks(scenario, accounts, num_users)

        #calculate utility of user with kickbacks
        for i in range(num_users): 
            username = Experiment.get_username(scenario,num_users,i)                      
            user_util_new_kickback = self.get_user_util_new(username, accounts, preferences[i])
            user_util_diff_kickback[i] = user_util_new_kickback - user_util_old[i]
            if self.debug and scenario != prof_scenario:
                print("utility w/ kickback", username, user_util_old[i], "->", user_util_new_kickback)

        #save raw data
        for i in range(num_users):
            with open(self.get_stat_filename(user_util_stat), "a") as f:
                f.write(f"{num_users},{i},{user_util_diff_kickback[i]},{iter_num},{scenario},{param}\n")

        if all_stats:
            random_walk_stats_data = [get_random_walk_stats(self.param_type, param,num_users, iter_num)]
            stats = [(trade_amount_stat, trade_amounts), 
                     (preferences_stat, preferece_amounts), 
                     (arbitrage_amount_stat, arbi_portfolio_diff), 
                     (user_util_nk_stat, user_util_diff),
                     (random_walk_stat, random_walk_stats_data)]
            for (stat_name, arr) in stats:
                for i in range(len(arr)):
                    with open(self.get_stat_filename(stat_name), "a") as f:
                        f.write(f"{num_users},{i},{arr[i]},{iter_num},{scenario},{param}\n")

    def run_all(self, save_state=True, all_stats=False):
        progress = None
        iter_start = -1
        user_start = -1
        param_start = -1
        scenario_start = -1

        if save_state:
            if os.path.isfile(self.file_location + f"_progress.txt"):
                with open(self.file_location + f"_progress.txt") as f:
                    progress = f.read().split(",")
            if progress is not None:
                iter_start = int(progress[0])
                user_start = int(progress[1])
                param_start = int(progress[2])
                scenario_start = int(progress[3])
                print(f"restarting from iter {iter_start}, user {self.users_range[user_start]} param {self.params[param_start]} scenario {all_scenarios[scenario_start]}")

        Experiment.set_random_seed(0) #for reproducibility
        for iter_num in range(self.num_iters):
            for user_index in range(len(self.users_range)):
                num_users = self.users_range[user_index]
                preferences = self.gen_preferences(num_users)    
                for param_index in range(len(self.params)):           
                    for scenario_index in range(len(all_scenarios)):
                        if iter_num <= iter_start or user_index <= user_start or param_index <= param_start or scenario_index <= scenario_start: 
                            #still want to generate the preferences for reproducibility with the random seed
                            continue  
                        scenario =  all_scenarios[scenario_index]
                        #all scenarios use the same preferences
                        self.run_iteration(scenario, num_users, iter_num, preferences[param_index], self.params[param_index], all_stats=all_stats) 
                        if save_state:
                            with open(self.file_location + f"_progress.txt", "w") as f:
                                f.write(f"{iter_num},{user_index},{param_index},{scenario_index}")
        
    def print_stats(self, all_stats=True, stat_names=None):
        stat_filenames = {}
        if stat_names is not None:
            for n in stat_names:
                stat_filenames[n] = self.get_stat_filename(n)
        elif all_stats:
            for stat_name in all_stat_names:
                stat_filenames[stat_name] = self.get_stat_filename(stat_name)
        else:
            stat_filenames[user_util_stat] = self.get_stat_filename(user_util_stat)
        print("\n\n\n----------------------------------------results----------------------------------------")
        
        dtypes = {
            "num_users": 'int',
            "user_index": 'int',
            "data":'float',
            "iter_num": 'int',
            "scenario": 'string',
            "param": 'float'
        }
        colnames= list(dtypes.keys())
        print("stat_filenames =", json.dumps(stat_filenames, indent=2))
        for n in stat_filenames:
            print()
            print("~~~~~~~~~~~",self.get_stat_name(n),"~~~~~~~~~~~")
            summary = pd.DataFrame(columns=["mean", "std", "min", "max"])
            df  = pd.read_csv(stat_filenames[n], dtype=dtypes, names=colnames, header=None)
            num_users_group = df.groupby(['num_users', "param", "scenario"])
            summary['mean'] = num_users_group['data'].mean()
            summary['std'] = num_users_group['data'].std()
            summary['min'] = num_users_group['data'].min()
            summary['max'] = num_users_group['data'].max()
            print(summary)

    def parse_data(self, stat_name=user_util_stat, by="param", merge=False):
        assert by in ["num_users", "param"]
        if merge:
            data_filename = self.file_location + f"_{stat_name}_{by}_merged.json"
        else:
            data_filename=self.file_location + f"_{stat_name}_{by}.json"
        if os.path.isfile(data_filename):
            return json.load(open(data_filename))

        dtypes = {
            "num_users": 'int',
            "user_index": 'int',
            "data":'float',
            "iter_num": 'int',
            "scenario": 'string',
            "param": 'float'
        }
        colnames= list(dtypes.keys())
        parsed_data = {}
        stat_filename = self.get_stat_filename(stat_name)
        parsed_data = {}
        df  = pd.read_csv(stat_filename, dtype=dtypes, names=colnames, header=None)
        nb_name = "num_users" if by == "param" else "param"

        nb_min = df[nb_name].min()
        nb_max = df[nb_name].max()
        if merge:
            if nb_min != nb_max:
                parsed_data["nb_range"] = f"{nb_min} to {nb_max} ({len(df[nb_name].unique())})"
            else:
                parsed_data["nb_range"] = f"{nb_min}"
            for scenario in all_scenarios:
                parsed_data[scenario] = {}
                parsed_data[scenario]["xs"] = list(df[by].unique())
                by_group = df[df['scenario'] == scenario].groupby(by)
                averages = by_group['data'].mean()
                stds = by_group['data'].std()
                parsed_data[scenario]["averages"] = list(averages)
                parsed_data[scenario]["stds"] = list(stds)
        else:
            num_users_group = df.groupby(['num_users', "param", "scenario"], group_keys=False)
            summary = pd.DataFrame(columns=["mean", "std", "scenario", "num_users", "param"])
            summary['mean'] = num_users_group[['data']].mean()
            summary['std'] = num_users_group[['data']].std()
            summary['scenario'] = num_users_group[['scenario']].first()
            summary['num_users'] = num_users_group[['num_users']].first()
            summary['param'] = num_users_group[['param']].first()
            not_by_group = summary[nb_name].unique()
            parsed_data = {}
            print("not_by_group", not_by_group)
            for nb in not_by_group:
                nb = int(nb)
                for scenario in all_scenarios:
                    if scenario not in parsed_data:
                        parsed_data[scenario] = {}
                    parsed_data[scenario][nb] = {}
                    parsed_data[scenario][nb]["xs"] = list(summary[by].unique())
                    by_group = summary[(summary['scenario'] == scenario) & (summary[nb_name] == nb)]
                    averages = by_group['mean'].apply(lambda x: x)
                    stds = by_group['std'].apply(lambda x: x)
                    parsed_data[scenario][nb]["averages"] = list(averages)
                    parsed_data[scenario][nb]["stds"] = list(stds)
            print()
        json.dump(parsed_data, open(data_filename, "w"), indent=2)
        print(data_filename)
        return parsed_data

    def plot_stats_line(self, all_stats=False, stat_names=None, by="param", merge=False):
        assert by in ["num_users", "param"]
        stat_filenames = {}
        if stat_names is not None:
            for n in stat_names:
                stat_filenames[n] = self.get_stat_filename(n)
        elif all_stats:
            for stat_name in all_stat_names:
                stat_filenames[stat_name] = self.get_stat_filename(stat_name)
        else:
            stat_filenames[user_util_stat] = self.get_stat_filename(user_util_stat)

        global figure_num
        for n in stat_filenames:
            stat_data = self.parse_data(stat_name=n, by=by, merge=merge)
            nb_name = "num_users" if by == "param" else "param"
            if merge:
                fig = plt.figure(figure_num)
                figure_num+=1
                ax = fig.add_subplot()
                min_y =  np.inf
                max_y = -np.inf
                for scenario in all_scenarios:
                    xs = stat_data[scenario]["xs"]
                    averages = np.array(stat_data[scenario]["averages"])
                    if len(averages) == 0:
                        continue
                    stds = np.array(stat_data[scenario]["stds"])
                    min_y = min(min_y, min(np.array(averages)-np.array(stds)))
                    max_y = max(max_y, max(averages+stds))
                    ax.errorbar(xs, averages, yerr=stds,  label = f"{scenario}")
                if by == "param":
                    ax.set_xlabel(param_type_xlabels[self.param_type])
                else:
                    ax.set_xlabel(by)
                ax.set_ylabel(n)
                range_y = max_y - min_y
                if np.isclose(range_y, 0):
                    range_y = max_y*.5
                ax.set_ylim(min_y-(range_y/2), max_y+(range_y/2))
                plt.legend()
                plt.title(f'{stat_data["nb_range"]} {nb_name}\n{self.arbitrage_type} arbitrage, {self.param_type} {self.num_iters} iterations')
                fig_filename = self.file_location + f"_by{by}_line.png"
                fig.savefig(fig_filename)
                print(fig_filename)
            else:         
                print("nb", list(stat_data[all_scenarios[0]].keys()))       
                for nb in stat_data[all_scenarios[0]].keys():
                    fig = plt.figure(figure_num)
                    figure_num+=1
                    ax = fig.add_subplot()
                    min_y =  np.inf
                    max_y = -np.inf
                    for scenario in all_scenarios:
                        xs = stat_data[scenario][nb]["xs"]
                        averages = np.array(stat_data[scenario][nb]["averages"])
                        if len(averages) == 0:
                            continue
                        stds = np.array(stat_data[scenario][nb]["stds"])
                        min_y = min(min_y, min(averages-stds))
                        max_y = max(max_y, max(averages+stds))
                        ax.errorbar(xs, averages, yerr=stds,  label = f"{scenario}")
                    if by == "param":
                        ax.set_xlabel(param_type_xlabels[self.param_type])
                    else:
                        ax.set_xlabel(by)
                    ax.set_ylabel(n)
                    range_y = max_y - min_y
                    if np.isclose(range_y, 0):
                        range_y = max_y*.5
                    ax.set_ylim(min_y-(range_y/2), max_y+(range_y/2))
                    plt.legend()
                    plt.title(f'{nb} {nb_name}\n{self.arbitrage_type} arbitrage, {self.param_type} preference type, {self.num_iters} iterations')
                    fig_filename = self.file_location + f"_{nb}{nb_name}_by{by}_line.png"
                    fig.savefig(fig_filename)
                    print(fig_filename)
            
    def plot_stats_bar(self, all_stats=False, stat_names=None, by="param", merge=False):
        assert by in ["num_users", "param"]
        stat_filenames = {}
        if stat_names is not None:
            for n in stat_names:
                stat_filenames[n] = self.get_stat_filename(n)
        elif all_stats:
            for stat_name in all_stat_names:
                stat_filenames[stat_name] = self.get_stat_filename(stat_name)
        else:
            stat_filenames[user_util_stat] = self.get_stat_filename(user_util_stat)
 
        global figure_num
        for n in stat_filenames:
            stat_data = self.parse_data(stat_name=n, by=by, merge=merge)
            nb_name = "num_users" if by == "param" else "param"            
            width = 1/(len(all_scenarios)+1)
            if merge:
                fig = plt.figure(figure_num)
                figure_num+=1
                ax = fig.add_subplot()
                min_y =  np.inf
                max_y = -np.inf
                multiplier=1
                for scenario in all_scenarios:
                    averages = np.array(stat_data[scenario]["averages"])
                    if len(averages) == 0:
                        continue
                    xlabels = stat_data[scenario]["xs"]
                    xs = np.arange(len(xlabels))
                    stds = np.array(stat_data[scenario]["stds"])
                    min_y = min(min_y, min(averages-stds))
                    max_y = max(max_y, max(averages+stds))
                    offset = width*multiplier
                    multiplier+=1
                    ax.bar(xs + offset, averages, width, label=scenario, yerr=stds)
                ax.set_xticks(xs + .5, xlabels)
                if by == "param":
                    ax.set_xlabel(param_type_xlabels[self.param_type])
                else:
                    ax.set_xlabel(by)
                ax.set_ylabel(n)
                range_y = max_y - min_y
                if np.isclose(range_y, 0):
                    range_y = max_y*.5
                ax.set_ylim(min_y-(range_y/2), max_y+(range_y/2))
                plt.legend()
                plt.title(f'{stat_data["nb_range"]} {nb_name}\n{self.arbitrage_type} arbitrage, {self.param_type} {self.num_iters} iterations')
                fig_filename = self.file_location + f"_by{by}_{n}_bar.png"
                fig.savefig(fig_filename)
                print(fig_filename)
            else:
                for nb in stat_data[all_scenarios[0]].keys():
                    fig = plt.figure(figure_num)
                    figure_num+=1
                    ax = fig.add_subplot()
                    multiplier=1
                    min_y =  np.inf
                    max_y = -np.inf
                    for scenario in all_scenarios:
                        averages = np.array(stat_data[scenario][nb]["averages"])
                        if len(averages) == 0:
                            continue
                        xlabels = stat_data[scenario][nb]["xs"]
                        xs = np.arange(len(xlabels))
                        stds = np.array(stat_data[scenario][nb]["stds"])
                        min_y = min(min_y, min(averages-stds))
                        max_y = max(max_y, max(averages+stds))
                        offset = width*multiplier
                        multiplier+=1
                        ax.bar(xs + offset, averages, width, label=scenario, yerr=stds)
                    ax.set_xticks(xs + .5, xlabels)

                    if by == "param":
                        ax.set_xlabel(param_type_xlabels[self.param_type])
                    else:
                        ax.set_xlabel(by)
                    ax.set_ylabel(n)
                    range_y = max_y - min_y
                    if np.isclose(range_y, 0):
                        range_y = max_y*.5
                    ax.set_ylim(min_y-(range_y/2), max_y+(range_y/2))
                    plt.legend()
                    plt.title(f'{nb} {nb_name}\n{self.arbitrage_type} arbitrage, {self.param_type} {self.num_iters} iterations')
                    fig_filename = self.file_location + f"_{nb}{nb_name}_by{by}_{n}_bar.png"
                    fig.savefig(fig_filename)
                    print(fig_filename)

def make_paper_plots_bar(arb_types, param_type, num_iters=500, num_users=list(range(2,101)), scenarios_list = ['prof', 'prof_nokickback', 'mevs'],file_location='paper_data/', scale_values=[.25,.5,.75,1]):
    import json
    averages_data = {}
    for at in arb_types:
        averages_data[at]={}
        for s in scale_values:
            averages_data[at][str(s)] = json.load(open(f"{file_location}averages_data_{at}-{param_type}_{s}.json"))
        print("averages_data",at, averages_data[at].keys())

    averages_across_users = {}
    stds_across_users = {}
    scenario_labels = {'prof': 'PROF-Share', 'prof_nokickback': 'PROF', 'mevs': "MEV-Share"}
    def get_evens(ls):
        r = []
        for i in range(0,len(ls),2):
            r.append(ls[i])
        return r
    # num_users = get_evens(num_users)
    for scenario in scenarios_list:
        averages_across_users[scenario_labels[scenario]] = []
        stds_across_users[scenario_labels[scenario]] = []
    for d in scale_values:
        for at in averages_data:
            for scenario in scenarios_list:
                averages = np.average(averages_data[at][str(d)][scenario],axis=1)
                stds = np.std(averages_data[at][str(d)][scenario],axis=1)
                cis = confidence_interval(averages, stds, num_iters, alpha=.05, z=1.624)
                avg = np.average(averages[20:])
                std = np.std(averages[20:])
                averages_across_users[scenario_labels[scenario]].append(avg)
                stds_across_users[scenario_labels[scenario]].append(std)
    fig, axs = plt.subplots(2, sharex=False, layout='constrained', figure=(10,10))

    width = 0.25  # the width of the bars
    print(scale_values)
    vals = [[0.25,.5,.75,1],[2,4,8,16]]
    c=['C0','C1','C2']
    for i in range(2):
        ax = axs[i]
        v = vals[i]
        x = np.arange(len(v))
        multiplier = 0
        j = 0
        for attribute in averages_across_users.keys():
            if i == 1 and attribute == 'PROF':
                j+=1
                continue
            if i == 0:
                offset = width * multiplier
            else:
                offset = width * multiplier + (width/2)

            measurement = averages_across_users[attribute][i*len(v):(i+1)*len(v)]
            stds = stds_across_users[attribute][i*len(v):(i+1)*len(v)]
            print(len(measurement),measurement)
            print(len(stds),stds)
            print(len(x+offset),x+offset)
            rects = ax.bar(x + offset, measurement, width, label=attribute, yerr=stds, color=c[j])
            j+=1
            # ax.bar_label(rects, padding=3)
            multiplier += 1
        ax.set_xlabel('Demand Ratio')
        ax.set_ylabel('Average User Utility')
        scale_values_per = [str(s) for s in v]
        ax.set_xticks(x + width, scale_values_per)
        if i == 1:
            r = .000005
            t=[199.999984 ,199.999986 ,199.999988 ,199.99999 , 199.999992, 199.999994, 199.999996]
            ax.set_ylim(200-r*3, 200-r)
            print("get_yticks",ax.get_yticks())
            ax.set_yticks(t, [round(x,6) for x in t])
        else:
            r = .00002
            t = [199.99994, 199.99996, 199.99998, 200.,      200.00002]
            ax.set_ylim(200-r*3, 200+r/2)
            ax.set_yticks(t, [round(x,5) for x in t])
    labels = {.5: "0.5", 1 : "1", 2: "2", 4: "4"} 
    print("scale", scale_values)

    fig.legend(scenario_labels.values(), ncols=3)
    print(plt.yticks())
    fig.savefig("cexdexbar_3.png")
    # plt.show()
