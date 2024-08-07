import math
import json
import requests
import os
import time
import logging

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from eth_abi import abi
from bs4 import BeautifulSoup

BLOCK_NUMBER_MIN=15537572
BLOCK_NUMBER_MAX=18473541
BLOCK_DATA_FOLDER = "swap_data"
CONTRACT_DATA_FOLDER = "contract_data"
DEMAND_RATIO_DATA_FOLDER = "demand_ratio"
POOL_DATA_FOLDER = "pool_data"
TOKEN_DATA_FOLDER = "token_data"
FIGURE_FOLDER = "figure"
LOG_FOLDER = "log"
MEV_DATA_FOLDER = "mev_data"
INFURA_API_KEY = os.environ.get("INFURA_API_KEY")
INFURA_SECRET = os.environ.get("INFURA_SECRET")
INFURA_URL = f"https://mainnet.infura.io/v3/{INFURA_API_KEY}"
NONATOMIC_DATASET="1b3TafqpVzP88VhK9UT8oj49jsHee2yXt"

UNISWAPV3_TOPIC = {
    "create_pool": "0x783cca1c0412dd0d695e784568c96da2e9c22ff989357a2e8b1d9b2b4e6b7118",
    "create_pool_data_types":['int24', 'address'],
    "create_pool_topic_types": [None, ['address'], ['address'], ['uint24']],
    "pair_name_foo": lambda topics, token_names: f"{token_names(topics[1])}_{token_names(topics[2])}_{topics[3]/10000}",
    "pool_address_foo": lambda create_event: create_event[1],
    "swap": "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67",
    "swap_data_types": ['int256', 'int256', 'uint160', 'uint128', 'int24'],
    "swap_data_names": ['amount0','amount1','price','liq','tick'],
}

UNISWAPV2_TOPIC = {
    "create_pool": "0x0d3648bd0f6ba80134a33ba9275ac585d9d315f0ad8355cddefde31afa28d0e9",
    "create_pool_data_types":['address', "uint256"],
    "create_pool_topic_types": [None, ['address'], ['address']],
    "pair_name_foo": lambda topics, token_names: f"{token_names[topics[1].lower()]}_{token_names[topics[2].lower()]}",
    "pool_address_foo": lambda create_event: create_event[0],
    "swap": "0xd78ad95fa46c994b6551d0da85fc275fe613ce37657fb8d5e3d130840159d822",
    "swap_data_types": None, #TODO
    "swap_data_names": None, #TODO
}

POOL_FACTORY = {
    "sushiswap": {
        "factory": "0xbACEB8eC6b9355Dfc0269C18bac9d6E2Bdc29C4F", 
        "topic": UNISWAPV3_TOPIC
    },
    "uniswapv3": {
        "factory": "0x1F98431c8aD98523631AE4a59f267346ea31F984", 
        "topic": UNISWAPV3_TOPIC
    },
    # "uniswapv2": {
    #     "factory": "0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f", 
    #     "topic": UNISWAPV2_TOPIC
    # }
}

FILTERED_TAG = {True: "filtered", False: "unfiltered", None: "all"}
FIGURE_NUM = 0


HISTOGRAM_BINS = {
    "0-8+" : { 
        "not enough samples\nin block": lambda x: x==-1, 
        "0": lambda x: x==0, 
        "(0-0.25]": lambda x: x > 0 and x <= .25,
        "(0.25-0.50]": lambda x: x > .25 and x <= .5,
        "(0.50-0.75]": lambda x: x > .5 and x <= .75,
        "(0.75-1.0]": lambda x: x > .75 and x <= 1,
        "(1.0-2.0]": lambda x: x > 1 and x <= 2,
        "(2.0-4.0]": lambda x: x > 2 and x <= 4,
        "(4.0-8.0]": lambda x: x > 4 and x <= 8,
        "(8.0+)": lambda x: x > 8,
        "std=0": lambda x: x==-2,
    },
    "0-2+": { 
        "not enough samples\nin block": lambda x: x==-1, 
        "0": lambda x: x==0, 
        "(0-0.25]": lambda x: x > 0 and x <= .25,
        "(0.25-0.50]": lambda x: x > .25 and x <= .5,
        "(0.50-0.75]": lambda x: x > .5 and x <= .75,
        "(0.75-1.0]": lambda x: x > .75 and x <= 1,
        "(1.0-1.25]": lambda x: x > 1 and x <= 1.25,
        "(1.25-1.5]": lambda x: x > 1.25 and x <= 1.5,
        "(1.5-1.75]": lambda x: x > 1.5 and x <= 1.75,
        "(1.75-2]": lambda x: x > 1.75 and x <= 2,
        "(2.0+)": lambda x: x > 2,
        "std=0": lambda x: x==-2,
    }
}

def get_top_tokens(TOKEN_DATA_FOLDER=TOKEN_DATA_FOLDER):
    if not os.path.isdir(TOKEN_DATA_FOLDER):
        os.makedirs(TOKEN_DATA_FOLDER)
    num_pages = 2
    tokens_info = {"0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2": "WETH"}
    for i in range(num_pages):
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.4 Safari/605.1.15'}
        res = requests.get(f"https://etherscan.io/tokens?p={i+1}", headers=headers)
        if res.status_code != 200:
            exit(res.status_code)
        data = res.text 
        soup = BeautifulSoup(data, 'html.parser')
        with open(f"{TOKEN_DATA_FOLDER}/page{i+1}.html", "w") as f:
            f.write(soup.prettify())
        tokens = soup.find_all(id='ContentPlaceHolder1_tblErc20Tokens')[0].find_all('a')
        for tag in tokens:
            token_address = tag.get("href")
            if '/token/0x' in token_address:
                token_symbol = tag.find_all(class_="text-muted")[0].contents[0][1:-1]
                address = token_address.replace('/token/', '').lower()
                tokens_info[address] = token_symbol
                print(address, token_symbol)
    json.dump(tokens_info,open(f"{TOKEN_DATA_FOLDER}/top_tokens.json", "w"), indent=2)

def retry_with_block_range(make_request_foo, check_request_foo, parse_result_foo, data_dir, block_start, block_end, block_range=800):
    if not os.path.isdir(LOG_FOLDER):
        os.mkdir(LOG_FOLDER)
    if "/" in data_dir:
        tag = data_dir[:data_dir.find('/')]
    else:
        tag = data_dir
        data_dir += "/"
    logging.basicConfig(filename=f"{LOG_FOLDER}/{tag}.log",
                    format='%(asctime)s %(message)s',
                    filemode='a')
    
    progress_file = f"{LOG_FOLDER}/progress_{tag}"
    
    if os.path.exists(progress_file):
        with open(progress_file) as f:
            block_start=int(f.read())+1
        logging.warning(f"actually starting at {block_start} to {block_start + block_range}")
        logging.warning(f"delete {progress_file} to start downloading from scratch")
    if block_start > block_end:
        return

    orig_block_range=block_range
    num_range_large = 0
    num_range_ok = 0
    max_times_range = 5
    
    curr_time = time.time()
    block_curr = block_start
    block_next = block_start + block_range

    while block_curr == "earliest" or block_curr < block_end:
        res = make_request_foo(block_curr, block_next)
        success = check_request_foo(res)
        duration = time.time() - curr_time
        if success:
            df = parse_result_foo(res)
            filename=f'{data_dir}{block_curr}_{block_next}.pkl'
            df.to_pickle(filename)
            
            logging.info("%.2fs %s done, blocks left %d queries left: %d", [round(duration,2), filename,block_end-block_next,(block_end-block_next)/block_range])
            with open(progress_file, "w") as f:
                f.write(str(block_next))
            block_curr=block_next+1
            block_next+=block_range

            num_range_large=0
            num_range_ok+=1
            if num_range_ok>=max_times_range and block_range < orig_block_range:
                old_block_range = block_range
                block_range = max(orig_block_range, int(block_range*2))
                logging.info(f"adjusting range from {old_block_range} to {block_range}")
        else:
            num_range_ok=0
            num_range_large+=1
            old_range = f"{block_curr}-{block_next}"
            block_next = block_curr + int((block_next - block_curr)/2)
            logging.warning(f"retry with [{old_range}] -> [{block_curr}-{block_next}] length {block_next-block_curr}")
            if num_range_large>=max_times_range:
                old_block_range = block_range
                block_range = int(block_range/2)
                assert block_range > 0
                logging.warning(f"adjusting range from {old_block_range} to {block_range}")
        curr_time = time.time()
    logging.info(f"delete {progress_file} to start downloading from scratch")

def make_request_infura_foo(url, jsonpayload, headers, auth):
    assert len(jsonpayload['params']) == 1

    def make_request_infura(block_start, block_end):
        if block_start != "earliest":
            block_start = hex(block_start)
        if block_end != "latest":
            block_end = hex(block_end)
        jsonpayload['params'][0]['fromBlock'] = block_start
        jsonpayload['params'][0]['toBlock'] = block_end
        res = requests.post(url=url, json=jsonpayload, headers=headers, auth=auth)
        return res
        
    return make_request_infura

def check_request_infura(res):
    if res.status_code == 200:
        data = res.json()
        if 'error' in data:
            return False
        else:
            return True
    else:
        print(res.status_code, res.text)
        exit(res.status_code)

def parse_request_infura(res):
    data = res.json()
    df = pd.DataFrame.from_records(data["result"])
    df["blockNumber"] = df['blockNumber'].apply(int, base=16)
    df["logIndex"] = df['logIndex'].apply(int, base=16)
    df["transactionIndex"] = df['transactionIndex'].apply(int, base=16)
    return df

def download_create_pool_topic(POOL_DATA_FOLDER=POOL_DATA_FOLDER):
    assert INFURA_API_KEY != ""
    assert INFURA_SECRET != ""
    for protocol in POOL_FACTORY:
        if not os.path.isdir(f"{POOL_DATA_FOLDER}"):
            os.makedirs(f"{POOL_DATA_FOLDER}")
        factory_address = POOL_FACTORY[protocol]["factory"]
        topic = POOL_FACTORY[protocol]["topic"]["create_pool"]
        payload = {
            "jsonrpc": "2.0",
            "method": "eth_getLogs",
            "id":1,
            "params":[{
                "fromBlock": hex(BLOCK_NUMBER_MIN),
                "toBlock": hex(BLOCK_NUMBER_MAX), 
                "address": factory_address, 
                "topics": [topic]
                }
            ]
        }
        headers = {}
        auth = (INFURA_API_KEY, INFURA_SECRET)
        make_request_foo = make_request_infura_foo(INFURA_URL, payload, headers, auth)
        retry_with_block_range(make_request_foo, check_request_infura, parse_request_infura, f"{POOL_DATA_FOLDER}/create_pool_topic_{protocol}_", BLOCK_NUMBER_MIN, BLOCK_NUMBER_MAX, block_range=1000)

def parse_create_pool_topic(POOL_DATA_FOLDER=POOL_DATA_FOLDER, TOKEN_DATA_FOLDER=TOKEN_DATA_FOLDER):
    top_tokens = json.load(open(f"{TOKEN_DATA_FOLDER}/top_tokens.json"))

    def address_to_token(token):
        token=token.lower()
        if token in top_tokens:
            return top_tokens[token]
        return token
    pool_files = os.listdir(POOL_DATA_FOLDER)
    num_pools = len(pool_files)
    top_pools = {}
    for protocol in POOL_FACTORY:
        top_pools[protocol] = {}
    for pi in range(len(pool_files)):
        pool_data_filename = pool_files[pi]
        s1 = pool_data_filename.find('create_pool_topic_') + len('create_pool_topic_')
        e1 = pool_data_filename[s1:].find('_')+s1
        protocol = pool_data_filename[s1:e1]
        if protocol not in POOL_FACTORY:
            continue
        print(f"parse_create_pool_topic: {pi+1}/{num_pools} pool_data_filename", pool_data_filename)
        dataTypesArray = POOL_FACTORY[protocol]['topic']['create_pool_data_types']
        topicTypesArray = POOL_FACTORY[protocol]['topic']['create_pool_topic_types']
        factorydata = pd.read_pickle(f"{POOL_DATA_FOLDER}/{pool_data_filename}")
        def decodePoolAddress(data):
            params = abi.decode(dataTypesArray, bytes.fromhex(data[2:]))
            return POOL_FACTORY[protocol]['topic']['pool_address_foo'](params)
        def decodePairName(topics):
            all_topics = []
            for i in range(len(topics)):
                t = topics[i]
                if topicTypesArray[i] is None:
                    all_topics.append(None)
                else:
                    all_topics += abi.decode(topicTypesArray[i], bytes.fromhex(t[2:]))
            return POOL_FACTORY[protocol]['topic']['pair_name_foo'](all_topics, address_to_token)

        factorydata["poolAddress"] = factorydata["data"].apply(decodePoolAddress)
        factorydata["pairName"] = factorydata["topics"].apply(decodePairName)
        pools = factorydata[["poolAddress", "pairName"]].set_index("poolAddress")
        result = pools["pairName"].to_dict()
        top_pools[protocol] = {**top_pools[protocol], **result}
    json.dump(result, open(f"{POOL_DATA_FOLDER}/top_pools.json", "w"), indent=2)
    
def download_swap_topic(BLOCK_DATA_FOLDER=BLOCK_DATA_FOLDER):
    assert INFURA_API_KEY != ""
    assert INFURA_SECRET != ""
    if not os.path.isdir(f"{BLOCK_DATA_FOLDER}v3/"):
        os.makedirs(f"{BLOCK_DATA_FOLDER}v3/")
    # if not os.path.isdir(f"{BLOCK_DATA_FOLDER}v2/"):
    #     os.makedirs(f"{BLOCK_DATA_FOLDER}v2/")

    block_end= BLOCK_NUMBER_MAX
    block_start = BLOCK_NUMBER_MIN
    headers = {}
    auth = (INFURA_API_KEY, INFURA_SECRET)
    payloadv3 = {
        "jsonrpc": "2.0",
        "method": "eth_getLogs",
        "id" : 1,
        "params": [{
            "fromBlock": hex(block_start), 
            "toBlock": hex(block_end), 
            "topics": [UNISWAPV3_TOPIC["swap"]]
        }]
    }
    make_request_foov3 = make_request_infura_foo(INFURA_URL, payloadv3, headers, auth)
    retry_with_block_range(make_request_foov3, check_request_infura, parse_request_infura, f"{BLOCK_DATA_FOLDER}v3/swap_topicv3_", block_start, block_end, block_range=800)

    # payloadv2 = {
    #     "jsonrpc": "2.0",
    #     "method": "eth_getLogs",
    #     "id" : 1,
    #     "params": [{
    #         "fromBlock": hex(block_start), 
    #         "toBlock": hex(block_end), 
    #         "topics": [UNISWAPV2_TOPIC["swap"]]
    #     }]
    # }
    # make_request_foov2 = make_request_infura_foo(INFURA_URL, payloadv2, headers, auth)
    # retry_with_block_range(make_request_foov2, check_request_infura, parse_request_infura, f"{BLOCK_DATA_FOLDER}v2/swap_topicv2", block_start, block_end, block_range=800)

def combine_chunks(combine_df_foo, files_dir, files_list, tmp_dir, output_dir, size_chunk):
    if not os.path.isdir(LOG_FOLDER):
        os.mkdir(LOG_FOLDER)
    progress_file = f"{LOG_FOLDER}/progress_{files_dir}_combine_chunks"
    iter_num = 0
    
    if os.path.exists(progress_file):
        with open(progress_file) as f:
            last_iter=int(f.read())
        files_dir = f"{tmp_dir}/tmp_chunks{last_iter}"
        files_list = os.listdir(files_dir)
        iter_num = last_iter+1
        print(f"actually starting at {iter_num}")
        print(f"delete {progress_file} to start combining from scratch")    
    
    num_files = len(files_list)
    while num_files >= size_chunk:
        chunk_folder = f"{tmp_dir}/tmp_chunks{iter_num}"
        print(f"{iter_num}: combining {files_dir} into {chunk_folder}")
        if not os.path.isdir(chunk_folder):
            os.makedirs(chunk_folder)

        for i in range(0, num_files-size_chunk, size_chunk):
            combine_df_foo([f"{files_dir}/{f}" for f in files_list[i:i+size_chunk]], f"{int(i/size_chunk)}_", chunk_folder)
        if iter_num > 0:
            print(f"{iter_num}: removing {files_dir}")

            for f in files_list:
                os.remove(f"{files_dir}/{f}")
            os.rmdir(files_dir)

        with open(progress_file, "w") as f:
            f.write(str(iter_num))

        iter_num+=1
        files_list = os.listdir(chunk_folder)
        files_dir = chunk_folder
        num_files = len(files_list)

    combine_df_foo([f"{files_dir}/{f}" for f in files_list], "", output_dir)
    
def download_mev_dataset():
    if not os.path.isdir(MEV_DATA_FOLDER):
        os.makedirs(MEV_DATA_FOLDER)
    if not os.path.isdir(f"{MEV_DATA_FOLDER}_zeromev"):
        os.makedirs(f"{MEV_DATA_FOLDER}_zeromev")

    column_names = [
        "block_number", 
        "tx_index",
        "mev_type",
        "protocol",
        "user_loss_usd",
        "extractor_profit_usd",
        "user_swap_volume_usd",
        "user_swap_count",
        "extractor_swap_volume_usd",
        "extractor_swap_count",
        "imbalance",
        "address_from",
        "address_to",
        "arrival_time_us",
        "arrival_time_eu",
        "arrival_time_as"
    ]
    def make_request(block_start, block_end):
        params = {"block_number": block_start, 'count': block_end-block_start}
        headers = {'Accept':'application/json'}
        res = requests.get("https://data.zeromev.org/v1/mevBlock", params=params, headers=headers, timeout=10)
        time.sleep(1)
        return res
    
    def check_request(res):
        if res.status_code == 200:
            return True
        else:
            print("error", res.status_code, res.text)
            exit(res.status_code)
    
    def parse_request(res):
        df = pd.DataFrame(res.json(), columns=column_names)
        df = df[df['mev_type'].isin(['arb', 'frontrun', 'backrun'])]
        df = df.rename(columns={"block_number": "blockNumber", "tx_index": "transactionIndex"})
        return df
    
    retry_with_block_range(make_request, check_request, parse_request, f"{MEV_DATA_FOLDER}_zeromev/zeromev_", BLOCK_NUMBER_MIN, BLOCK_NUMBER_MAX, block_range=100)

    def combine_df(files_list, chunk_name, output_folder):
        dfs = []
        for f in files_list:
            file_df = pd.read_pickle(f)
            dfs.append(file_df)
        df = pd.concat(dfs, axis=0 , ignore_index=True, sort=False)
        df.to_pickle(f"{output_folder}/{chunk_name}zeromev.pkl")

    # combine_chunks(combine_df, f"{MEV_DATA_FOLDER}_zeromev", os.listdir(f"{MEV_DATA_FOLDER}_zeromev"), f"{MEV_DATA_FOLDER}_zeromev_tmp", MEV_DATA_FOLDER, 1000)
    combine_df([f"{MEV_DATA_FOLDER}_zeromev/{f}" for f in os.listdir(f"{MEV_DATA_FOLDER}_zeromev")], "", MEV_DATA_FOLDER)

    with open(f"cleanup_files.sh", "a") as f:
        f.write(f"rm -rf {MEV_DATA_FOLDER}_zeromev_tmp\n")
    print("run $bash cleanup_files.sh")

def parse_swap_topic(CONTRACT_DATA_FOLDER=CONTRACT_DATA_FOLDER):
    nonAtomicSwaps = pd.read_pickle('filteredSwapsWithPrice.pkl')
    mevtxs = pd.read_pickle(f"{MEV_DATA_FOLDER}/zeromev.pkl")
    mevtxs['mev_tx_tag'] = mevtxs['blockNumber'].astype(str) + "_" + mevtxs['transactionIndex'].astype(str)
    filter_block_min = min(min(nonAtomicSwaps['block']), min(mevtxs['blockNumber']))
    filter_block_max = max(max(nonAtomicSwaps['block']), max(mevtxs['blockNumber']))
    top_pools = json.load(open(f"{POOL_DATA_FOLDER}/top_pools.json"))
    include_unf = (filter_block_min != BLOCK_NUMBER_MIN) or (filter_block_max != BLOCK_NUMBER_MAX) 
    for t in FILTERED_TAG.values():
        if not include_unf and t == FILTERED_TAG[False]:
            continue
        if not os.path.isdir(f"{CONTRACT_DATA_FOLDER}_{t}_parse"):
            os.makedirs(f"{CONTRACT_DATA_FOLDER}_{t}_parse")

    progress_file = f"{LOG_FOLDER}/parse_swap_topic_progress"
    start = 0
    if os.path.isfile(progress_file):
        with open(progress_file) as f:
            start = int(f.read())+1

    block_files = os.listdir(f"{BLOCK_DATA_FOLDER}v3")
    num_block_files = len(block_files)
    typesArray = UNISWAPV3_TOPIC["swap_data_types"]
    column_names = UNISWAPV3_TOPIC["swap_data_names"]
    for fi in range(start, num_block_files):
        block_data_filename = block_files[fi]
        print(f"parse_swap_topicv3 {fi+1}/{num_block_files} parsing block_data_filename: {block_data_filename}")
        
        dfBlock = pd.read_pickle(f"{BLOCK_DATA_FOLDER}v3/{block_data_filename}")
        decodedData = dfBlock['data'].apply(lambda x: list(abi.decode(typesArray, bytes.fromhex(x[2:]))))
        decodedDataDF = pd.DataFrame(decodedData.tolist(), columns=column_names)

        dfBlock["poolName"] = dfBlock['address'].apply(lambda x: top_pools[x] if x in top_pools else x)
        dfBlockFinal = pd.concat([dfBlock, decodedDataDF], axis=1)
        dfBlockFinal['mev_tx_tag'] = dfBlockFinal['blockNumber'].astype(str) + "_" + dfBlockFinal['transactionIndex'].astype(str)

        dfBlockFiltered = dfBlockFinal[
            (dfBlockFinal['blockNumber'] >= filter_block_min) & 
            (dfBlockFinal['blockNumber'] <= filter_block_max) & 
            (dfBlockFinal['transactionHash'].isin(nonAtomicSwaps['hash']) == False) &
            ( dfBlockFinal['mev_tx_tag'].isin(mevtxs['mev_tx_tag']) == False)]

        for pool_name in dfBlockFinal["poolName"].unique():
            dfAllContract = dfBlockFinal[dfBlockFinal['poolName'] == pool_name]
            if len(dfAllContract) > 0:
                if not os.path.isdir(f"{CONTRACT_DATA_FOLDER}_{FILTERED_TAG[None]}_parse/{pool_name}"):
                    os.makedirs(f"{CONTRACT_DATA_FOLDER}_{FILTERED_TAG[None]}_parse/{pool_name}")
                print("parse_blocks_swaps",f"{pool_name}_{FILTERED_TAG[None]}_{fi}.pkl", len(dfAllContract))
                dfAllContract.to_pickle(f"{CONTRACT_DATA_FOLDER}_{FILTERED_TAG[None]}_parse/{pool_name}/{pool_name}_{FILTERED_TAG[None]}_{fi}.pkl")

        for pool_name in dfBlockFiltered["poolName"].unique():
            dfFilteredContract = dfBlockFiltered[dfBlockFiltered['poolName'] == pool_name]
            if len(dfFilteredContract) > 0:
                if not os.path.isdir(f"{CONTRACT_DATA_FOLDER}_{FILTERED_TAG[True]}_parse/{pool_name}"):
                    os.makedirs(f"{CONTRACT_DATA_FOLDER}_{FILTERED_TAG[True]}_parse/{pool_name}")
                print("parse_blocks_swaps",f"{pool_name}_{FILTERED_TAG[True]}_{fi}.pkl", len(dfFilteredContract))
                dfFilteredContract.to_pickle(f"{CONTRACT_DATA_FOLDER}_{FILTERED_TAG[True]}_parse/{pool_name}/{pool_name}_{FILTERED_TAG[True]}_{fi}.pkl")

        if include_unf:
            dfBlockUnf = dfBlockFinal[
                (dfBlockFinal['blockNumber'] >= filter_block_min) & 
                (dfBlockFinal['blockNumber'] <= filter_block_max)]
            for pool_name in dfBlockUnf["poolName"].unique():
                dfUnfilteredContract = dfBlockUnf[dfBlockUnf['poolName'] == pool_name]
                if len(dfUnfilteredContract) > 0:
                    if not os.path.isdir(f"{CONTRACT_DATA_FOLDER}_{FILTERED_TAG[False]}_parse/{pool_name}"):
                        os.makedirs(f"{CONTRACT_DATA_FOLDER}_{FILTERED_TAG[False]}_parse/{pool_name}")
                    print("parse_blocks_swaps",f"{pool_name}_{FILTERED_TAG[False]}_{fi}.pkl", len(dfUnfilteredContract))
                    dfUnfilteredContract.to_pickle(f"{CONTRACT_DATA_FOLDER}_{FILTERED_TAG[False]}_parse/{pool_name}/{pool_name}_{FILTERED_TAG[False]}_{fi}.pkl")
        with open(progress_file, "w") as f:
            f.write(str(fi))


def combine_contract_swaps(CONTRACT_DATA_FOLDER=CONTRACT_DATA_FOLDER):
    if not os.path.isdir(LOG_FOLDER):
        os.mkdir(LOG_FOLDER)
    for filter_tag in FILTERED_TAG.values():
        output_dir = f"{CONTRACT_DATA_FOLDER}_{filter_tag}"
        data_dir = f"{CONTRACT_DATA_FOLDER}_{filter_tag}_parse"
        progress_file = f"{LOG_FOLDER}/combine_contract_swaps_{filter_tag}_progress"
        if not os.path.isdir(data_dir):
            print(f"ignoring... {data_dir}_parse")
            continue
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        all_pool_names = os.listdir(data_dir)
        num_pools = len(all_pool_names)
        start_pi = 0
        if os.path.isfile(progress_file):
            with open(progress_file) as f:
                start_pi = int(f.read())
            
        for pi in range(start_pi, num_pools):
            pool_name = all_pool_names[pi]
            print(f"combine_contract_swaps {pi+1}/{num_pools} combining pool_name {pool_name} {filter_tag}")

            def combine_df(file_list, chunk_name, output_folder):
                df_list= []
                for f in file_list:
                    df = pd.read_pickle(f)
                    df_list.append(df)
                df_all = pd.concat(df_list, axis = 0, ignore_index=True, sort=False)
                df_all.to_pickle(f"{output_folder}/{chunk_name}{pool_name}_{filter_tag}.pkl")

            size_chunk=1000
            files_dir = f"{data_dir}/{pool_name}"
            files_list = os.listdir(files_dir)

            combine_df([f"{files_dir}/{f}" for f in files_list], "", output_dir)
            # combine_chunks(combine_df, files_dir, files_list, f"{data_dir}_{pool_name}_tmp", output_dir, size_chunk)

            with open(progress_file, "w") as f:
                f.write(str(pi+1))
            with open(f"cleanup_files.sh", "a") as f:
                f.write(f"rm -rf {files_dir}\n")

    print("run $bash cleanup_files.sh")

def calculate_demand_ratio(std_samples, hist_name="0-2+",filter_nonatomic=True,CONTRACT_DATA_FOLDER=CONTRACT_DATA_FOLDER,DEMAND_RATIO_DATA_FOLDER=DEMAND_RATIO_DATA_FOLDER):
    hist_filter = HISTOGRAM_BINS[hist_name]
    bin_names = list(hist_filter.keys())
    dtypes = {
        "address": 'str',
        "blockHash": 'str',
        "blockNumber":'int',
        "data":  'str',
        "logIndex": 'int',
        "removed": 'bool',
        "topics": 'object',
        "transactionHash": 'str',
        "transactionIndex": 'int',
        "poolName": 'str',
        "amount0": 'str',
        "amount1": 'object',
        "price": 'str',
        "liq": 'str',
        "tick":'str'
    }
    filter_tag = FILTERED_TAG[filter_nonatomic]
    if not os.path.isdir(DEMAND_RATIO_DATA_FOLDER):
        os.makedirs(DEMAND_RATIO_DATA_FOLDER)
    assert os.path.isdir(f"{CONTRACT_DATA_FOLDER}_{filter_tag}")
    pool_filenames = os.listdir(f"{CONTRACT_DATA_FOLDER}_{filter_tag}")
    num_pools = len(pool_filenames)

    num_swaps={}
    ratio_data={}
    for num_samples in std_samples:
        ratio_data[num_samples] = {}
        
    for pi in range(num_pools):
        filename = pool_filenames[pi]
        pool_name = filename[:-4]
        print(f"calculate_demand_ratio {pi+1}/{num_pools} calculate pool_name {pool_name} {filter_tag}")
        df = pd.read_pickle(f"{CONTRACT_DATA_FOLDER}_{filter_tag}/{filename}")
        df = df.astype(dtypes)
        df['topics'] = df['topics'].apply(sorted).astype(str)
        dup1=df.drop_duplicates(['blockNumber','transactionIndex', 'logIndex'])
        dup2=df.drop_duplicates()
        assert dup1.equals(dup2)
        df = dup2
        df['amount1'] = df["amount1"].apply(int)
        df['txVolume'] = df["amount1"].apply(abs)
        df = df.sort_values(by=['blockNumber','transactionIndex', 'logIndex'])
        block_group = df.groupby("blockNumber")
        dfBlock = pd.DataFrame()
        dfBlock['blockDemand'] = block_group["amount1"].sum().abs()
        num_swaps[pool_name] =  int(df['txVolume'].gt(0).count())

        def getRatio(volumeStd, blockDemand):
            if volumeStd == 0:
                return -2
            elif volumeStd > 0:
                return blockDemand / volumeStd
            elif pd.isnull(volumeStd):
                return -1
            else:
                raise ValueError

        def getBin(blockRatio):
            name = None
            for bin_name in hist_filter:
                filter_foo = hist_filter[bin_name]
                if filter_foo(blockRatio):
                    assert name is None
                    name = bin_name
            return name


        for num_samples in std_samples:
            df[f'volumdSTD{num_samples}'] = df['txVolume'].rolling(window=num_samples, center=True, min_periods=num_samples).std()
            block_group = df.groupby("blockNumber")
            dfBlock[f'volumdSTD{num_samples}'] = block_group[f"volumdSTD{num_samples}"].apply(lambda x: x.to_list()[int(len(x)/2)] if len(x) > 0 else np.nan)
            dfBlock[f'blockRatio{num_samples}'] = dfBlock.apply(lambda x: getRatio(x[f'volumdSTD{num_samples}'], x['blockDemand']), axis=1)
            dfBlock[f'poolBin{num_samples}'] = dfBlock[f'blockRatio{num_samples}'].apply(getBin)
            ratio_data[num_samples][pool_name] =  dfBlock.groupby([f'poolBin{num_samples}']).count().to_dict()
        
    for num_samples in std_samples:
        json.dump(ratio_data[num_samples], open(f"{DEMAND_RATIO_DATA_FOLDER}/{num_samples}stdsamples_{filter_tag}_{hist_name}.json", "w"), indent=2)
    json.dump(num_swaps, open(f"{DEMAND_RATIO_DATA_FOLDER}/num_swaps_{filter_tag}_{hist_name}.json", "w"), indent=2)
        

def plot_demand_ratio_top_pools(hist_name="0-2+", num_top=10, pools=None, filter_nonatomic=True, std_samples=7000, figure_tag=None, DEMAND_RATIO_DATA_FOLDER=DEMAND_RATIO_DATA_FOLDER, FIGURE_FOLDER=FIGURE_FOLDER):
    global FIGURE_NUM
    ratio_data = json.load(open(f"{DEMAND_RATIO_DATA_FOLDER}/{std_samples}stdsamples_{FILTERED_TAG[filter_nonatomic]}_{hist_name}.json"))

    num_swaps = ratio_data['num_swaps']
    pools_bins = ratio_data['pools_bins']
    pools_bins_percent = ratio_data['pools_bins_percent']
    pools_bins_percentz = ratio_data['pools_bins_percentz']
    hist_filter = HISTOGRAM_BINS[hist_name]
    bin_names = list(hist_filter.keys())

    if pools is None:
        pools_to_plot = num_swaps[-num_top:]
        print("pools_to_plot", pools_to_plot)
        pools_to_plot = [p[0] for p in pools_to_plot]
        print("pools =", pools_to_plot)
        if figure_tag:
            figure_tag = f"validblocks"
    else:
        pools_to_plot = pools
        num_top = len(pools_to_plot)

    width = 1/(num_top+2)

    # plot_vals = bin_names
    # x = np.arange(len(plot_vals))
    # figblock = plt.figure(FIGURE_NUM, layout='constrained', figsize=(12, 6))
    # FIGURE_NUM+=1
    # axblock = figblock.subplots()
    # multiplier = 0
    # for attribute in pools_to_plot:
    #     if attribute  in pools_bins:
    #         measurement = pools_bins[attribute]
    #         offset = width * multiplier
    #         rects = axblock.bar(x + offset, measurement, width, label=attribute)
    #         # ax.bar_label(rects, padding=3)
    #     multiplier += 1
    # axblock.set_xticks(x + .4, plot_vals)
    # axblock.set_xlabel("Demand Ratio")
    # axblock.set_ylabel("number of blocks")
    # axblock.legend()
    # axblock.set_title(f"number of blocks for Demand Ratio = (block demand)/(std volume in {std_samples} blocks) \n{FILTERED_TAG[filter_nonatomic]}, min samples {std_samples}")
    # multiplier = 0
    # figpercent = plt.figure(FIGURE_NUM, layout='constrained', figsize=(12, 6))
    # FIGURE_NUM+=1
    # axpercent = figpercent.subplots()
    # for attribute in pools_to_plot:
    #     if attribute in pools_bins_percent:
    #         measurement = pools_bins_percent[attribute]
    #         offset = width * multiplier
    #         rects = axpercent.bar(x + offset, measurement, width, label=attribute)#, yerr=stds_across_users[attribute])
    #     # ax.bar_label(rects, padding=3)
    #     multiplier += 1
    # axpercent.set_xticks(x + .4, plot_vals)
    # axpercent.set_xlabel("Demand Ratio")
    # axpercent.set_ylabel("percentage of blocks")
    # axpercent.legend()
    # axpercent.set_title(f"percentage of blocks for Demand Ratio= (block demand)/(std volume in {std_samples} blocks) \n{FILTERED_TAG[filter_nonatomic]}, min samples {std_samples}")

    multiplier = 1
    plot_vals = bin_names[2:-1]
    x = np.arange(len(plot_vals))
    figpercentz = plt.figure(FIGURE_NUM, layout='constrained', figsize=(10, 6))
    FIGURE_NUM+=1
    axpercentz = figpercentz.subplots()
    for attribute in pools_to_plot:
        if attribute in pools_bins_percentz:
            measurement = pools_bins_percentz[attribute]
            offset = width * multiplier
            rects = axpercentz.bar(x + offset, measurement[2:-1], width, label=attribute[:attribute.index(FILTERED_TAG[filter_nonatomic])-1])#, yerr=stds_across_users[attribute])
        # ax.bar_label(rects, padding=3)
        multiplier += 1
    axpercentz.set_xticks(x + .4, plot_vals)
    axpercentz.set_xlabel("Demand Ratio")
    axpercentz.set_ylabel("percentage of valid blocks")
    axpercentz.legend()
    axpercentz.set_title(f"percentage of valid blocks for Demand Ratio= (block demand)/({std_samples} swaps volatility)) \n{FILTERED_TAG[filter_nonatomic]}, min samples {std_samples}")
    if figure_tag is not None:
        # figblock.savefig(f"{FIGURE_FOLDER}/{std_samples}stdsamples_{FILTERED_TAG[filter_nonatomic]}_top{num_top}:{figure_tag}_numblocks.png")
        # figpercent.savefig(f"{FIGURE_FOLDER}/{std_samples}stdsamples_{FILTERED_TAG[filter_nonatomic]}_top{num_top}:{figure_tag}_percentblocks.png")
        figpercentz.savefig(f"{FIGURE_FOLDER}/{std_samples}stdsamples_{FILTERED_TAG[filter_nonatomic]}_top{num_top}:{figure_tag}_percentvalidblocks_{hist_name}.png")

def plot_demand_ratio_all_pools(hist_name="0-2+", std_samples=7000, filter_nonatomic=True, figure=False, DEMAND_RATIO_DATA_FOLDER=DEMAND_RATIO_DATA_FOLDER, FIGURE_FOLDER=FIGURE_FOLDER):
    global FIGURE_NUM
    ratio_data = json.load(open(f"{DEMAND_RATIO_DATA_FOLDER}/{std_samples}stdsamples_{FILTERED_TAG[filter_nonatomic]}_{hist_name}.json"))
    num_swaps = ratio_data['num_swaps']
    pools_bins = ratio_data['pools_bins']
    # block_ratio = ratio_data['block_ratio']
    hist_filter = HISTOGRAM_BINS[hist_name]
    bin_names = list(hist_filter.keys())

    num_pool_groups = 5
    width = 1/(num_pool_groups+2)

    pools_bins_percentz = {}
    for p in pools_bins:
        total_block = sum(pools_bins[p][2:-1])
        if total_block > 0:
            pools_bins_percentz[p] = [100*pools_bins[p][i]/total_block for i in range(len(pools_bins[p]))]

    num_pools = len(pools_bins_percentz.keys())
    pool_group_size = math.ceil(num_pools/num_pool_groups)
    num_swaps = list(filter(lambda x: x[0] in pools_bins_percentz.keys(), num_swaps))
    pool_group_size = math.ceil(num_pools/num_pool_groups)
    plot_vals = bin_names[2:-1]
    x = np.arange(len(plot_vals))


    figpercentzt = plt.figure(FIGURE_NUM, layout='constrained', figsize=(10, 6))
    FIGURE_NUM+=1
    multiplier = 1
    axpercentzt = figpercentzt.subplots()
    for i in range(num_pool_groups):
        best_pools = num_swaps[-pool_group_size*(i+1):]
        combined_bins = [pools_bins_percentz[pool[0]] for pool in best_pools]
        combined_bins = np.average(combined_bins, axis=0)[2:-1]
        print("std",std_samples, FILTERED_TAG[filter_nonatomic],f"top {int(100*(i+1)/num_pool_groups)}%",sum(combined_bins), "pools_bins_percentz",list(map(lambda x: round(x,2),combined_bins)))
        offset = width*multiplier
        axpercentzt.bar(x + offset, combined_bins, width, label=f"top {int(100*(i+1)/num_pool_groups)}%")
        multiplier+=1
        # print()
    axpercentzt.set_xticks(x + .4, plot_vals)
    axpercentzt.set_xlabel("Demand Ratio")
    axpercentzt.set_ylabel("Percentage of Blocks")
    axpercentzt.legend([f"top {int(100*(i+1)/num_pool_groups)}%" for i in range(num_pool_groups)])
    axpercentzt.set_title(f"percentage of valid blocks for Demand Ratio = (block demand)/({std_samples} swaps volatility) \n{FILTERED_TAG[filter_nonatomic]}")
    print("")


    figpercentz = plt.figure(FIGURE_NUM, layout='constrained', figsize=(10, 6))
    FIGURE_NUM+=1
    multiplier = 1
    axpercentz = figpercentz.subplots()
    for i in range(num_pool_groups):
        s = math.ceil((num_pool_groups-i-1)*num_pools/num_pool_groups)
        e = math.ceil((num_pool_groups-i)*num_pools/num_pool_groups)
        best_pools = num_swaps[s:e]
        combined_bins = [pools_bins_percentz[pool[0]] for pool in best_pools]
        summed = np.sum(combined_bins, axis=0)[2:-1]
        combined_bins = np.average(combined_bins, axis=0)[2:-1]
        print("std",std_samples, FILTERED_TAG[filter_nonatomic],f"top {int(100*i/num_pool_groups)} to {int(100*(i+1)/num_pool_groups)}%", sum(combined_bins), "pools_bins_percentz",list(map(lambda x: round(x,2),combined_bins)))
        # print("summed_bins", sum(summed), e-s, "\n", list(map(lambda x: round(x,2),summed)))
        offset = width*multiplier
        axpercentz.bar(x + offset, combined_bins, width, label=f"top {int(100*i/num_pool_groups)} to {int(100*(i+1)/num_pool_groups)}%")
        multiplier+=1
    axpercentz.set_xticks(x + .4, plot_vals)
    axpercentz.set_xlabel("Demand Ratio")
    axpercentz.set_ylabel("Percentage of Blocks")
    axpercentz.legend([f"top {int(100*i/num_pool_groups)} to {int(100*(i+1)/num_pool_groups)}%" for i in range(num_pool_groups)])
    axpercentz.set_title(f"percentage of valid blocks for Demand Ratio = (block demand)/({std_samples} swaps volatility) \n{FILTERED_TAG[filter_nonatomic]}")

    print("\n")


    figblock = plt.figure(FIGURE_NUM, layout='constrained', figsize=(12, 6))
    FIGURE_NUM+=1
    axblock = figblock.subplots()
    multiplier = 1
    for i in range(num_pool_groups):
        s = math.ceil((num_pool_groups-i-1)*num_pools/num_pool_groups)
        e = math.ceil((num_pool_groups-i)*num_pools/num_pool_groups)
        best_pools = num_swaps[s:e]
        # print(f"top {int(100*(i+1)/num_pool_groups)}%", best_pools)
        combined_bins = [pools_bins[pool[0]] for pool in best_pools]
        combined_bins = np.average(combined_bins, axis=0)[2:-1]

        print("std",std_samples, FILTERED_TAG[filter_nonatomic],f"top {int(100*i/num_pool_groups)} to {int(100*(i+1)/num_pool_groups)}%",sum(combined_bins),"pools_bins",list(map(lambda x: round(x,2),combined_bins)))
        offset = width*multiplier
        rects = axblock.bar(x + offset, combined_bins, width, label=f"top {int(100*i/num_pool_groups)} to {int(100*(i+1)/num_pool_groups)}%")
        # axblock.bar_label(rects, padding=3)
        multiplier+=1
    axblock.set_xticks(x + .4, plot_vals)
    axblock.set_xlabel("Demand Ratio")
    axblock.set_ylabel("number of blocks")
    axblock.legend([f"top {int(100*i/num_pool_groups)} to {int(100*(i+1)/num_pool_groups)}%" for i in range(num_pool_groups)])
    axblock.set_title(f"num blocks for Demand Ratio= (block demand)/(std volume in {std_samples} swaps) \n{FILTERED_TAG[filter_nonatomic]}")

    print()

    figblockt = plt.figure(FIGURE_NUM, layout='constrained', figsize=(12, 6))
    FIGURE_NUM+=1
    axblockt = figblockt.subplots()
    multiplier = 1
    for i in range(num_pool_groups):
        best_pools = num_swaps[-pool_group_size*(i+1):]
        combined_bins = [pools_bins[pool[0]] for pool in best_pools]
        combined_bins = np.average(combined_bins, axis=0)[2:-1]

        print("std",std_samples, FILTERED_TAG[filter_nonatomic],f"top {int(100*(i+1)/num_pool_groups)}%",sum(combined_bins),"pools_bins",list(map(lambda x: round(x,2),combined_bins)))
        offset = width*multiplier
        rects = axblockt.bar(x + offset, combined_bins, width, label=f"top {int(100*(i+1)/num_pool_groups)}%")
        # axblockt.bar_label(rects, padding=3)
        multiplier+=1
    axblockt.set_xticks(x + .4, plot_vals)
    axblockt.set_xlabel("Demand Ratio")
    axblockt.set_ylabel("number of blocks")
    axblockt.legend([f"top {int(100*(i+1)/num_pool_groups)}%" for i in range(num_pool_groups)])
    axblockt.set_title(f"num blocks for Demand Ratio= (block demand)/(std volume in {std_samples} swaps) \n{FILTERED_TAG[filter_nonatomic]}")
    print("\n\n")

    if figure:
        figblock.savefig( f"{FIGURE_FOLDER}/{std_samples}stdsamples_{FILTERED_TAG[filter_nonatomic]}_numblocks_{hist_name}_grp.png")
        figblockt.savefig(f"{FIGURE_FOLDER}/{std_samples}stdsamples_{FILTERED_TAG[filter_nonatomic]}_numblocks_{hist_name}_cum.png")
        figpercentz.savefig( f"{FIGURE_FOLDER}/{std_samples}stdsamples_{FILTERED_TAG[filter_nonatomic]}_percent_{hist_name}_grp.png")
        figpercentzt.savefig(f"{FIGURE_FOLDER}/{std_samples}stdsamples_{FILTERED_TAG[filter_nonatomic]}_percent_{hist_name}_cum.png")

def plot_demand_ratio_all_pools_cumulative(hist_name="0-2+",percentage=.2, std_samples=7000, filter_nonatomic=True, figure=False, DEMAND_RATIO_DATA_FOLDER=DEMAND_RATIO_DATA_FOLDER, FIGURE_FOLDER=FIGURE_FOLDER):
    global FIGURE_NUM
    ratio_data = json.load(open(f"{DEMAND_RATIO_DATA_FOLDER}/{std_samples}stdsamples_{FILTERED_TAG[filter_nonatomic]}_{hist_name}.json"))
    num_swaps = ratio_data['num_swaps']
    block_ratio = ratio_data['block_ratio']
    hist_filter = HISTOGRAM_BINS[hist_name]

    num_pools = len(num_swaps)
    num_swaps = sorted(num_swaps,key=lambda x: x[1])

    pool_group_size = int(num_pools*percentage)
    n_bins = 100

    best_pools = num_swaps[-pool_group_size:]
    combined_bins = [block_ratio[pool[0]] for pool in best_pools]
    combined_bins = np.concatenate(combined_bins, axis=0)
    s = sorted(combined_bins)
    l = 2
    print(np.median(combined_bins), max(combined_bins), len(combined_bins), len(set(combined_bins)), s[:l], s[int((len(s)-l)/2): int((len(s)+l)/2)], s[len(s)-l:])
    combined_bins = [round(x,2) if x < 1 else round(x,0) for x in combined_bins]
    s = sorted(combined_bins)
    print(np.median(combined_bins), max(combined_bins), len(combined_bins), len(set(combined_bins)), s[:l], s[int((len(s)-l)/2): int((len(s)+l)/2)], s[len(s)-l:])
    fig = plt.figure(FIGURE_NUM, layout='constrained', figsize=(10, 6))
    ax = fig.subplots()

    # Cumulative distributions.
    # ax.ecdf(combined_bins, label="top 20%")
    max_v = 800
    # max_r = math.ceil(math.log(max(combined_bins),2))+1
    max_r = math.ceil(math.log(max_v,2))+1

    n_bins = [n/n_bins for n in range(n_bins)] + [2**n for n in range(0, max_r, int(max_r/10))]
    # n_bins = [x for x in range(max_r)]
    # n_bins = 100
    n, bins, patches = ax.hist(combined_bins, n_bins, density=True, histtype="step", cumulative=True, label="Cumulative histogram")
    # print("n", n)
    # print("bin", bins)
    # print("patches", patches)
    ax.set_xlabel("Demand Ratio")
    ax.set_ylabel("Number of Blocks")
    ax.set_xscale("log")
    ax.set_title(f"top 20% pools Demand Ratio = (block demand)/(std volume in surrounding {std_samples} swaps) \n{FILTERED_TAG[filter_nonatomic]}")
 

    if figure:
        fig.savefig(f"{FIGURE_FOLDER}/{std_samples}stdsamples_{FILTERED_TAG[filter_nonatomic]}_{percentage}cumulative_{hist_name}.png")


    return


    plot_vals = bin_names
    x = np.arange(len(plot_vals))
    num_pool_groups = 5
    width = 1/(num_pool_groups+2)
    pool_group_size = math.ceil(num_pools/num_pool_groups)

    plot_vals = bin_names[2:-1]
    x = np.arange(len(plot_vals))
    figpercentz = plt.figure(FIGURE_NUM, layout='constrained', figsize=(10, 6))
    FIGURE_NUM+=1
    multiplier = 1
    axpercentz = figpercentz.subplots()
    for i in range(num_pool_groups):
        best_pools = num_swaps[-(pool_group_size*(i+1)):]
        combined_bins = [pools_bins_percentz[pool[0]] for pool in best_pools]
        combined_bins = np.average(combined_bins, axis=0)
        # print("pools_bins_percentz combined_bins",combined_bins[2:-1])
        offset = width*multiplier
        axpercentz.bar(x + offset, combined_bins[2:-1], width, label=f"top {int(100*(i+1)/num_pool_groups)}%")
        multiplier+=1
    axpercentz.set_xticks(x + .4, plot_vals)
    axpercentz.set_xlabel("Demand Ratio")
    axpercentz.set_ylabel("Percentage of Blocks")
    axpercentz.legend([f"top {int(100*(i+1)/num_pool_groups)}%" for i in range(num_pool_groups)])
    axpercentz.set_title(f"percentage of valid blocks for Demand Ratio = (block demand)/({std_samples} swaps volatility) \n{FILTERED_TAG[filter_nonatomic]}")
    if figure:
        # figblock.savefig(f"{FIGURE_FOLDER}/{std_samples}stdsamples_{FILTERED_TAG[filter_nonatomic]}_numblocks.png")
        # figpercent.savefig(f"{FIGURE_FOLDER}/{std_samples}stdsamples_{FILTERED_TAG[filter_nonatomic]}_percentblocks.png")
        figpercentz.savefig(f"{FIGURE_FOLDER}/{std_samples}stdsamples_{FILTERED_TAG[filter_nonatomic]}_percentvalidblocks.png")

def paper_plots_filtering(hist_name="0-2+", num_top=10, pools=None, std_samples=7000, figure=None, DEMAND_RATIO_DATA_FOLDER=DEMAND_RATIO_DATA_FOLDER, FIGURE_FOLDER=FIGURE_FOLDER):
    global FIGURE_NUM
    tags = [FILTERED_TAG[True], FILTERED_TAG[False]]
    ratio_data = {}
    num_swaps = {}
    pools_bins = {}
    for t in tags:
        ratio_data[t] = json.load(open(f"{DEMAND_RATIO_DATA_FOLDER}/{std_samples}stdsamples_{t}_{hist_name}.json"))
        num_swaps[t] = ratio_data[t]['num_swaps']
        pools_bins[t] = ratio_data[t]['pools_bins']
    # block_ratio = ratio_data['block_ratio']
    hist_filter = HISTOGRAM_BINS[hist_name]
    bin_names = list(hist_filter.keys())

    num_pool_groups = 5
    width = 1/(num_pool_groups+2)

    pools_bins_percentz = {}
    for t in tags:
        pools_bins_percentz[t] = {}
        for p in pools_bins[t]:
            total_block = sum(pools_bins[t][p][2:-1])
            if total_block > 0:
                pools_bins_percentz[t][p] = [100*pools_bins[t][p][i]/total_block for i in range(len(pools_bins[t][p]))]
        num_swaps[t] = list(filter(lambda x: x[0] in pools_bins_percentz[t].keys(), num_swaps[t]))

    num_pools = len(pools_bins_percentz[tags[0]].keys())
    pool_group_size = math.ceil(num_pools/num_pool_groups)
    pool_group_size = math.ceil(num_pools/num_pool_groups)
    plot_vals = bin_names[2:-1]
    x = np.arange(len(plot_vals))


    # figpercentz = plt.figure(FIGURE_NUM, layout='constrained', figsize=(12, 12))
    # figpercentz, axs = plt.subplots(len(tags), sharex=True)
    figpercentz = plt.figure(FIGURE_NUM, layout='constrained', figsize=(20, 6))
    figpercentz, axs = plt.subplots(1,2, sharey=True)
    FIGURE_NUM+=1
    for ti in range(len(tags)):
        t = tags[ti]
        axpercentz = axs[ti]
        multiplier = 1
        for i in range(num_pool_groups):
            s = math.ceil((num_pool_groups-i-1)*num_pools/num_pool_groups)
            e = math.ceil((num_pool_groups-i)*num_pools/num_pool_groups)
            best_pools = num_swaps[t][s:e]
            combined_bins = [pools_bins_percentz[t][pool[0]] for pool in best_pools]
            summed = np.sum(combined_bins, axis=0)[2:-1]
            combined_bins = np.average(combined_bins, axis=0)[2:-1]
            print("std",std_samples, t,f"top {int(100*i/num_pool_groups)} to {int(100*(i+1)/num_pool_groups)}%", sum(combined_bins), "pools_bins_percentz",list(map(lambda x: round(x,2),combined_bins)))
            # print("summed_bins", sum(summed), e-s, "\n", list(map(lambda x: round(x,2),summed)))
            offset = width*multiplier
            axpercentz.bar(x + offset, combined_bins, width, label=f"top {int(100*i/num_pool_groups)} to {int(100*(i+1)/num_pool_groups)}%")
            multiplier+=1
        # if ti == 1:
        #     axpercentz.set_xticks(x + .5, plot_vals)
        #     print("xticks", x + .4, plot_vals)
        axpercentz.set_xlabel("Demand Ratio")
        axpercentz.set_ylabel("Percentage of Blocks")
        axpercentz.set_title(f"{t} Swaps")

    figpercentz.legend([f"top {int(100*i/num_pool_groups)} to {int(100*(i+1)/num_pool_groups)}%" for i in range(num_pool_groups)])
    # figpercentz.set_title(f"percentage of valid blocks for Demand Ratio = (block demand)/({std_samples} swaps volatility) \n{FILTERED_TAG[filter_nonatomic]}")
    for ax in figpercentz.get_axes():
        ax.label_outer()

    print("\n")

    if figure:
        print(f"{FIGURE_FOLDER}/{std_samples}stdsamples_percent_{hist_name}_grp.png")
        figpercentz.savefig( f"{FIGURE_FOLDER}/{std_samples}stdsamples_percent_{hist_name}_grp.png")

def paper_plots_std_bar(hist_name="0-2+", num_top=10, pools=None, std_samples=[7000], filter_nonatomic=True, figure=None, DEMAND_RATIO_DATA_FOLDER=DEMAND_RATIO_DATA_FOLDER, FIGURE_FOLDER=FIGURE_FOLDER):
    global FIGURE_NUM
    tags = std_samples
    ratio_data = {}
    num_swaps = {}
    pools_bins = {}
    for t in tags:
        ratio_data[t] = json.load(open(f"{DEMAND_RATIO_DATA_FOLDER}/{t}stdsamples_{FILTERED_TAG[filter_nonatomic]}_{hist_name}.json"))
        num_swaps[t] = ratio_data[t]['num_swaps']
        pools_bins[t] = ratio_data[t]['pools_bins']
    # block_ratio = ratio_data['block_ratio']
    hist_filter = HISTOGRAM_BINS[hist_name]
    bin_names = list(hist_filter.keys())

    num_pool_groups = 5
    num_pool_groups = 1
    width = 1/(num_pool_groups+2)

    pools_bins_percentz = {}
    num_pools = {}
    pool_group_size = {}
    for t in tags:
        pools_bins_percentz[t] = {}
        for p in pools_bins[t]:
            total_block = sum(pools_bins[t][p][2:-1])
            if total_block > 0:
                pools_bins_percentz[t][p] = [100*pools_bins[t][p][i]/total_block for i in range(len(pools_bins[t][p]))]
        num_swaps[t] = list(filter(lambda x: x[0] in pools_bins_percentz[t].keys(), num_swaps[t]))

        num_pools[t] = len(pools_bins_percentz[t].keys())
        pool_group_size[t] = math.ceil(num_pools[t]/num_pool_groups)
    names = {"(0-0.25]":'<0.25',
        "(0.25-0.50]": '0.25-0.5',
        "(0.50-0.75]": '0.5-0.75',
        "(0.75-1.0]": '0.75-1',
        "(1.0-1.25]": '1-1.25',
        "(1.25-1.5]": '1.25-1.5',
        "(1.5-1.75]": '1.5-1.75',
        "(1.75-2]": '1.75-2',
        "(2.0+)": '2 <'
        }

    plot_vals = []
    for b in bin_names[2:-1]:
        plot_vals.append(names[b])
    x = np.arange(len(plot_vals))


    # figpercentz, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, layout='constrained', figsize=(14, 6))#sharey=True, sharex=True,
    # axs = [ax1, ax2, ax3, ax4]
    figpercentz, axs = plt.subplots(len(tags), sharex=False, layout='constrained')#, figsize=(10, 6))
    axs=[axs]
    pool_group_names = {
        "top 0 to 20%": "top 20% of pools",
        "top 0 to 33%": "Top 33% of pools",
        "top 20 to 40%": "20 to 40% of pools",
        "top 33 to 66%": "Middle 33 to 66% of pools",
        "top 40 to 60%": "40% to 60% of pools",
        "top 60 to 80%": "60 to 80% of pools",
        "top 66 to 100%": "Bottom 66 to 100% of pools",
        "top 80 to 100%": "80 to 100% of pools",
    }
    FIGURE_NUM+=1
    for ti in range(len(tags)):
        t = tags[ti]
        axpercentz = axs[ti]
        multiplier = 1
        print(t)
        for i in range(num_pool_groups):
            s = math.ceil((num_pool_groups-i-1)*num_pools[t]/num_pool_groups)
            e = math.ceil((num_pool_groups-i)*num_pools[t]/num_pool_groups)
            best_pools = num_swaps[t][s:e]
            # print("best_pools", best_pools)
            combined_bins_or = [pools_bins_percentz[t][pool[0]] for pool in best_pools]
            # summed = np.sum(combined_bins, axis=0)[2:-1]
            combined_bins_avg = np.average(combined_bins_or, axis=0)
            combined_bins_std = np.std(combined_bins_or, axis=0)
            combined_bins_avg = combined_bins_avg[2:-1]
            combined_bins_std = combined_bins_std[2:-1]
            print(combined_bins_avg)
            print(combined_bins_std)
            # print("std",FILTERED_TAG[filter_nonatomic], t,f"top {int(100*i/num_pool_groups)} to {int(100*(i+1)/num_pool_groups)}%", sum(combined_bins), "pools_bins_percentz",list(map(lambda x: round(x,2),combined_bins)))
            # print("summed_bins", sum(summed), e-s, "\n", list(map(lambda x: round(x,2),summed)))
            offset = width*multiplier

            axpercentz.bar(x + offset, combined_bins_avg, width, label=[FILTERED_TAG[f]])#, label=pool_group_names[f"top {int(100*i/num_pool_groups)} to {int(100*(i+1)/num_pool_groups)}%"])#,yerr=combined_bins_std)
            multiplier+=1
        # if ti == 1:
        axpercentz.set_xlabel("Demand Ratio")
        axpercentz.set_xticks(x + .4, plot_vals)
        axpercentz.set_ylabel("Percentage of Blocks")
        axpercentz.set_title(f"{t:,} Swap Variability of Filtered Dataset")

    # figpercentz.legend([pool_group_names[f"top {int(100*i/num_pool_groups)} to {int(100*(i+1)/num_pool_groups)}%"] for i in range(num_pool_groups)],loc=(0.75,0.8))
    # figpercentz.legend()#[pool_group_names[f"top {int(100*i/num_pool_groups)} to {int(100*(i+1)/num_pool_groups)}%"] for i in range(num_pool_groups)],loc=(.75,.8))

    # for ax in figpercentz.get_axes():
    #     ax.label_outer()

    print("\n")

    if figure:
        # print(f"{FIGURE_FOLDER}/{FILTERED_TAG[filter_nonatomic]}_percent_{hist_name}_grp.png")
        # figpercentz.savefig( f"{FIGURE_FOLDER}/{FILTERED_TAG[filter_nonatomic]}_percent_{hist_name}_grp.png")
        l = [str(std_samples[0]), "","no", "4"]
        figpercentz.savefig( f"{FIGURE_FOLDER}/{FILTERED_TAG[filter_nonatomic]}_data{l[len(std_samples)-1]}.png")
        print(f"{FIGURE_FOLDER}/{FILTERED_TAG[filter_nonatomic]}_data{l[len(std_samples)-1]}_3.png")

def paper_plots_std_bar2(hist_name="0-2+", num_top=10, pools=None, std_samples=7000, filter_nonatomic=[True, False], figure=None, DEMAND_RATIO_DATA_FOLDER=DEMAND_RATIO_DATA_FOLDER, FIGURE_FOLDER=FIGURE_FOLDER):
    global FIGURE_NUM
    tags = filter_nonatomic
    ratio_data = {}
    num_swaps = {}
    pools_bins = {}
    for t in tags:
        ratio_data[t] = json.load(open(f"{DEMAND_RATIO_DATA_FOLDER}/{std_samples}stdsamples_{FILTERED_TAG[t]}_{hist_name}.json"))
        num_swaps[t] = ratio_data[t]['num_swaps']
        pools_bins[t] = ratio_data[t]['pools_bins']
    # block_ratio = ratio_data['block_ratio']
    hist_filter = HISTOGRAM_BINS[hist_name]
    bin_names = list(hist_filter.keys())

    num_pool_groups = 5
    num_pool_groups = 2
    width = 1/(num_pool_groups+2)

    pools_bins_percentz = {}
    num_pools = {}
    pool_group_size = {}
    for t in tags:
        pools_bins_percentz[t] = {}
        for p in pools_bins[t]:
            total_block = sum(pools_bins[t][p][2:-1])
            if total_block > 0:
                pools_bins_percentz[t][p] = [100*pools_bins[t][p][i]/total_block for i in range(len(pools_bins[t][p]))]
        num_swaps[t] = list(filter(lambda x: x[0] in pools_bins_percentz[t].keys(), num_swaps[t]))

        num_pools[t] = len(pools_bins_percentz[t].keys())
        pool_group_size[t] = math.ceil(num_pools[t]/num_pool_groups)
    names = {"(0-0.25]":'<0.25',
        "(0.25-0.50]": '0.25-0.5',
        "(0.50-0.75]": '0.5-0.75',
        "(0.75-1.0]": '0.75-1',
        "(1.0-1.25]": '1-1.25',
        "(1.25-1.5]": '1.25-1.5',
        "(1.5-1.75]": '1.5-1.75',
        "(1.75-2]": '1.75-2',
        "(2.0+)": '2 <'
        }

    plot_vals = []
    for b in bin_names[2:-1]:
        plot_vals.append(names[b])
    x = np.arange(len(plot_vals))


    # figpercentz, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, layout='constrained', figsize=(14, 6))#sharey=True, sharex=True,
    # axs = [ax1, ax2, ax3, ax4]
    # figpercentz, axs = plt.subplots(len(tags), sharex=False, layout='constrained')#, figsize=(10, 6))
    figpercentz, axpercentz = plt.subplots(layout='constrained')
    FIGURE_NUM+=1
    multiplier=1
    for i in range(len(tags)):
        t = tags[i]
        best_pools = num_swaps[t]
        combined_bins_or = [pools_bins_percentz[t][pool[0]] for pool in best_pools]
        combined_bins_avg = np.average(combined_bins_or, axis=0)
        combined_bins_avg = combined_bins_avg[2:-1]
        print("cb", combined_bins_avg)
        offset = width*multiplier
        axpercentz.bar(x + offset, combined_bins_avg, width, label=FILTERED_TAG[t])#, label=pool_group_names[f"top {int(100*i/num_pool_groups)} to {int(100*(i+1)/num_pool_groups)}%"])#,yerr=combined_bins_std)
        multiplier+=1
    axpercentz.set_xlabel("Demand Ratio")
    axpercentz.set_xticks(x + .4, plot_vals)
    axpercentz.set_ylabel("Percentage of Blocks")
    axpercentz.set_title(f"{std_samples} Swap Variability")
    figpercentz.legend(loc=(.8,.8))

    print("\n")

    if figure:
        # print(f"{FIGURE_FOLDER}/{FILTERED_TAG[filter_nonatomic]}_percent_{hist_name}_grp.png")
        # figpercentz.savefig( f"{FIGURE_FOLDER}/{FILTERED_TAG[filter_nonatomic]}_percent_{hist_name}_grp.png")
        # l = [str(std_samples[0]), "","no", "4"]
        figpercentz.savefig( f"{FIGURE_FOLDER}/filtered_data{std_samples}.png")
        print(f"{FIGURE_FOLDER}/filtered_data{std_samples}.png")
        # print(f"{FIGURE_FOLDER}/{FILTERED_TAG[filter_nonatomic]}_data{l[len(std_samples)-1]}_3.png")

def paper_plots_std(hist_name="0-2+", num_top=10, pools=None, std_samples=[7000], filter_nonatomic=True, figure=None, DEMAND_RATIO_DATA_FOLDER=DEMAND_RATIO_DATA_FOLDER, FIGURE_FOLDER=FIGURE_FOLDER):
    global FIGURE_NUM
    tags = std_samples
    ratio_data = {}
    num_swaps = {}
    pools_bins = {}
    for t in tags:
        ratio_data[t] = json.load(open(f"{DEMAND_RATIO_DATA_FOLDER}/{t}stdsamples_{FILTERED_TAG[filter_nonatomic]}_{hist_name}.json"))
        num_swaps[t] = ratio_data[t]['num_swaps']
        pools_bins[t] = ratio_data[t]['pools_bins']
    # block_ratio = ratio_data['block_ratio']
    hist_filter = HISTOGRAM_BINS[hist_name]
    bin_names = list(hist_filter.keys())

    num_pool_groups = 5
    num_pool_groups = 3
    width = 1/(num_pool_groups+.5)

    pools_bins_percentz = {}
    num_pools = {}
    pool_group_size = {}
    for t in tags:
        pools_bins_percentz[t] = {}
        for p in pools_bins[t]:
            total_block = sum(pools_bins[t][p][2:-1])
            if total_block > 0:
                pools_bins_percentz[t][p] = [100*pools_bins[t][p][i]/total_block for i in range(len(pools_bins[t][p]))]
        num_swaps[t] = list(filter(lambda x: x[0] in pools_bins_percentz[t].keys(), num_swaps[t]))

        num_pools[t] = len(pools_bins_percentz[t].keys())
        pool_group_size[t] = math.ceil(num_pools[t]/num_pool_groups)
    names = {"(0-0.25]":'<0.25',
        "(0.25-0.50]": '0.25-0.5',
        "(0.50-0.75]": '0.5-0.75',
        "(0.75-1.0]": '0.75-1',
        "(1.0-1.25]": '1-1.25',
        "(1.25-1.5]": '1.25-1.5',
        "(1.5-1.75]": '1.5-1.75',
        "(1.75-2]": '1.75-2',
        "(2.0+)": '2 <'
        }

    plot_vals = []
    for b in bin_names[2:-1]:
        plot_vals.append(names[b])
    x = np.arange(len(plot_vals))
    scale_values = [0, .25, .5, .75, 1, 1.25, 1.5, 1.75, 2]

    # figpercentz, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, layout='constrained', figsize=(14, 6))#sharey=True, sharex=True,
    # axs = [ax1, ax2, ax3, ax4]
    figpercentz, axs = plt.subplots(len(tags), sharex=False, layout='constrained', figsize=(7, 7))
    pool_group_names = {
        "top 0 to 20%": "top 20% of pools",
        "top 0 to 33%": "Top 33% of pools",
        "top 20 to 40%": "20 to 40% of pools",
        "top 33 to 66%": "Middle 33 to 66% of pools",
        "top 40 to 60%": "40% to 60% of pools",
        "top 60 to 80%": "60 to 80% of pools",
        "top 66 to 100%": "Bottom 66 to 100% of pools",
        "top 80 to 100%": "80 to 100% of pools",
    }
    FIGURE_NUM+=1
    for ti in range(len(tags)):
        t = tags[ti]
        axpercentz = axs[ti]
        multiplier = .25
        for i in range(num_pool_groups):
            s = math.ceil((num_pool_groups-i-1)*num_pools[t]/num_pool_groups)
            e = math.ceil((num_pool_groups-i)*num_pools[t]/num_pool_groups)
            best_pools = num_swaps[t][s:e]
            print("best_pools", best_pools)
            combined_bins_or = [pools_bins_percentz[t][pool[0]] for pool in best_pools]
            # summed = np.sum(combined_bins, axis=0)[2:-1]
            combined_bins_avg = np.average(combined_bins_or, axis=0)
            combined_bins_std = np.std(combined_bins_or, axis=0)
            print("cb", combined_bins_avg)
            combined_bins_avg = combined_bins_avg[2:-1]
            combined_bins_std = combined_bins_std[2:-1]
            # print("std",FILTERED_TAG[filter_nonatomic], t,f"top {int(100*i/num_pool_groups)} to {int(100*(i+1)/num_pool_groups)}%", sum(combined_bins), "pools_bins_percentz",list(map(lambda x: round(x,2),combined_bins)))
            # print("summed_bins", sum(summed), e-s, "\n", list(map(lambda x: round(x,2),summed)))
            offset = width*multiplier
            print(len(scale_values), len(combined_bins_avg))
            axpercentz.errorbar(scale_values, combined_bins_avg, label=pool_group_names[f"top {int(100*i/num_pool_groups)} to {int(100*(i+1)/num_pool_groups)}%"],yerr=combined_bins_std)
            # axpercentz.bar(x + offset, combined_bins_avg, width, label=pool_group_names[f"top {int(100*i/num_pool_groups)} to {int(100*(i+1)/num_pool_groups)}%"],yerr=combined_bins_std)
            multiplier+=1
        # if ti == 1:
        axpercentz.set_xlabel("Demand Ratio")
        # axpercentz.set_xticks(x + .4, plot_vals)
        axpercentz.set_xticks(scale_values, plot_vals)
        axpercentz.set_ylabel("Percentage of Blocks")
        axpercentz.set_title(f"{t:,} Swap Variability")

    # figpercentz.legend([pool_group_names[f"top {int(100*i/num_pool_groups)} to {int(100*(i+1)/num_pool_groups)}%"] for i in range(num_pool_groups)],loc=(0.75,0.8))
    figpercentz.legend([pool_group_names[f"top {int(100*i/num_pool_groups)} to {int(100*(i+1)/num_pool_groups)}%"] for i in range(num_pool_groups)],loc=(.6,.85))

    # for ax in figpercentz.get_axes():
    #     ax.label_outer()

    print("\n")

    if figure:
        # print(f"{FIGURE_FOLDER}/{FILTERED_TAG[filter_nonatomic]}_percent_{hist_name}_grp.png")
        # figpercentz.savefig( f"{FIGURE_FOLDER}/{FILTERED_TAG[filter_nonatomic]}_percent_{hist_name}_grp.png")
        l = [str(std_samples[0]), "","no", "2"]
        figpercentz.savefig( f"{FIGURE_FOLDER}/{FILTERED_TAG[filter_nonatomic]}_data{l[len(std_samples)-1]}_line.png")

def paper_plots_std2(hist_name="0-2+", num_top=10, pools=None, std_samples=7000, filter_nonatomic=[True, False], figure=None, DEMAND_RATIO_DATA_FOLDER=DEMAND_RATIO_DATA_FOLDER, FIGURE_FOLDER=FIGURE_FOLDER):
    global FIGURE_NUM
    tags = filter_nonatomic
    ratio_data = {}
    num_swaps = {}
    pools_bins = {}
    for t in tags:
        ratio_data[t] = json.load(open(f"{DEMAND_RATIO_DATA_FOLDER}/{std_samples}stdsamples_{FILTERED_TAG[t]}_{hist_name}.json"))
        num_swaps[t] = ratio_data[t]['num_swaps']
        pools_bins[t] = ratio_data[t]['pools_bins']
    # block_ratio = ratio_data['block_ratio']
    hist_filter = HISTOGRAM_BINS[hist_name]
    bin_names = list(hist_filter.keys())

    num_pool_groups = 5
    num_pool_groups = 3
    width = 1/(num_pool_groups+.5)

    pools_bins_percentz = {}
    num_pools = {}
    pool_group_size = {}
    for t in tags:
        pools_bins_percentz[t] = {}
        for p in pools_bins[t]:
            total_block = sum(pools_bins[t][p][2:-1])
            if total_block > 0:
                pools_bins_percentz[t][p] = [100*pools_bins[t][p][i]/total_block for i in range(len(pools_bins[t][p]))]
        num_swaps[t] = list(filter(lambda x: x[0] in pools_bins_percentz[t].keys(), num_swaps[t]))

        num_pools[t] = len(pools_bins_percentz[t].keys())
        pool_group_size[t] = math.ceil(num_pools[t]/num_pool_groups)
    names = {"(0-0.25]":'<0.25',
        "(0.25-0.50]": '0.25-0.5',
        "(0.50-0.75]": '0.5-0.75',
        "(0.75-1.0]": '0.75-1',
        "(1.0-1.25]": '1-1.25',
        "(1.25-1.5]": '1.25-1.5',
        "(1.5-1.75]": '1.5-1.75',
        "(1.75-2]": '1.75-2',
        "(2.0+)": '2 <'
        }

    plot_vals = []
    for b in bin_names[2:-1]:
        plot_vals.append(names[b])
    x = np.arange(len(plot_vals))
    scale_values = [0, .25, .5, .75, 1, 1.25, 1.5, 1.75, 2]

    # figpercentz, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, layout='constrained', figsize=(14, 6))#sharey=True, sharex=True,
    # axs = [ax1, ax2, ax3, ax4]
    figpercentz, axs = plt.subplots(len(tags), sharex=False, layout='constrained', figsize=(7, 7))
    pool_group_names = {
        "top 0 to 20%": "top 20% of pools",
        "top 0 to 33%": "Top 33% of pools",
        "top 20 to 40%": "20 to 40% of pools",
        "top 33 to 66%": "Middle 33 to 66% of pools",
        "top 40 to 60%": "40% to 60% of pools",
        "top 60 to 80%": "60 to 80% of pools",
        "top 66 to 100%": "Bottom 66 to 100% of pools",
        "top 80 to 100%": "80 to 100% of pools",
    }
    FIGURE_NUM+=1
    for ti in range(len(tags)):
        t = tags[ti]
        axpercentz = axs[ti]
        multiplier = .25
        for i in range(num_pool_groups):
            s = math.ceil((num_pool_groups-i-1)*num_pools[t]/num_pool_groups)
            e = math.ceil((num_pool_groups-i)*num_pools[t]/num_pool_groups)
            best_pools = num_swaps[t][s:e]
            print("best_pools", best_pools)
            combined_bins_or = [pools_bins_percentz[t][pool[0]] for pool in best_pools]
            # summed = np.sum(combined_bins, axis=0)[2:-1]
            combined_bins_avg = np.average(combined_bins_or, axis=0)
            combined_bins_std = np.std(combined_bins_or, axis=0)
            print("cb", combined_bins_avg)
            combined_bins_avg = combined_bins_avg[2:-1]
            combined_bins_std = combined_bins_std[2:-1]
            # print("std",FILTERED_TAG[filter_nonatomic], t,f"top {int(100*i/num_pool_groups)} to {int(100*(i+1)/num_pool_groups)}%", sum(combined_bins), "pools_bins_percentz",list(map(lambda x: round(x,2),combined_bins)))
            # print("summed_bins", sum(summed), e-s, "\n", list(map(lambda x: round(x,2),summed)))
            offset = width*multiplier
            print(len(scale_values), len(combined_bins_avg))
            axpercentz.errorbar(scale_values, combined_bins_avg, label=pool_group_names[f"top {int(100*i/num_pool_groups)} to {int(100*(i+1)/num_pool_groups)}%"],yerr=combined_bins_std)
            # axpercentz.bar(x + offset, combined_bins_avg, width, label=pool_group_names[f"top {int(100*i/num_pool_groups)} to {int(100*(i+1)/num_pool_groups)}%"],yerr=combined_bins_std)
            multiplier+=1
        # if ti == 1:
        axpercentz.set_xlabel("Demand Ratio")
        # axpercentz.set_xticks(x + .4, plot_vals)
        axpercentz.set_xticks(scale_values, plot_vals)
        axpercentz.set_ylabel("Percentage of Blocks")
        axpercentz.set_title(f"{t:,} Swap Variability")

    # figpercentz.legend([pool_group_names[f"top {int(100*i/num_pool_groups)} to {int(100*(i+1)/num_pool_groups)}%"] for i in range(num_pool_groups)],loc=(0.75,0.8))
    figpercentz.legend([pool_group_names[f"top {int(100*i/num_pool_groups)} to {int(100*(i+1)/num_pool_groups)}%"] for i in range(num_pool_groups)],loc=(.6,.85))

    # for ax in figpercentz.get_axes():
    #     ax.label_outer()

    print("\n")

    if figure:
        # print(f"{FIGURE_FOLDER}/{FILTERED_TAG[filter_nonatomic]}_percent_{hist_name}_grp.png")
        # figpercentz.savefig( f"{FIGURE_FOLDER}/{FILTERED_TAG[filter_nonatomic]}_percent_{hist_name}_grp.png")
        l = [str(std_samples[0]), "","no", "2"]
        figpercentz.savefig( f"{FIGURE_FOLDER}/{FILTERED_TAG[filter_nonatomic]}_data{l[len(std_samples)-1]}_line.png")


if __name__ == "__main__":
    get_top_tokens()
    download_create_pool_topic()
    parse_create_pool_topic()

    download_mev_dataset()

    download_swap_topic()
    parse_swap_topic()
    combine_contract_swaps()

    calculate_demand_ratio(std_samples=[100,500,1000,7000], filter_nonatomic=True)
    calculate_demand_ratio(std_samples=[100,500,1000,7000], filter_nonatomic=None)
    calculate_demand_ratio(std_samples=[100,500,1000,7000], filter_nonatomic=False)


    plot_demand_ratio_all_pools(std_samples=100,filter_nonatomic=True, figure=True)
    plot_demand_ratio_all_pools(std_samples=500,filter_nonatomic=True, figure=True)
    plot_demand_ratio_all_pools(std_samples=1000,filter_nonatomic=True, figure=True)
    plot_demand_ratio_all_pools(std_samples=7000,filter_nonatomic=True, figure=True)