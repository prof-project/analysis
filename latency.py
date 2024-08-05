import requests
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import os.path
import json
import time
import random
import sys
import math
import csv
import matplotlib.colors as colors
import matplotlib.ticker as mtick
np.set_printoptions(threshold=sys.maxsize)


# Better data available at https://beaconcha.in/block/19287164 but no api
FIRST_SLOT = 4700567 # Sep-15-2022 08:33:47 AM +UTC (first on flashbots dashboard: https://docs.flashbots.net/flashbots-data/dashboard)
LAST_SLOT = 8837998 # Apr-11-2024 11:59:59 PM +UTC
FIRST_BLOCK = 15537940 # Sep-15-2022 08:33:47 AM +UTC (first on flashbots dashboard: https://docs.flashbots.net/flashbots-data/dashboard)
LAST_BLOCK = 19635809 # Apr-11-2024 11:59:59 PM +UTC
BASE_LATENCY_MS = 6.24809794
LATENCY_PER_GAS_MS = 5.26026644e-6
# AVERAGE_TX_FEE = 0.00344 # ETH during the above period of 100 days, via source https://bitinfocharts.com/ethereum/

    #divnorm = colors.TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=30.5)
DAYS = 100
DAY_TOTAL_SLOT = 7200
TOTAL_SLOT = DAY_TOTAL_SLOT * DAYS
DAY_SAMPLE_SLOT = 100 # max 650

MS_PER_SLOT = 12000
MS_PER_PLOT = 250 #1000 # Max latency in latency-penalty figures
MS_PER_INTERVAL = 2 #10 # Lantency interval in figures
NUM_INTERVALS = MS_PER_PLOT // MS_PER_INTERVAL + 1

MIN_INCLUSION_RATE = 70
MAX_INCLUSION_RATE = 99
INCLUSION_RATE_INTERVAL = 1

MIN_PROF_GAS = int(2.1e4)
MAX_PROF_GAS = int(3e6)
GAS_INTERVAL = int(2.5e4)


RELAYS = {
    "Flashbots": "https://boost-relay.flashbots.net", 
    "Ultra Sound": "https://relay.ultrasound.money", 
    "Agnostic": "https://agnostic-relay.net",
    #"bloXroute Regulated": "https://bloxroute.regulated.blxrbdn.com",
    #"bloXroute Max Profit": "https://bloxroute.max-profit.blxrbdn.com", # missing data, API not working properly
    }

# FLASHBOTS_URL_WIN = "https://api.allorigins.win/raw?url=https://boost-relay.flashbots.net/relay/v1/data/bidtraces/proposer_payload_delivered?cursor="
def get_url_winning_bid(relay, slot):
    return "https://api.allorigins.win/raw?url=" + RELAYS[relay] + "/relay/v1/data/bidtraces/proposer_payload_delivered?cursor=" + str(slot)

# FLASHBOTS_URL_BID = "https://api.allorigins.win/raw?url=https://boost-relay.flashbots.net/relay/v1/data/bidtraces/builder_blocks_received?slot="
def get_url_all_bids(relay, slot):
    return "https://api.allorigins.win/raw?url=" + RELAYS[relay] + "/relay/v1/data/bidtraces/builder_blocks_received?slot=" + str(slot)

PATH = "./dataset/"
BASE_FEE_FILE = PATH + "base_fee_data_big_query.csv"

def get_path_relay(relay):
    return PATH + relay + "/"

def get_path_winning_bid(relay):
    return get_path_relay(relay) + "win/"

def get_path_all_bids(relay):
    return get_path_relay(relay) + "bid/"

def mkdir(path):
    if os.path.isdir(path) == False:
        os.mkdir(path)

def mkalldirs():
    mkdir(PATH)
    for relay in RELAYS:
        mkdir(get_path_relay(relay))
        mkdir(get_path_winning_bid(relay))
        mkdir(get_path_all_bids(relay))

def fetch_base_fee():
    with open(BASE_FEE_FILE, mode ='r') as file:
        csvFile = csv.reader(file)
        next(csvFile, None)
        base_fee_all = dict()
        for line in csvFile:
            base_fee_all[int(line[0])] = int(line[2])
        return base_fee_all

def api_request(url):
    try:
        response = requests.get(url).json()
    except requests.exceptions.RequestException as err:
        raise Exception(err)
    except json.JSONDecodeError as err:
        raise Exception(err)
    else:
        return response

def write_to_file(file, content):
    if os.path.isfile(file) == False:
        f = open(file, "w")
        f.write(json.dumps(content))
    else:
        print(file, "already exists")

    
def fetch_winning_bids(relay):
    last_slot = LAST_SLOT
    winning_slots = set()
    while last_slot >= LAST_SLOT - TOTAL_SLOT:
        file = get_path_winning_bid(relay) + str(last_slot)
        if os.path.isfile(file):
            f = open(file, "r")
            content = f.read()
            winning_bids = json.loads(content)
            if len(winning_bids) == 0:
                break

            for slot in winning_bids:
                winning_slots.add(slot['slot'])

            last_slot = int(winning_bids[-1]['slot']) - 1
            print(file, "already exists, last winning slot", relay, last_slot)
            continue
        
        try:
            response = api_request(get_url_winning_bid(relay, last_slot))
        except Exception as err:
            print("[Error] last winning slot", relay, last_slot)
            print(err)
            time.sleep(2)
            continue
        else:
            write_to_file(file, response)
            if len(response) == 0:
                break
            
            for slot in winning_bids:
                winning_slots.add(slot['slot'])
            
            last_slot = int(response[-1]['slot']) - 1
            print("finish winning slot", relay, last_slot + 1)
    return winning_slots


def select_random_slots(relay, day_sample_slot, base_fee_all):
    win_file = []
    win_file_index = 0
    cur_slot = LAST_SLOT + 1
    cur_day = 1
    slots = []
    block_nums = []
    bids = []
    day_slots = []
    day_block_nums = []
    day_bids = []
    while cur_slot > LAST_SLOT - DAYS * DAY_TOTAL_SLOT:
        if win_file_index == len(win_file):
            f = open(get_path_winning_bid(relay) + str(cur_slot - 1), "r")
            content = f.read()
            win_file = json.loads(content)
            win_file_index = 0
            if len(win_file) == 0:
                break
       
        while win_file_index < len(win_file):
            cur_slot = int(win_file[win_file_index]['slot'])
            cur_block_num = int(win_file[win_file_index]['block_number'])
            if cur_slot <= LAST_SLOT - cur_day * DAY_TOTAL_SLOT:
                slots.append(day_slots)
                block_nums.append(day_block_nums)
                bids.append(day_bids)
                day_slots = []
                day_block_nums = []
                day_bids = []
                cur_day += 1
                if cur_day > DAYS:
                    break
            day_slots.append(cur_slot)
            day_block_nums.append(cur_block_num)
            day_bids.append(int(win_file[win_file_index]['value']))
            win_file_index += 1

    sample_slots = []
    winning_bids = {}
    base_fee = {}
    for day in range(0, DAYS):
        day_slots = []
        if day >= len(slots) or day_sample_slot > len(slots[day]):
            break
        for j in range(0, day_sample_slot):
            seed_str = "select slot " + str(j) + " for day " + str(day)
            random.seed(seed_str)
            select_index = random.randrange(0, len(slots[day]))
            day_slots.append(slots[day][select_index])
            winning_bids[slots[day][select_index]] = bids[day][select_index]
            base_fee[slots[day][select_index]] = base_fee_all[block_nums[day][select_index]]
            slots[day][select_index] = slots[day][-1]
            slots[day].pop()
            bids[day][select_index] = bids[day][-1]
            bids[day].pop()
        sample_slots.append(day_slots)
    
    return sample_slots, winning_bids, base_fee

def fetch_builder_bids(relay, sample_slots):
    for i in range(0, DAYS):
        if i >= len(sample_slots):
            break
        for slot in sample_slots[i]:
            file = get_path_all_bids(relay) + str(slot)
            if os.path.isfile(file):
                print(file, "already exists")
                continue
            while True:
                try:
                    response = api_request(get_url_all_bids(relay, slot))
                except Exception as err:
                    print("[Error] current bid slot", relay, slot)
                    print(err)
                    time.sleep(2)
                else:
                    write_to_file(file, response)
                    print("finish bid slot", relay, slot)
                    break

def process_bids(relay, sample_slots, winning_bids):
    data = np.zeros((DAY_SAMPLE_SLOT * min(DAYS, len(sample_slots)), MS_PER_PLOT + 1), dtype = float)
    for day in range(0, min(DAYS, len(sample_slots))):
        for slot_id in range(0, DAY_SAMPLE_SLOT):
            row = day * DAY_SAMPLE_SLOT + slot_id

            slot = sample_slots[day][slot_id]
            file = get_path_all_bids(relay) + str(slot)
            f = open(file, "r")
            content = f.read()
            bids = json.loads(content)

            penalty = np.zeros((len(bids), 2))
            for (i, bid) in enumerate(bids):
                penalty[i][0] = (winning_bids[slot] - int(bid['value'])) / 1e18
                penalty[i][1] = bid['timestamp_ms']
            penalty = penalty[penalty[:, 1].argsort()]

            winning_timestamp = 0
            for i in range(1, len(bids)):
                penalty[i][0] = min(penalty[i][0], penalty[i - 1][0])
                if penalty[i][0] <= 0:
                    winning_timestamp = penalty[i][1]
                    break
            for i in range(0, len(bids)):
                if winning_timestamp - penalty[i][1] > MS_PER_PLOT:
                    continue
                # print(slot, winning_timestamp, penalty[i][1])
                # data[row][int((winning_timestamp - penalty[i][1] - 1) // MS_PER_INTERVAL + 1)] = penalty[i][0]
                data[row][int(winning_timestamp - penalty[i][1])] = penalty[i][0]
                if penalty[i][0] <= 0:
                    break

            for i in range(1, MS_PER_PLOT + 1):
                data[row][i] = max(data[row][i], data[row][i - 1])
    return data

def plot_latency_penalty(relay, data, list_percentiles):
    plt.figure(figsize=(5, 3)) 
    #fig = plt.subplot(len(list_sample_slots), len(list_days), i * len(list_days) + j + 1)
    #plt.title(str(day_sample_slots) + ' slots/day, ' + str(days) + ' days')
    plt.grid(linestyle = '--')
    plt.xlabel("Latency (ms)")
    plt.ylabel("Penalty (ETH)")
    #ax = plt.gca()
    #ax.set_ylim([0, 3e11])
    
    header = ['latency', 'mean', 'sem']
    csv_data =[] 
    x = np.arange(NUM_INTERVALS) * MS_PER_INTERVAL
    csv_data.append(x)

    extracted_data = data[:, ::MS_PER_INTERVAL]
    
    y_mean = np.mean(extracted_data, axis = 0)
    csv_data.append(y_mean)
    y_mean_sem = np.std(extracted_data, axis = 0) / math.sqrt(len(extracted_data))
    csv_data.append(y_mean_sem)
    #plt.yscale("log")  
    plt.errorbar(x, y_mean, yerr = y_mean_sem, label = 'mean with SEM')

    for percentile in list_percentiles:
        y = np.percentile(extracted_data, percentile, axis = 0)
        csv_data.append(y)
        header.append(str(percentile) + '_percentile')
        #plt.yscale("log")  
        plt.plot(x, y, label = str(percentile) + " percentile")

    plt.legend()
    
    csv_data = map(list, zip(*csv_data))
    f = open("csv_latency_vs_penalty_" + relay + ".csv", 'w')
    write = csv.writer(f)
    write.writerow(header)
    write.writerows(csv_data)
           
    plt.axvline(x = 85, color = 'black', label = 'axvline - full height')
    #plt.axvline(x = BASE_LATENCY_MS, color = 'black', label = 'axvline - full height')
    plt.savefig("fig_latency_vs_penalty_" + relay + ".pdf", format="pdf", bbox_inches="tight")
    plt.show()

def plot_inclusionrate_feeoverhead(relay, data, sample_slots, base_fee):
    plt.figure(figsize=(5, 5))
    fig, ax = plt.subplots()
    #plt.xlabel("Average tx latency (ms)")
    plt.xlabel("Gas used in PROF bundle (" + r'$g$'+ ")")
    plt.ylabel("Inclusion likelihood (" + r'$\alpha$' + ")")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    
    header = ['proftxgas', 'inclusionrate', 'feeoverhead']
    csv_data =[]
    # x = np.arange(MS_AVG_LATENCY_INTERVAL, MS_MAX_AVG_TX_LATENCY + 1, MS_AVG_LATENCY_INTERVAL)
    x = np.arange(MIN_PROF_GAS, MAX_PROF_GAS + 1, GAS_INTERVAL)
    y = np.arange(MIN_INCLUSION_RATE, MAX_INCLUSION_RATE + 1, INCLUSION_RATE_INTERVAL)
    X, Y = np.meshgrid(x, y)
    z = np.zeros((len(y), len(x)), dtype = float)
   
    overhead_data = np.zeros((DAY_SAMPLE_SLOT * min(DAYS, len(sample_slots)), len(x)), dtype = float)
    for day in range(0, min(DAYS, len(sample_slots))):
        for slot_id in range(0, DAY_SAMPLE_SLOT):
            row = day * DAY_SAMPLE_SLOT + slot_id
            slot = sample_slots[day][slot_id]
            for j, gas in enumerate(x):
                latency = int(BASE_LATENCY_MS + LATENCY_PER_GAS_MS * gas)
                overhead_data[row][j] = data[row][latency] * 1e18 / (base_fee[slot] * gas)

    for i, percentile in enumerate(y):
        z[i] = np.percentile(overhead_data, percentile, axis = 0)
        for j, gas in enumerate(x):
            csv_data.append([gas, percentile / 100, z[i][j]])



    bounds = np.linspace(0, 0.25, 11)
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    pcm = ax.pcolormesh(X, Y, z, norm=norm, cmap='YlGnBu', shading='auto')
    cb = plt.colorbar(pcm, label="Transaction-fee overhead (" + r'$\gamma$' + ")", extend='max', orientation='horizontal')

    #divnorm = colors.TwoSlopeNorm(vmin=0, vcenter=0.1, vmax=0.22)
    #pcm = ax.pcolormesh(X, Y, z, norm=divnorm, cmap='tab20', shading='auto')
    #cb = plt.colorbar(pcm, label="Transaction-fee overhead " + r'$\gamma$', orientation="horizontal")
    #ticks = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]
    cb.set_ticks(bounds)
    cb.set_ticklabels(['{:.0%}'.format(x) for x in bounds])
    
    f = open("csv_inclusionrate_vs_feeoverhead_" + relay + ".csv", 'w')
    write = csv.writer(f)
    write.writerow(header)
    write.writerows(csv_data)
            
    plt.axvline(x = 0.75e6, linewidth = 1, color = 'black', label = 'axvline - full height')
    plt.savefig("fig_inclusionrate_vs_feeoverhead_" + relay + ".pdf", format="pdf", bbox_inches="tight")
    plt.show()

mkalldirs()

base_fee_all = fetch_base_fee()

num_winning_slots = []
winning_slots = set()
for relay in RELAYS:
    slots = fetch_winning_bids(relay)
    winning_slots.update(slots)
    num_winning_slots.append(len(slots))

    sample_slots, winning_bids, base_fee = select_random_slots(relay, DAY_SAMPLE_SLOT, base_fee_all)
    fetch_builder_bids(relay, sample_slots)
    data = process_bids(relay, sample_slots, winning_bids)
    
    list_percentiles = [50, 75, 90, 95]
    plot_latency_penalty(relay, data, list_percentiles)
    plot_inclusionrate_feeoverhead(relay, data, sample_slots, base_fee) 
   
    print("Done with", relay)

#for i, relay in enumerate(RELAYS):
#    print(relay, num_winning_slots[i], num_winning_slots[i] / DAYS / DAY_TOTAL_SLOT)
#print(len(winning_slots), len(winning_slots) / DAYS / DAY_TOTAL_SLOT)


